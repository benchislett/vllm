# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send concurrent chats and validate DFlash hidden-state traces."""

import argparse
import hashlib
import json
import math
import random
import statistics
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
from safetensors import safe_open

MODEL = "nvidia/smart-panda-mtp-graft-NVFP4-20260701"
DRAFT_MODEL = "nvidia/nano3.5_smart_panda_dflash"

CONVERSATIONS: list[list[dict[str, str]]] = [
    [
        {"role": "system", "content": "Be concise, accurate, and practical."},
        {
            "role": "user",
            "content": "Give five concrete ways to make a Python service observable.",
        },
    ],
    [
        {"role": "user", "content": "What is speculative decoding?"},
        {
            "role": "assistant",
            "content": "It uses a faster draft model and a target verifier.",
        },
        {
            "role": "user",
            "content": "Expand that into six numbered implementation concerns.",
        },
    ],
    [
        {"role": "system", "content": "Explain technical ideas plainly."},
        {
            "role": "user",
            "content": "Compare optimistic concurrency and database locking.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Write a short checklist for reviewing a CUDA kernel change.",
        },
    ],
    [
        {"role": "user", "content": "I have a slow API endpoint."},
        {
            "role": "assistant",
            "content": "Start by separating queueing, compute, and I/O latency.",
        },
        {
            "role": "user",
            "content": "Turn that into a seven-step debugging plan.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Explain prefix caching and list four correctness pitfalls.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Suggest six edge cases for testing a streaming tokenizer.",
        },
    ],
    [
        {"role": "system", "content": "Answer as a careful systems engineer."},
        {
            "role": "user",
            "content": "How would you investigate intermittent GPU OOM failures?",
        },
    ],
]


@dataclass
class RequestResult:
    index: int
    response_id: str
    completion_tokens: int
    latency_seconds: float
    dump_path: str
    engine_request_id: str


@dataclass
class TraceSummary:
    response_id: str
    engine_request_id: str
    dump_path: str
    completion_tokens: int
    prompt_tokens: int
    final_input_token_ids: list[int]
    generated_tokens_per_step: list[int]
    accepted_proposals_per_step: list[int]
    rejected_proposals_per_step: list[int]
    hidden_state_shapes: dict[str, list[int]]
    target_prediction_tokens: int
    mean_target_entropy_bits: float = math.nan
    median_target_entropy_bits: float = math.nan
    target_top1_hit_rate_pct: float = math.nan
    mean_target_top1_probability: float = math.nan
    median_target_top1_probability: float = math.nan

    @property
    def mean_acceptance_length(self) -> float:
        if not self.generated_tokens_per_step:
            return 1.0
        return statistics.fmean(self.generated_tokens_per_step)


@dataclass
class ValidatedTrace:
    summary: TraceSummary
    target_hidden_states: torch.Tensor
    target_token_ids: torch.Tensor


@dataclass
class TargetTokenMetrics:
    entropy_bits: torch.Tensor
    top1_hits: torch.Tensor
    top1_probabilities: torch.Tensor


class TargetOutputProjection:
    """Apply a target model's output norm and language-model head."""

    _E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    def __init__(
        self,
        norm_weight: torch.Tensor,
        norm_epsilon: float,
        head_weight: torch.Tensor,
        head_scale: torch.Tensor | None = None,
        head_global_scale: torch.Tensor | None = None,
        head_bias: torch.Tensor | None = None,
        device: str = "cpu",
        vocab_chunk_size: int = 4096,
    ) -> None:
        self.device = torch.device(device)
        self.norm_weight = norm_weight.to(self.device)
        self.norm_epsilon = norm_epsilon
        self.head_weight = head_weight.to(self.device)
        self.head_scale = head_scale.to(self.device) if head_scale is not None else None
        self.head_global_scale = (
            head_global_scale.to(self.device) if head_global_scale is not None else None
        )
        self.head_bias = head_bias.to(self.device) if head_bias is not None else None
        self.vocab_chunk_size = vocab_chunk_size

        hidden_size = self.norm_weight.numel()
        if self.head_weight.dtype == torch.uint8:
            if self.head_scale is None or self.head_global_scale is None:
                raise ValueError("packed NVFP4 LM head is missing its scales")
            if self.head_weight.shape[1] * 2 != hidden_size:
                raise ValueError("packed LM-head width does not match output norm")
            if hidden_size % self.head_scale.shape[1] != 0:
                raise ValueError("invalid NVFP4 LM-head group-scale shape")
            self.group_size = hidden_size // self.head_scale.shape[1]
        else:
            if self.head_weight.shape[1] != hidden_size:
                raise ValueError("LM-head width does not match output norm")
            self.group_size = 0

    @property
    def vocab_size(self) -> int:
        return self.head_weight.shape[0]

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        draft_model: str,
        revision: str | None,
        draft_revision: str | None,
        device: str,
        vocab_chunk_size: int,
    ) -> "TargetOutputProjection":
        config = _load_hf_json(model, "config.json", revision)
        draft_config = _load_hf_json(draft_model, "config.json", draft_revision)
        _validate_final_auxiliary_layer(config, draft_config)
        index = _load_hf_json(model, "model.safetensors.index.json", revision)
        weight_map = index["weight_map"]
        norm_name = _find_weight_name(
            weight_map,
            (
                "backbone.norm_f.weight",
                "model.norm.weight",
                "model.norm_f.weight",
                "transformer.ln_f.weight",
            ),
            "output norm",
        )
        head_name = _find_weight_name(
            weight_map, ("lm_head.weight", "output.weight"), "LM head"
        )
        names = [norm_name, head_name]
        head_scale_name = f"{head_name}_scale"
        head_global_scale_name = f"{head_name}_scale_2"
        head_bias_name = head_name.removesuffix("weight") + "bias"
        for optional_name in (
            head_scale_name,
            head_global_scale_name,
            head_bias_name,
        ):
            if optional_name in weight_map:
                names.append(optional_name)
        tensors = _load_hf_tensors(model, revision, weight_map, names)

        epsilon = next(
            (
                float(config[name])
                for name in (
                    "layer_norm_epsilon",
                    "rms_norm_eps",
                    "norm_epsilon",
                    "norm_eps",
                )
                if name in config
            ),
            None,
        )
        if epsilon is None:
            raise ValueError("could not find output-norm epsilon in target config")
        return cls(
            norm_weight=tensors[norm_name],
            norm_epsilon=epsilon,
            head_weight=tensors[head_name],
            head_scale=tensors.get(head_scale_name),
            head_global_scale=tensors.get(head_global_scale_name),
            head_bias=tensors.get(head_bias_name),
            device=device,
            vocab_chunk_size=vocab_chunk_size,
        )

    def _head_weight_chunk(self, start: int, end: int) -> torch.Tensor:
        weight = self.head_weight[start:end]
        if weight.dtype != torch.uint8:
            return weight.to(torch.bfloat16)

        low = weight & 0x0F
        high = weight >> 4
        nibbles = torch.stack((low, high), dim=-1).reshape(end - start, -1)
        magnitudes = nibbles & 0x07
        values = torch.tensor(
            self._E2M1_VALUES, dtype=torch.float32, device=self.device
        )[magnitudes.long()]
        values = torch.where(nibbles & 0x08 != 0, -values, values)
        scales = (
            self.head_scale[start:end].float().repeat_interleave(self.group_size, dim=1)
        )
        values.mul_(scales)
        values.mul_(self.head_global_scale.float())
        return values.to(torch.bfloat16)

    def score(
        self, hidden_states: torch.Tensor, target_token_ids: torch.Tensor
    ) -> TargetTokenMetrics:
        if len(hidden_states) != len(target_token_ids):
            raise ValueError(
                "hidden states and target token IDs must have equal length"
            )
        if torch.any(target_token_ids < 0) or torch.any(
            target_token_ids >= self.vocab_size
        ):
            raise ValueError("target token ID is outside the LM-head vocabulary")

        hidden = hidden_states.to(self.device, dtype=torch.float32)
        variance = hidden.square().mean(dim=-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.norm_epsilon)
        hidden = (hidden * self.norm_weight.float()).to(torch.bfloat16)
        num_tokens = len(hidden)
        max_logits = torch.full(
            (num_tokens,), -torch.inf, dtype=torch.float32, device=self.device
        )
        top1_ids = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        scaled_exp_sum = torch.zeros_like(max_logits)
        scaled_logit_exp_sum = torch.zeros_like(max_logits)

        for start in range(0, self.vocab_size, self.vocab_chunk_size):
            end = min(start + self.vocab_chunk_size, self.vocab_size)
            weight = self._head_weight_chunk(start, end)
            logits = torch.matmul(hidden, weight.T).float()
            if self.head_bias is not None:
                logits.add_(self.head_bias[start:end].float())
            chunk_max, chunk_top1 = logits.max(dim=-1)
            new_max = torch.maximum(max_logits, chunk_max)
            old_factor = torch.exp(max_logits - new_max)
            chunk_exp = torch.exp(logits - new_max[:, None])
            scaled_exp_sum = scaled_exp_sum * old_factor + chunk_exp.sum(dim=-1)
            scaled_logit_exp_sum = scaled_logit_exp_sum * old_factor + (
                chunk_exp * logits
            ).sum(dim=-1)
            replace_top1 = chunk_max > max_logits
            top1_ids = torch.where(replace_top1, chunk_top1 + start, top1_ids)
            max_logits = new_max

        log_partition = max_logits + torch.log(scaled_exp_sum)
        expected_logit = scaled_logit_exp_sum / scaled_exp_sum
        entropy_bits = (log_partition - expected_logit) / math.log(2)
        top1_logprobs = max_logits - log_partition
        if (
            not torch.isfinite(entropy_bits).all()
            or not torch.isfinite(top1_logprobs).all()
        ):
            raise AssertionError("target output metrics contain non-finite values")
        target_token_ids = target_token_ids.to(self.device)
        return TargetTokenMetrics(
            entropy_bits=entropy_bits.cpu(),
            top1_hits=top1_ids.eq(target_token_ids).cpu(),
            top1_probabilities=top1_logprobs.exp().cpu(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress a DFlash server and validate its one-file traces."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--draft-model", default=DRAFT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--dump-dir", type=Path, default=Path("/tmp/dflash-traces"))
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--dump-timeout", type=float, default=120.0)
    parser.add_argument("--api-key")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--target-metrics-device",
        default="auto",
        help="Device for target output projection (auto, cpu, or cuda).",
    )
    parser.add_argument("--logits-vocab-chunk-size", type=int, default=4096)
    return parser.parse_args()


def _resolve_hf_file(model: str, filename: str, revision: str | None) -> Path:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, filename, revision=revision))


def _load_hf_json(model: str, filename: str, revision: str | None) -> dict[str, Any]:
    with _resolve_hf_file(model, filename, revision).open() as file:
        return json.load(file)


def _find_weight_name(
    weight_map: dict[str, str], candidates: tuple[str, ...], description: str
) -> str:
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    raise ValueError(f"could not find {description}; tried {', '.join(candidates)}")


def _load_hf_tensors(
    model: str,
    revision: str | None,
    weight_map: dict[str, str],
    names: list[str],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    names_by_file: dict[str, list[str]] = {}
    for name in names:
        names_by_file.setdefault(weight_map[name], []).append(name)
    for filename, file_names in names_by_file.items():
        path = _resolve_hf_file(model, filename, revision)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in file_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def _validate_final_auxiliary_layer(
    target_config: dict[str, Any], draft_config: dict[str, Any]
) -> None:
    num_hidden_layers = int(target_config["num_hidden_layers"])
    auxiliary_layer_ids = draft_config.get("eagle_aux_hidden_state_layer_ids")
    if not isinstance(auxiliary_layer_ids, list) or not auxiliary_layer_ids:
        raise ValueError(
            "draft config does not declare eagle_aux_hidden_state_layer_ids"
        )
    if int(auxiliary_layer_ids[-1]) != num_hidden_layers:
        raise ValueError(
            "the last recorded auxiliary state is not the target model's final "
            "pre-norm hidden state"
        )


def chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return f"{base_url}/chat/completions"


def detokenize_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return f"{base_url}/detokenize"


def dump_path_for_request(dump_dir: Path, response_id: str) -> Path:
    digest = hashlib.sha256(response_id.encode()).hexdigest()
    return dump_dir / f"{digest}.safetensors"


def is_engine_request_id(engine_request_id: str, response_id: str) -> bool:
    """Match a public response ID to vLLM's internally suffixed request ID."""
    return engine_request_id == response_id or engine_request_id.startswith(
        f"{response_id}-"
    )


def send_conversation(
    index: int,
    args: argparse.Namespace,
    start_event: threading.Event,
) -> RequestResult:
    client_request_id = f"dflash-demo-{index:04d}-{uuid.uuid4().hex}"
    messages = CONVERSATIONS[index % len(CONVERSATIONS)]
    payload = {
        "model": args.model,
        "messages": messages,
        "request_id": client_request_id,
        "temperature": 1.0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": client_request_id,
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    request = Request(
        chat_completions_url(args.base_url),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    start_event.wait()
    started = time.monotonic()
    try:
        with urlopen(request, timeout=args.request_timeout) as response:
            body = json.load(response)
    except HTTPError as error:
        error_body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"request {index} failed: HTTP {error.code}: {error_body}"
        ) from error

    latency = time.monotonic() - started
    response_id = body["id"]
    completion_tokens = int(body.get("usage", {}).get("completion_tokens", 0))
    return RequestResult(
        index=index,
        response_id=response_id,
        completion_tokens=completion_tokens,
        latency_seconds=latency,
        dump_path="",
        engine_request_id="",
    )


def run_requests(args: argparse.Namespace) -> list[RequestResult]:
    if args.num_requests < 1:
        raise ValueError("--num-requests must be positive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    start_event = threading.Event()
    results: list[RequestResult] = []
    errors: list[Exception] = []
    workers = min(args.concurrency, args.num_requests)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: list[Future[RequestResult]] = [
            executor.submit(send_conversation, index, args, start_event)
            for index in range(args.num_requests)
        ]
        start_event.set()
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                errors.append(error)
                continue
            results.append(result)
            print(
                f"[response {result.index:02d}] id={result.response_id} "
                f"tokens={result.completion_tokens} "
                f"latency={result.latency_seconds:.2f}s",
                flush=True,
            )

    if errors:
        details = "\n".join(str(error) for error in errors)
        raise RuntimeError(f"{len(errors)} concurrent request(s) failed:\n{details}")
    return sorted(results, key=lambda result: result.index)


def detokenize_tokens(args: argparse.Namespace, token_ids: list[int]) -> str:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    payload = {"model": args.model, "tokens": token_ids}
    request = Request(
        detokenize_url(args.base_url),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.request_timeout) as response:
            body = json.load(response)
    except HTTPError as error:
        error_body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"detokenization failed: HTTP {error.code}: {error_body}"
        ) from error
    return str(body["prompt"])


def wait_for_dumps(
    results: list[RequestResult], dump_dir: Path, timeout: float
) -> list[Path]:
    pending = {result.response_id: result for result in results}
    resolved: dict[str, Path] = {}
    inspected_paths: set[Path] = set()
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for path in dump_dir.glob("*.safetensors"):
            if path in inspected_paths:
                continue
            try:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    engine_request_id = (handle.metadata() or {}).get("request_id")
            except Exception:
                continue
            inspected_paths.add(path)
            if engine_request_id is None:
                continue
            for response_id, result in tuple(pending.items()):
                if not is_engine_request_id(engine_request_id, response_id):
                    continue
                result.dump_path = str(path.resolve())
                result.engine_request_id = engine_request_id
                resolved[response_id] = path.resolve()
                pending.pop(response_id)
                break
        if pending:
            time.sleep(0.1)
    if pending:
        missing = "\n".join(sorted(pending))
        raise TimeoutError(f"timed out waiting for response IDs:\n{missing}")
    return [resolved[result.response_id] for result in results]


def validate_hidden_states(name: str, tensor: torch.Tensor, path: Path) -> None:
    if tensor.ndim != 3 or tensor.numel() == 0:
        raise AssertionError(f"{path}: {name} must be a nonempty 3-D tensor")
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    zero_count = tensor.numel() - int(torch.count_nonzero(tensor).item())
    all_zero_vectors = int(torch.count_nonzero(tensor, dim=-1).eq(0).sum().item())
    print(
        f"    {name}: shape={list(tensor.shape)}, zeros={zero_count}, "
        f"all-zero vectors={all_zero_vectors}, nan={nan_count}, inf={inf_count}"
    )
    if nan_count or inf_count:
        raise AssertionError(f"{path}: {name} contains non-finite values")
    if all_zero_vectors:
        raise AssertionError(f"{path}: {name} contains all-zero hidden states")


def primary_completion_hidden_states(
    hidden_states: torch.Tensor,
    verification_offsets: torch.Tensor,
    output_offsets: torch.Tensor,
    num_states: int,
) -> torch.Tensor:
    """Select anchor and accepted-token states from the primary trajectory."""
    selected: list[torch.Tensor] = []
    remaining_states = num_states
    for block_idx in range(len(output_offsets) - 1):
        output_len = int(output_offsets[block_idx + 1] - output_offsets[block_idx])
        primary_len = min(output_len, remaining_states)
        if primary_len <= 0:
            break
        block_start = int(verification_offsets[block_idx])
        selected.append(hidden_states[block_start : block_start + primary_len])
        remaining_states -= primary_len

    if remaining_states:
        raise AssertionError(
            f"trace is missing {remaining_states} primary completion hidden states"
        )
    return torch.cat(selected) if selected else hidden_states[:0]


def validate_trace(result: RequestResult) -> ValidatedTrace:
    path = Path(result.dump_path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        engine_request_id = metadata.get("request_id", "")
        if not is_engine_request_id(engine_request_id, result.response_id):
            raise AssertionError(
                f"{path}: metadata request ID {engine_request_id!r} does not match "
                f"public response ID {result.response_id!r}"
            )
        trace_format = metadata.get("format")
        if trace_format != "vllm-spec-decode-training-hidden-states-v1":
            raise AssertionError(
                f"{path}: expected generic v1 trace, found {trace_format!r}; "
                "restart the server with the updated recorder"
            )
        prompt_len = int(handle.get_tensor("prompt_len").item())
        prefill_token_ids = handle.get_tensor("prefill_token_ids").to(torch.int64)
        prefill_positions = handle.get_tensor("prefill_positions").to(torch.int64)
        if len(prefill_token_ids) != len(prefill_positions):
            raise AssertionError(f"{path}: prefill tokens do not match positions")
        prompt_positions = prefill_positions[prefill_positions < prompt_len]
        expected_prompt_positions = torch.arange(prompt_len, dtype=torch.int64)
        if not torch.equal(prompt_positions, expected_prompt_positions):
            raise AssertionError(f"{path}: prompt hidden-state trace is incomplete")
        prompt_token_ids = prefill_token_ids[prefill_positions < prompt_len].tolist()

        block_offsets = handle.get_tensor("verification_block_offsets").to(torch.int64)
        output_offsets = handle.get_tensor("output_block_offsets").to(torch.int64)
        output_token_ids = handle.get_tensor("output_token_ids").to(torch.int64)
        verification_input_token_ids = handle.get_tensor(
            "verification_input_token_ids"
        ).to(torch.int64)
        verification_hidden_states = handle.get_tensor("verification_hidden_states")
        num_steps = len(block_offsets) - 1
        if num_steps == 0:
            raise AssertionError(
                f"{path}: trace contains no speculative verification steps"
            )
        if len(output_offsets) != len(block_offsets):
            raise AssertionError(f"{path}: inconsistent block offsets")
        if int(block_offsets[-1]) != len(verification_hidden_states):
            raise AssertionError(f"{path}: verification rows do not match offsets")
        if len(verification_input_token_ids) != len(verification_hidden_states):
            raise AssertionError(f"{path}: verification tokens do not match states")
        if int(output_offsets[-1]) != len(output_token_ids):
            raise AssertionError(f"{path}: output tokens do not match offsets")

        generated_per_step = output_offsets[1:] - output_offsets[:-1]
        accepted = generated_per_step - 1
        verification_rows = block_offsets[1:] - block_offsets[:-1]
        proposal_counts = verification_rows - 1
        rejected = proposal_counts - accepted
        if torch.any(generated_per_step < 1):
            raise AssertionError(f"{path}: a verification step emitted no tokens")
        if torch.any(rejected < 0):
            raise AssertionError(f"{path}: accepted prefix exceeds proposal count")

        for block_idx in range(1, num_steps):
            anchor_idx = int(block_offsets[block_idx])
            previous_output_idx = int(output_offsets[block_idx]) - 1
            if (
                verification_input_token_ids[anchor_idx]
                != output_token_ids[previous_output_idx]
            ):
                raise AssertionError(
                    f"{path}: verification block {block_idx} does not continue "
                    "the preceding output block"
                )

        first_anchor_idx = int(block_offsets[0])
        reconstructed_completion = [int(verification_input_token_ids[first_anchor_idx])]
        reconstructed_completion.extend(output_token_ids.tolist())
        if len(reconstructed_completion) < result.completion_tokens:
            raise AssertionError(
                f"{path}: reconstructed only {len(reconstructed_completion)} of "
                f"{result.completion_tokens} API completion tokens"
            )
        completion_token_ids = reconstructed_completion[: result.completion_tokens]
        final_input_token_ids = prompt_token_ids + completion_token_ids

        hidden_shapes: dict[str, list[int]] = {}
        prefill_hidden_states = handle.get_tensor("prefill_hidden_states")
        for name in ("prefill_hidden_states", "verification_hidden_states"):
            tensor = (
                verification_hidden_states
                if name == "verification_hidden_states"
                else prefill_hidden_states
            )
            validate_hidden_states(name, tensor, path)
            hidden_shapes[name] = list(tensor.shape)

        final_prompt_rows = torch.nonzero(
            prefill_positions == prompt_len - 1, as_tuple=False
        ).flatten()
        if len(final_prompt_rows) != 1:
            raise AssertionError(
                f"{path}: expected exactly one final prompt hidden state"
            )
        final_prompt_state = prefill_hidden_states[int(final_prompt_rows[0]), -1]
        primary_states = primary_completion_hidden_states(
            verification_hidden_states[:, -1],
            block_offsets,
            output_offsets,
            result.completion_tokens - 1,
        )
        target_hidden_states = torch.cat(
            (final_prompt_state.unsqueeze(0), primary_states), dim=0
        )
        target_token_ids = torch.tensor(completion_token_ids, dtype=torch.int64)

    return ValidatedTrace(
        summary=TraceSummary(
            response_id=result.response_id,
            engine_request_id=engine_request_id,
            dump_path=str(path),
            completion_tokens=result.completion_tokens,
            prompt_tokens=prompt_len,
            final_input_token_ids=final_input_token_ids,
            generated_tokens_per_step=generated_per_step.tolist(),
            accepted_proposals_per_step=accepted.tolist(),
            rejected_proposals_per_step=rejected.tolist(),
            hidden_state_shapes=hidden_shapes,
            target_prediction_tokens=len(target_token_ids),
        ),
        target_hidden_states=target_hidden_states,
        target_token_ids=target_token_ids,
    )


def score_target_predictions(
    traces: list[ValidatedTrace], projection: TargetOutputProjection
) -> TargetTokenMetrics:
    hidden_states = torch.cat([trace.target_hidden_states for trace in traces])
    target_token_ids = torch.cat([trace.target_token_ids for trace in traces])
    metrics = projection.score(hidden_states, target_token_ids)
    offset = 0
    for trace in traces:
        end = offset + trace.summary.target_prediction_tokens
        entropy = metrics.entropy_bits[offset:end]
        hits = metrics.top1_hits[offset:end]
        probabilities = metrics.top1_probabilities[offset:end]
        trace.summary.mean_target_entropy_bits = float(entropy.mean())
        trace.summary.median_target_entropy_bits = float(
            torch.quantile(entropy.float(), 0.5)
        )
        trace.summary.target_top1_hit_rate_pct = float(hits.float().mean() * 100)
        trace.summary.mean_target_top1_probability = float(probabilities.mean())
        trace.summary.median_target_top1_probability = float(
            torch.quantile(probabilities.float(), 0.5)
        )
        offset = end
    return metrics


def _render_histogram(title: str, rows: list[tuple[str, int]]) -> None:
    total = sum(count for _, count in rows)
    largest = max((count for _, count in rows), default=0)
    print(f"\n{title}:")
    for label, count in rows:
        bar_length = round(32 * count / largest) if largest else 0
        percentage = 100 * count / total if total else 0.0
        print(f"  {label:>14} | {'#' * bar_length:<32} {count:>6} ({percentage:6.2f}%)")


def _binned_counts(
    values: list[float], bins: list[tuple[str, float, float]]
) -> list[tuple[str, int]]:
    return [
        (label, sum(lower <= value < upper for value in values))
        for label, lower, upper in bins
    ]


def print_metric_histograms(
    summaries: list[TraceSummary], metrics: TargetTokenMetrics
) -> None:
    entropy_bins = [
        ("< 0.25", -math.inf, 0.25),
        ("[0.25, 0.5)", 0.25, 0.5),
        ("[0.5, 1)", 0.5, 1.0),
        ("[1, 2)", 1.0, 2.0),
        ("[2, 4)", 2.0, 4.0),
        ("[4, 8)", 4.0, 8.0),
        ("[8, inf)", 8.0, math.inf),
    ]
    _render_histogram(
        "Per-token target entropy histogram (bits)",
        _binned_counts(metrics.entropy_bits.tolist(), entropy_bins),
    )

    num_hits = int(metrics.top1_hits.sum())
    _render_histogram(
        "Per-token target top-1 outcome histogram",
        [("miss", len(metrics.top1_hits) - num_hits), ("hit", num_hits)],
    )
    hit_rate_bins = [
        ("[0, 50)", 0.0, 50.0),
        ("[50, 60)", 50.0, 60.0),
        ("[60, 70)", 60.0, 70.0),
        ("[70, 80)", 70.0, 80.0),
        ("[80, 90)", 80.0, 90.0),
        ("[90, 95)", 90.0, 95.0),
        ("[95, 100]", 95.0, 100.000001),
    ]
    _render_histogram(
        "Per-request target top-1 hit-rate histogram (%)",
        _binned_counts(
            [summary.target_top1_hit_rate_pct for summary in summaries],
            hit_rate_bins,
        ),
    )

    step_lengths = Counter(
        length for summary in summaries for length in summary.generated_tokens_per_step
    )
    _render_histogram(
        "Verification-step output-length histogram (tokens)",
        [(str(length), step_lengths[length]) for length in sorted(step_lengths)],
    )
    completion_lengths = Counter(summary.completion_tokens for summary in summaries)
    _render_histogram(
        "API completion-length histogram (tokens)",
        [
            (str(length), completion_lengths[length])
            for length in sorted(completion_lengths)
        ],
    )


def main() -> None:
    args = parse_args()
    args.dump_dir = args.dump_dir.resolve()
    if args.logits_vocab_chunk_size < 1:
        raise ValueError("--logits-vocab-chunk-size must be positive")
    print(
        f"Sending {args.num_requests} conversations with concurrency "
        f"{min(args.concurrency, args.num_requests)}...",
        flush=True,
    )
    results = run_requests(args)
    dump_paths = wait_for_dumps(results, args.dump_dir, args.dump_timeout)

    print("\nCollected dump references:")
    for result, path in zip(results, dump_paths):
        print(
            f"  {result.response_id} -> {path} (engine ID: {result.engine_request_id})"
        )

    selected_result = random.Random(args.seed).choice(results)
    print(f"\nRandomly selected dump: {selected_result.dump_path}")

    print("\nTrace validation:")
    traces: list[ValidatedTrace] = []
    for result in results:
        trace = validate_trace(result)
        traces.append(trace)
        summary = trace.summary
        print(
            f"  {summary.response_id}: API completion tokens="
            f"{summary.completion_tokens}, generated per verification step="
            f"{summary.generated_tokens_per_step}, mean acceptance length="
            f"{summary.mean_acceptance_length:.3f}"
        )

    metrics_device = args.target_metrics_device
    if metrics_device == "auto":
        metrics_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"\nLoading target output norm and LM head on {metrics_device}...",
        flush=True,
    )
    projection = TargetOutputProjection.from_pretrained(
        model=args.model,
        draft_model=args.draft_model,
        revision=args.revision,
        draft_revision=args.draft_revision,
        device=metrics_device,
        vocab_chunk_size=args.logits_vocab_chunk_size,
    )
    target_metrics = score_target_predictions(traces, projection)
    summaries = [trace.summary for trace in traces]
    print("Target output metrics:")
    for summary in summaries:
        print(
            f"  {summary.response_id}: entropy mean="
            f"{summary.mean_target_entropy_bits:.3f} bits, median="
            f"{summary.median_target_entropy_bits:.3f} bits, top-1 hit rate="
            f"{summary.target_top1_hit_rate_pct:.2f}%, top-1 probability "
            f"mean={summary.mean_target_top1_probability:.4%}, "
            f"median={summary.median_target_top1_probability:.4%}"
        )

    all_step_lengths = [
        length for summary in summaries for length in summary.generated_tokens_per_step
    ]
    global_mean = statistics.fmean(all_step_lengths) if all_step_lengths else 1.0
    print(f"\nGlobal mean acceptance length: {global_mean:.3f}")

    print(
        "Global target output metrics: "
        f"entropy mean={float(target_metrics.entropy_bits.mean()):.3f} bits, "
        f"median="
        f"{float(torch.quantile(target_metrics.entropy_bits, 0.5)):.3f} bits, "
        f"top-1 hit rate="
        f"{float(target_metrics.top1_hits.float().mean() * 100):.2f}%, "
        f"top-1 probability mean="
        f"{float(target_metrics.top1_probabilities.mean()):.4%}, "
        f"median="
        f"{float(torch.quantile(target_metrics.top1_probabilities, 0.5)):.4%}"
    )
    print_metric_histograms(summaries, target_metrics)

    selected = next(
        summary
        for summary in summaries
        if summary.response_id == selected_result.response_id
    )
    print("Selected trace summary:")
    selected_summary = asdict(selected)
    selected_summary.pop("final_input_token_ids")
    print(json.dumps(selected_summary, indent=2))
    print(
        "Reconstructed final input token IDs "
        f"({selected.prompt_tokens} prompt + {selected.completion_tokens} "
        f"completion):\n{selected.final_input_token_ids}"
    )

    prompt_token_ids = selected.final_input_token_ids[: selected.prompt_tokens]
    completion_token_ids = selected.final_input_token_ids[selected.prompt_tokens :]
    detokenized_prompt = detokenize_tokens(args, prompt_token_ids)
    detokenized_completion = detokenize_tokens(args, completion_token_ids)
    print("\nDetokenized reconstructed conversation:")
    print(detokenized_prompt)
    print("\n---------- COMPLETION BEGINS ----------\n")
    print(detokenized_completion)

    print("Validation passed: all hidden-state vectors are nonzero and finite.")


if __name__ == "__main__":
    main()
