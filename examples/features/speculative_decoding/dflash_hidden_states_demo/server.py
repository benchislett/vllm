# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch an OpenAI-compatible DFlash server with hidden-state recording."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

TARGET_MODEL = "nvidia/smart-panda-mtp-graft-NVFP4-20260701"
DRAFT_MODEL = "nvidia/nano3.5_smart_panda_dflash"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the NVIDIA Smart Panda target with its DFlash drafter and "
            "write one hidden-state trace per request. Unknown arguments are "
            "forwarded to `vllm serve`."
        )
    )
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--draft-model", default=DRAFT_MODEL)
    parser.add_argument("--dump-dir", type=Path, default=Path("/tmp/dflash-traces"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the vLLM command and exit."
    )
    return parser.parse_known_args()


def build_vllm_args(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    """Build arguments for the vLLM CLI."""
    speculative_config = {
        "method": "dflash",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "max_model_len": args.max_model_len,
        "verification_hidden_states_output_dir": str(args.dump_dir.resolve()),
    }
    return [
        "vllm",
        "serve",
        args.target_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--enable-request-id-headers",
        "--speculative-config",
        json.dumps(speculative_config),
        *extra_args,
    ]


def main() -> None:
    args, extra_args = parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    command = build_vllm_args(args, extra_args)
    print(f"Hidden-state dump directory: {args.dump_dir.resolve()}", flush=True)
    print(f"Launching: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return

    sys.argv = command
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
