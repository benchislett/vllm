# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed, versioned serving workloads."""

import argparse
import json
import os

_COMMON = {
    "dataset_name": "random",
    "random_range_ratio": 0.2,
    "backend": "openai-chat",
    "endpoint": "/v1/chat/completions",
    "random_prefix_len": 0,
    "seed": 0,
    "num_warmups": 1,
    "request_rate": float("inf"),
}
WORKLOADS = {
    name: {**_COMMON, "random_input_len": isl, "random_output_len": osl}
    for name, isl, osl in (
        ("1k1k-v1", 1024, 1024),
        ("8k1k-v1", 8192, 1024),
    )
}


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workload",
        choices=WORKLOADS,
        help="Fixed versioned workload; conflicting workload options are rejected.",
    )
    # None distinguishes an omitted option from an explicit value equal to the
    # legacy default. Resolve once before any benchmark code consumes the args.
    defaults = {
        key: parser.get_default(key) for key in (*WORKLOADS["8k1k-v1"], "num_prompts")
    }
    parser.set_defaults(**dict.fromkeys(defaults), _workload_defaults=defaults)


def resolve_workload(args: argparse.Namespace) -> None:
    defaults = getattr(args, "_workload_defaults", {})
    preset = WORKLOADS.get(getattr(args, "workload", None) or "", {})
    if preset:
        if args.max_concurrency is None or args.max_concurrency <= 0:
            raise ValueError("--workload requires a positive --max-concurrency")
        preset = {**preset, "num_prompts": max(20, 10 * args.max_concurrency)}
        for alias, key in (
            ("input_len", "random_input_len"),
            ("output_len", "random_output_len"),
        ):
            if getattr(args, alias, None) not in (None, preset[key]):
                raise ValueError(
                    f"--{alias.replace('_', '-')} conflicts with --workload"
                )
        for key in ("ramp_up_strategy", "max_model_len", "probe_request_rate"):
            if getattr(args, key, None) not in (None, 0):
                raise ValueError(f"--{key.replace('_', '-')} conflicts with --workload")
        if set(getattr(args, "extra_body", None) or {}) & {
            "prompt",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "ignore_eos",
            "n",
        }:
            raise ValueError("--extra-body overrides fixed workload request fields")
    for key, default in defaults.items():
        value = getattr(args, key)
        expected = preset.get(key, default)
        if preset and key == "random_range_ratio" and isinstance(value, str):
            value = json.loads(value)
        if preset and value is not None and value != expected:
            raise ValueError(
                f"--{key.replace('_', '-')} is fixed at {expected} by --workload; "
                "omit --workload for custom settings"
            )
        if value is None:
            setattr(args, key, expected)


def workload_metadata(args: argparse.Namespace) -> dict | None:
    name = getattr(args, "workload", None)
    if name is None:
        return None
    resolved = {key: getattr(args, key) for key in (*WORKLOADS[name], "num_prompts")}
    if isinstance(resolved["random_range_ratio"], str):
        resolved["random_range_ratio"] = json.loads(resolved["random_range_ratio"])
    return {
        "name": name,
        "resolved": resolved,
    }


async def prefix_cache_snapshot(
    base_url: str, headers: dict | None, ssl_context
) -> float:
    """Require disabled prefix caching and read counters across all serving engines."""
    import aiohttp
    from prometheus_client.parser import text_string_to_metric_families

    headers = dict(headers or {})
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers.setdefault("Authorization", f"Bearer {api_key}")
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session,
        session.get(
            base_url.rstrip("/") + "/metrics", headers=headers, ssl=ssl_context
        ) as response,
    ):
        response.raise_for_status()
        samples = [
            sample
            for metric in text_string_to_metric_families(await response.text())
            for sample in metric.samples
        ]
    configurations = [s for s in samples if s.name == "vllm:cache_config_info"]
    if not configurations or any(
        s.labels.get("enable_prefix_caching", "").lower() != "false"
        for s in configurations
    ):
        raise ValueError(
            "--workload requires verified disabled prefix caching; start the server "
            "with --no-enable-prefix-caching and expose /metrics"
        )
    hits = [
        s.value
        for s in samples
        if s.name
        in {"vllm:prefix_cache_hits_total", "vllm:external_prefix_cache_hits_total"}
    ]
    if not hits:
        raise ValueError("Cannot verify workload prefix-cache hits: counters missing")
    return sum(hits)
