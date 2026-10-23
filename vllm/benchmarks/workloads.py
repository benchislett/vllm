# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned defaults for fixed-length serving workloads."""

import argparse

_COMMON = {
    "dataset_name": "random",
    "random_range_ratio": 0.0,
    "random_prefix_len": 0,
    "ignore_eos": True,
    "seed": 0,
    "num_prompts": 1000,
    "num_warmups": 1,
    "request_rate": float("inf"),
}
WORKLOADS = {
    name: {**_COMMON, "random_input_len": isl, "random_output_len": osl}
    for name, isl, osl in (
        ("fixed-1k1k-v1", 1024, 1024),
        ("fixed-8k1k-v1", 8192, 1024),
        ("fixed-1k8k-v1", 1024, 8192),
    )
}


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workload",
        choices=WORKLOADS,
        help="Versioned workload defaults. Explicit options override the preset.",
    )
    # None distinguishes an omitted option from an explicit value equal to the
    # legacy default. Resolve once before any benchmark code consumes the args.
    defaults = {key: parser.get_default(key) for key in WORKLOADS["fixed-8k1k-v1"]}
    parser.set_defaults(**dict.fromkeys(defaults), _workload_defaults=defaults)


def resolve_workload(args: argparse.Namespace) -> None:
    defaults = getattr(args, "_workload_defaults", {})
    preset = WORKLOADS.get(getattr(args, "workload", None) or "", {})
    if not hasattr(args, "_allow_eos"):
        args._allow_eos = bool(defaults) and args.ignore_eos is False
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, default))
    if preset and args.dataset_name != "random":
        raise ValueError("Fixed-length workloads require --dataset-name random")
    if preset and args.max_concurrency is None:
        raise ValueError("--workload requires an explicit --max-concurrency")


def workload_metadata(args: argparse.Namespace) -> dict | None:
    name = getattr(args, "workload", None)
    if name is None:
        return None
    resolved = {key: getattr(args, key) for key in WORKLOADS[name]}
    if isinstance(resolved["random_range_ratio"], str):
        import json

        resolved["random_range_ratio"] = json.loads(resolved["random_range_ratio"])
    return {
        "name": name,
        "modified": resolved != WORKLOADS[name],
        "resolved": resolved,
    }
