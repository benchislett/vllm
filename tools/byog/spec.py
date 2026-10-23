# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a single-server BYOG spec before allocating inference resources."""

import copy
import hashlib
import json
import math
from pathlib import Path

import regex as re
import yaml

from tools.recipes.recipe_json_to_vllm_config import argv_to_config, recipe_argv


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve(path: Path, overrides: list[Path]) -> dict:
    spec = {}
    for source in [path, *overrides]:
        value = yaml.safe_load(source.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping in {source}")
        if "recipe" in value:
            value["recipe"]["path"] = str(
                (source.parent / value["recipe"]["path"]).resolve()
            )
        spec = merge(spec, value)
    unknown = set(spec) - {
        "schema",
        "name",
        "model",
        "recipe",
        "serve",
        "evals",
        "benchmarks",
    }
    if unknown or spec.get("schema") != "byog.spec/v0":
        raise ValueError(f"Invalid spec schema or unknown fields: {sorted(unknown)}")
    model = spec.get("model", {})
    if set(model) != {"id", "revision"} or not isinstance(model["id"], str):
        raise ValueError("model must contain id and revision")
    if not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"])):
        raise ValueError("model.revision must be a full Hugging Face commit SHA")
    serve = spec.get("serve", {})
    if set(serve) - {
        "config",
        "env",
        "startup_timeout_seconds",
        "client_timeout_seconds",
    }:
        raise ValueError("Unknown serve settings")
    config, environment = {}, {}
    if recipe := spec.get("recipe"):
        if set(recipe) != {"path", "sha256"}:
            raise ValueError("recipe requires path and sha256")
        raw = Path(recipe["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != recipe["sha256"]:
            raise ValueError("Recipe content does not match its pinned SHA256")
        rendering = json.loads(raw)
        config = argv_to_config(recipe_argv(rendering))
        environment = rendering.get("env", {})
        if config.pop("model") != model["id"]:
            raise ValueError("Recipe model differs from spec model")
    config = merge(config, serve.get("config", {}))
    # The runner owns endpoint and model identity; overrides cannot redirect it.
    if set(config) & {
        "model",
        "revision",
        "tokenizer",
        "tokenizer-revision",
        "host",
        "port",
        "served-model-name",
        "api-key",
        "uds",
        "headless",
        "ssl-keyfile",
        "ssl-certfile",
        "config",
    }:
        raise ValueError(
            "serve.config overrides a runner-owned identity or endpoint setting"
        )
    environment = merge(environment, serve.get("env", {}))
    for key, value in environment.items():
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or value is not None
            and not isinstance(value, str)
        ):
            raise ValueError(
                "serve.env requires environment variable names and string values"
            )
        if re.search(r"(^|_)(TOKEN|SECRET|PASSWORD|API_KEY)$", key):
            raise ValueError(
                "Pass credentials through the process environment, not a recorded spec"
            )
    # Avoid CI's synchronization checking changing benchmark behavior.
    environment.setdefault("VLLM_GPU_SYNC_CHECK", None)
    environment.setdefault("VLLM_USE_RUST_BENCH", "0")
    spec["serve"] = {
        "config": config,
        "env": environment,
        "startup_timeout_seconds": serve.get("startup_timeout_seconds", 1200),
        "client_timeout_seconds": serve.get("client_timeout_seconds", 3600),
    }
    for key in ("startup_timeout_seconds", "client_timeout_seconds"):
        value = spec["serve"][key]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be positive")
    spec.setdefault(
        "evals",
        [
            {
                "name": "gsm8k",
                "args": {
                    "num-questions": 32,
                    "use-chat-completions": True,
                    "max-tokens": 1024,
                    "temperature": 0,
                },
            }
        ],
    )
    spec.setdefault(
        "benchmarks",
        [{"workload": "8k1k-v1", "concurrency": [1, 16, 64, 256], "args": {}}],
    )
    for evaluation in spec["evals"]:
        if set(evaluation) - {"name", "args"} or evaluation.get("name") != "gsm8k":
            raise ValueError("Only the gsm8k evaluation is currently supported")
    for benchmark in spec["benchmarks"]:
        if set(benchmark) - {"workload", "concurrency", "args"}:
            raise ValueError("Unknown benchmark settings")
        points = benchmark.get("concurrency", [])
        if not points or any(type(n) is not int or n <= 0 for n in points):
            raise ValueError(
                "Benchmark concurrency must be a nonempty list of positive integers"
            )
    return spec


def options(values: dict) -> list[str]:
    """Encode config values as CLI arguments without shell interpretation."""
    result = []
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", key):
            raise ValueError(f"Invalid option name: {key!r}")
        if value is None:
            continue
        if isinstance(value, bool):
            result.append(f"--{key}" if value else f"--no-{key}")
        elif isinstance(value, (dict, list)):
            result += [f"--{key}", json.dumps(value)]
        else:
            result += [f"--{key}", str(value)]
    return result
