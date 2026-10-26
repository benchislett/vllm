# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin EvalScope launcher; adapters own datasets, prompts and scoring."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import regex as re
import yaml

from vllm.benchmarks.lib.report import make_report

# EvalScope's OpenAI MRCR adapter counts message content with o200k_base.
# These are input-length bins, not the serving model's context-window sizes.
PRESETS: dict[str, dict[str, Any]] = {
    f"mrcr-v2-8needle-{size}k": {
        "dataset": "openai_mrcr",
        "extra_params": {
            "needle_count": [8],
            "min_context_size": size * 1024 // 2,
            "max_context_size": size * 1024,
            "tik_enc": "o200k_base",
        },
    }
    for size in (128, 256)
}


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "eval_names",
        nargs="*",
        help="EvalScope dataset names (e.g. gsm8k, gpqa_diamond, aime25), or "
        + ", ".join(PRESETS),
    )
    parser.add_argument("--eval-config", type=Path, help="EvalScope or Qwen eval YAML.")
    parser.add_argument("--model", help="Served model name; overrides the config.")
    parser.add_argument("--api-url", help="OpenAI API URL, including /v1.")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--dataset-hub", choices=["modelscope", "huggingface", "local"])
    parser.add_argument("--limit", type=int, help="Maximum samples per evaluation.")
    parser.add_argument("--generation-config", type=json.loads, help="JSON object.")
    parser.add_argument(
        "--dataset-args", type=json.loads, help="EvalScope JSON object."
    )
    parser.add_argument(
        "--work-dir", type=Path, help="Empty directory for raw artifacts."
    )
    parser.add_argument(
        "--report-format", choices=["byog", "evalscope"], default="byog"
    )
    parser.add_argument(
        "--save-results", type=Path, help="Defaults to WORK_DIR/result.json."
    )


def task_config(config: dict[str, Any]) -> dict[str, Any]:
    """Accept EvalScope config and the existing Qwen accuracy-test YAML shape."""
    config = copy.deepcopy(config)
    if "model_name" in config:
        config.setdefault("model", config.pop("model_name"))
    # Server startup and threshold checks belong to the caller, not EvalScope.
    for key in ("server_args", "startup_max_wait_seconds", "env"):
        config.pop(key, None)
    if "datasets" in config:
        config["datasets"] = list(config["datasets"])
    config.setdefault("eval_batch_size", 32)
    return config


def run_cli(args: argparse.Namespace) -> None:
    try:
        from evalscope.config import TaskConfig
        from evalscope.run import run_task
    except ImportError as exc:
        raise RuntimeError(
            'Install the evaluation extra: pip install "vllm[eval]"'
        ) from exc

    config = task_config(
        yaml.safe_load(args.eval_config.read_text()) if args.eval_config else {}
    )
    for key in (
        "model",
        "api_url",
        "eval_batch_size",
        "limit",
        "generation_config",
        "dataset_args",
        "dataset_hub",
    ):
        if (value := getattr(args, key)) is not None:
            config[key] = value
    names = args.eval_names or config.get("datasets", [])
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "Specify at least one evaluation; duplicate names are not allowed."
        )
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("Evaluation names must contain only letters, digits, - or _.")
    if not config.get("model"):
        raise ValueError("Specify --model or model/model_name in --eval-config.")
    config.setdefault("api_url", "http://localhost:8000/v1")
    config.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
    # Never accidentally load a local model or reuse stale prediction caches.
    if config.get("use_cache"):
        raise ValueError("Use a fresh work directory; cached runs are not supported.")
    config.update(eval_type="openai_api", no_timestamp=True, ignore_errors=True)
    work_dir = args.work_dir or Path("eval_results") / uuid.uuid4().hex
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError(f"Work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    settings, metrics, reports, errors = {}, {}, {}, {}
    for name in names:
        preset = PRESETS.get(name, {})
        dataset = preset.get("dataset", name)
        current = copy.deepcopy(config)
        current.update(datasets=[dataset], work_dir=str(work_dir / name))
        dataset_args = current.setdefault("dataset_args", {}).setdefault(dataset, {})
        if preset:
            extra = {**preset["extra_params"], **dataset_args.get("extra_params", {})}
            dataset_args["extra_params"] = extra
        if dataset in ("openai_mrcr", "vllm_mrcr"):
            import vllm.evals.mrcr  # noqa: F401

            if dataset == "openai_mrcr":
                current["dataset_args"]["vllm_mrcr"] = current["dataset_args"].pop(
                    dataset
                )
                dataset = "vllm_mrcr"
                current["datasets"] = [dataset]
        try:
            task = TaskConfig(**current)
            # Pydantic masks API keys in this snapshot, as in EvalScope's own files.
            settings[name] = task.model_dump(mode="json")
            result = run_task(task)[dataset]
            reports[name] = result.model_dump(mode="json")
            metrics[name] = {"score": result.score, "num": result.num}
            completion = result.execution_summary
            if (
                not completion
                or completion.incomplete
                or completion.errored
                or completion.succeeded == 0
                or result.score is None
            ):
                errors[name] = "Evaluation did not produce a complete scored result."
        except Exception as exc:
            # Keep other evaluations and partial on-disk predictions available.
            logging.getLogger(__name__).exception("Evaluation %s failed", name)
            errors[name] = type(exc).__name__
            print(f"{name} failed ({type(exc).__name__}); inspect {work_dir / name}")
    artifacts = {}
    for path in sorted(work_dir.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                artifacts[str(path.relative_to(work_dir))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    report = make_report(
        kind="evaluation",
        name=",".join(names),
        settings={
            "evalscope_version": importlib.metadata.version("evalscope"),
            "tasks": settings,
        },
        metrics=metrics,
        records={
            "evalscope_reports": reports,
            "artifacts": artifacts,
            "errors": errors,
        },
        status="partial" if errors else "completed",
    )
    destination = args.save_results or work_dir / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report if args.report_format == "byog" else reports, indent=2) + "\n"
    )
    print(f"Evaluation report: {destination}; raw artifacts: {work_dir}")
    if errors:
        raise SystemExit(1)
