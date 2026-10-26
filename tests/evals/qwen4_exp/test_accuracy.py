# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K and AIME25 accuracy evaluation for Qwen3.8-Flash-Next-FP8."""

import shlex
from pathlib import Path

import yaml

from tests.evals.utils import run_evaluation
from tests.utils import RemoteOpenAIServer


def test_qwen4_exp_accuracy(config_filename: Path, tmp_path: Path):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    server_args = shlex.split(eval_config.get("server_args", ""))
    server_args.extend(["--trust-remote-code", "--disable-uvicorn-access-log"])

    model_name = eval_config["model_name"]
    print(f"Starting Qwen4Exp evaluation for model: {model_name}")
    print(f"Datasets: {', '.join(eval_config['datasets'])}")
    print(f"Server args: {' '.join(server_args)}")

    with RemoteOpenAIServer(
        model_name,
        server_args,
        env_dict=eval_config.get("env"),
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 1800),
    ) as remote_server:
        report = run_evaluation(eval_config, remote_server.url_for("v1"), tmp_path)
        scores = {name: value["score"] for name, value in report["metrics"].items()}

    for dataset, metric_config in eval_config["datasets"].items():
        score = scores[dataset]
        threshold = metric_config["metric_threshold"]
        tolerance = metric_config.get("tolerance", 0.0)
        minimum_score = threshold - tolerance

        print(
            f"{dataset}: measured={score:.4f}, expected={threshold:.4f}, "
            f"tolerance={tolerance:.4f}"
        )
        assert score >= minimum_score, (
            f"{dataset} score too low: {score:.4f} < {threshold:.4f} - "
            f"{tolerance:.4f} = {minimum_score:.4f}"
        )
