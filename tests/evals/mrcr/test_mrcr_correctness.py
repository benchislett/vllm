# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRCR long-context accuracy test.

Usage:
    pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
        --config-list-file=configs/models-small.txt
"""

import shlex

import requests
import yaml

from tests.evals.utils import run_evaluation
from tests.utils import RemoteOpenAIServer


def test_mrcr_correctness(config_filename, tmp_path):
    cfg = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    server_args = shlex.split(cfg.get("server_args", ""))
    server_args += ["--trust-remote-code", "--disable-uvicorn-access-log"]

    print(
        f"MRCR eval for {cfg['model_name']} (threshold {cfg['match_ratio_threshold']})"
    )

    with RemoteOpenAIServer(
        cfg["model_name"],
        server_args,
        env_dict=cfg.get("env"),
        max_wait_seconds=cfg.get("startup_max_wait_seconds", 600),
    ) as server:
        base_url = server.url_for("v1").rstrip("/")
        max_tokens = cfg.get("max_tokens", 2048)
        max_prompt_tokens = cfg.get("max_prompt_tokens")
        if max_prompt_tokens is None:
            response = requests.get(base_url + "/models", timeout=30)
            response.raise_for_status()
            max_prompt_tokens = (
                response.json()["data"][0]["max_model_len"] - max_tokens - 256
            )
        needles = cfg.get("needles", [2, 4, 8])
        num_samples = cfg.get("num_samples", 40)
        per_bucket, leftover = divmod(num_samples, len(needles))
        scores, counts = {}, {}
        for index, needle in enumerate(needles):
            limit = per_bucket + (index < leftover)
            if limit == 0:
                continue
            report = run_evaluation(
                {
                    "model": cfg["model_name"],
                    "datasets": ["openai_mrcr"],
                    "dataset_hub": "huggingface",
                    "limit": limit,
                    "seed": cfg.get("seed", 42),
                    "eval_batch_size": cfg.get("concurrency", 8),
                    "dataset_args": {
                        "openai_mrcr": {
                            "extra_params": {
                                "needle_count": [needle],
                                "max_context_size": max_prompt_tokens,
                            },
                        }
                    },
                    "generation_config": {
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "extra_body": cfg.get(
                            "extra_body",
                            {"chat_template_kwargs": {"enable_thinking": False}},
                        ),
                    },
                },
                base_url,
                tmp_path / f"needle-{needle}",
            )
            result = report["metrics"]["openai_mrcr"]
            scores[needle], counts[needle] = result["score"], result["num"]
        results = {
            "match_ratio": sum(scores[n] * counts[n] for n in scores)
            / sum(counts.values()),
            "per_needle": {f"match_ratio_n{n}": value for n, value in scores.items()},
        }

    threshold = cfg["match_ratio_threshold"]
    tol = cfg.get("tolerance", 0.05)

    print(f"  match_ratio:     {results['match_ratio']:.4f}")
    for k, v in results["per_needle"].items():
        print(f"  {k}: {v:.4f}")

    failures: list[str] = []
    if isinstance(threshold, dict):
        for n, expected in threshold.items():
            key = f"match_ratio_n{int(n)}"
            measured = results["per_needle"].get(key)
            if measured is None:
                failures.append(f"{key}: no samples collected")
            elif measured < expected - tol:
                failures.append(f"{key}: {measured:.4f} < {expected:.4f} - {tol:.4f}")
    else:
        measured = results["match_ratio"]
        if measured < threshold - tol:
            failures.append(
                f"match_ratio: {measured:.4f} < {threshold:.4f} - {tol:.4f}"
            )

    assert not failures, "MRCR thresholds failed: " + "; ".join(failures)
