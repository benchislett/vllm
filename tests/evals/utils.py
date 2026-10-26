# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the installed evaluation CLI and retain its reports in the test artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


def run_evaluation(config: dict, base_url: str, work_dir: Path, timeout=1800) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config))
    report_path = work_dir / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "eval",
            "--eval-config",
            str(config_path),
            "--api-url",
            base_url,
            "--work-dir",
            str(work_dir / "artifacts"),
            "--save-results",
            str(report_path),
        ],
        check=True,
        timeout=timeout,
    )
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed", report["records"].get("errors")
    return report
