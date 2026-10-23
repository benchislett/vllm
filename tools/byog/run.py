# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a resolved BYOG spec and retain a bundle, including partial results."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import urllib.request
import uuid
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

import yaml

from tools.byog.spec import options, resolve


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def process(command: list[str], log: Path, env: dict):
    with log.open("w") as output:
        child = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            yield child
        finally:
            # Stop the process group even if its leader exited first.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                child.wait(timeout=15)
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def ready(child: subprocess.Popen, url: str, model: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Server exited with status {child.returncode}")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                models = json.load(response)["data"]
            if any(item["id"] == model for item in models):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Server did not become ready with the expected model identity")


def probe(url: str, alias: str, spec: dict) -> dict:
    settings = spec["evals"][0].get("args", {}) if spec["evals"] else {}
    chat = settings.get("use-chat-completions", False)
    payload = {"model": alias, "max_tokens": 32, "temperature": 0}
    if chat:
        payload["messages"] = [{"role": "user", "content": "Reply with the number 1."}]
        for key in ("chat-template-kwargs", "reasoning-effort"):
            if key in settings:
                payload[key.replace("-", "_")] = settings[key]
    else:
        payload["prompt"] = "Reply with the number 1."
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    request = urllib.request.Request(
        url + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=spec["serve"]["startup_timeout_seconds"]
    ) as response:
        result = json.load(response)
    choice = result["choices"][0]
    message = choice.get("message", {})
    if not (choice.get("text") or message.get("content") or message.get("reasoning")):
        raise RuntimeError("Server probe returned no generated content")
    return {"request": payload, "response": result}


def plan(spec: dict, output: Path, url: str, alias: str, tokenizer: str) -> list[dict]:
    jobs = []
    owned = {
        "model",
        "tokenizer",
        "base-url",
        "host",
        "port",
        "report-format",
        "save-results",
        "result-dir",
        "result-filename",
        "config",
        "append-result",
    }
    for i, evaluation in enumerate(spec["evals"]):
        args = evaluation.get("args", {})
        if set(args) & owned:
            raise ValueError(
                "Eval args override runner-owned endpoint or report settings"
            )
        name = f"eval-{i}-gsm8k"
        report = output / f"{name}.json"
        command = [
            "vllm",
            "eval",
            "gsm8k",
            *options(args),
            "--base-url",
            url,
            "--model",
            alias,
            "--report-format",
            "byog",
            "--save-results",
            str(report),
        ]
        if "data-dir" not in args and os.getenv("BYOG_DATA_DIR"):
            command += ["--data-dir", os.environ["BYOG_DATA_DIR"]]
        jobs.append({"name": name, "command": command, "report": str(report)})
    for i, benchmark in enumerate(spec["benchmarks"]):
        args = benchmark.get("args", {})
        if set(args) & (owned | {"workload", "max-concurrency"}):
            raise ValueError("Benchmark args override runner-owned settings")
        for j, concurrency in enumerate(benchmark["concurrency"]):
            name = f"bench-{i}-{j}-c{concurrency}"
            command = [
                "vllm",
                "bench",
                "serve",
                "--workload",
                benchmark["workload"],
                "--max-concurrency",
                str(concurrency),
                *options(args),
                "--model",
                alias,
                "--tokenizer",
                tokenizer,
                "--base-url",
                url,
                "--report-format",
                "byog",
                "--result-dir",
                str(output),
                "--result-filename",
                f"{name}.json",
            ]
            jobs.append(
                {
                    "name": name,
                    "command": command,
                    "report": str(output / f"{name}.json"),
                }
            )
    if not jobs:
        raise ValueError("Spec must select at least one evaluation or benchmark")
    return jobs


def validate_commands(jobs: list[dict], config: dict) -> None:
    # Use the clients' own parsers to reject unknown/mistyped settings pre-launch.
    from vllm import platforms
    from vllm.benchmarks.serve import add_cli_args as bench_args
    from vllm.benchmarks.workloads import resolve_workload
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.evals.gsm8k import add_cli_args as eval_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if platforms.current_platform.is_unspecified():
        from vllm.platforms.cpu import CpuPlatform

        platforms.current_platform = CpuPlatform()
    for job in jobs:
        parser = FlexibleArgumentParser()
        if job["command"][1] == "bench":
            bench_args(parser)
            resolve_workload(parser.parse_args(job["command"][3:]))
        else:
            eval_args(parser)
            parser.parse_args(job["command"][3:])
    parser = make_arg_parser(FlexibleArgumentParser())
    parser.parse_args(["MODEL", *options(config)])


def hardware() -> dict:
    result = {
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
    }
    if shutil.which("nvidia-smi"):
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            result["gpu_query"] = {
                "returncode": query.returncode,
                "output": query.stdout.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["gpu_query"] = {"error": str(exc)}
    return result


def run(spec: dict, output: Path, dry_run: bool) -> int:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty to prevent mixing runs")
    alias = "byog-dry-run" if dry_run else "byog-" + uuid.uuid4().hex
    port = 8000
    if not dry_run:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    jobs = plan(spec, output, url, alias, spec["model"]["id"])
    validate_commands(jobs, spec["serve"]["config"])
    replay_spec = dict(spec)
    if recipe := spec.get("recipe"):
        shutil.copyfile(recipe["path"], output / "recipe.json")
        replay_spec["recipe"] = {**recipe, "path": "recipe.json"}
    (output / "spec.resolved.yaml").write_text(
        yaml.safe_dump(replay_spec, sort_keys=False)
    )
    image_reference = os.getenv("BYOG_IMAGE_REFERENCE")
    if image_reference:
        if "@sha256:" not in image_reference:
            raise ValueError("BYOG_IMAGE_REFERENCE must identify an image by digest")
        reproduction = (
            'docker run --rm --gpus all --ipc host -v "$PWD:/out" '
            '-v "${HF_HUB_CACHE:?}:/root/.cache/huggingface/hub" '
            f"-e BYOG_IMAGE_REFERENCE={shlex.quote(image_reference)} "
            f"{shlex.quote(image_reference)} --spec /out/spec.resolved.yaml "
            "--output /out/replay"
        )
    else:
        reproduction = (
            "python -m tools.byog.run --spec spec.resolved.yaml --output replay"
        )
    (output / "repro.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + reproduction + "\n"
    )
    write_json(output / "plan.json", {"jobs": jobs})
    if dry_run:
        return 0
    summary = {
        "schema": "byog.result/v0",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "image_reference_reported_by_user": image_reference,
        "jobs": [],
        "hardware_before": hardware(),
    }
    build_info = Path("/opt/vllm-byog/build-info.json")
    summary["build"] = (
        json.loads(build_info.read_text()) if build_info.exists() else None
    )
    try:
        from vllm.transformers_utils.repo_utils import hf_api

        tokenizer = hf_api().snapshot_download(
            spec["model"]["id"],
            revision=spec["model"]["revision"],
            allow_patterns=["*.json", "*.model", "*.tiktoken", "*vocab*", "merges.txt"],
        )
        jobs = plan(spec, output, url, alias, tokenizer)
        write_json(output / "plan.json", {"jobs": jobs})
        config = {
            **spec["serve"]["config"],
            "revision": spec["model"]["revision"],
            "tokenizer-revision": spec["model"]["revision"],
            "host": "127.0.0.1",
            "port": port,
            "served-model-name": alias,
        }
        config_path = output / "server.yaml"
        config_path.write_text(yaml.safe_dump(config))
        command = ["vllm", "serve", spec["model"]["id"], "--config", str(config_path)]
        environment = dict(os.environ)
        for key, value in spec["serve"]["env"].items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        write_json(
            output / "server.json",
            {"command": command, "environment_overrides": spec["serve"]["env"]},
        )
        with process(command, output / "server.log", environment) as server:
            ready(server, url, alias, spec["serve"]["startup_timeout_seconds"])
            write_json(output / "probe.json", probe(url, alias, spec))
            for job in jobs:
                if server.poll() is not None:
                    raise RuntimeError("Server exited during suite execution")
                with process(
                    job["command"], output / f"{job['name']}.log", environment
                ) as client:
                    try:
                        code = client.wait(
                            timeout=spec["serve"]["client_timeout_seconds"]
                        )
                    except subprocess.TimeoutExpired:
                        code = None
                report_path = Path(job["report"])
                report = (
                    json.loads(report_path.read_text())
                    if report_path.exists()
                    else None
                )
                succeeded = (
                    code == 0
                    and isinstance(report, dict)
                    and report.get("schema") == "byog.measurement/v1"
                    and all(
                        isinstance(report.get(key), dict)
                        for key in ("metrics", "records", "settings")
                    )
                    and report.get("status") == "completed"
                )
                summary["jobs"].append(
                    {
                        "name": job["name"],
                        "exit_code": code,
                        "status": "completed" if succeeded else "failed",
                        "report": report_path.name if report else None,
                    }
                )
                write_json(output / "result.json", summary)
        summary["status"] = (
            "completed"
            if all(j["status"] == "completed" for j in summary["jobs"])
            else "partial"
        )
    except (Exception, KeyboardInterrupt) as exc:
        summary["status"] = "partial" if summary["jobs"] else "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["finished_at"] = datetime.now(UTC).isoformat()
        summary["hardware_after"] = hardware()
        write_json(output / "result.json", summary)
        manifest = {
            str(p.relative_to(output)): {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "size": p.stat().st_size,
            }
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", {"files": manifest})
        with tarfile.open(output / "bundle.tar.gz", "w:gz") as archive:
            for name in [*manifest, "manifest.json"]:
                archive.add(output / name, arcname=name, recursive=False)
    return 0 if summary["status"] == "completed" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # CI images set this to 'error'; the resolved server environment owns it.
    os.environ.pop("VLLM_GPU_SYNC_CHECK", None)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(
        run(resolve(args.spec, args.override), args.output.resolve(), args.dry_run)
    )


if __name__ == "__main__":
    main()
