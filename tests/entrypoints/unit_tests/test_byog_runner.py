# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BYOG control-plane checks; external server/registry services are fixtures."""

import hashlib
import io
import json
import os
import sys

import pytest
import yaml

from tools.byog.request_build import request_build
from tools.byog.run import run
from tools.byog.spec import resolve


def source_spec(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "byog.spec/v0",
                "name": "fixture",
                "model": {"id": "fixture/model", "revision": "a" * 40},
                "serve": {"config": {}, "startup_timeout_seconds": 10},
                "evals": [
                    {"name": "gsm8k", "args": {"limit": 1}},
                    {"name": "mrcr-v2-8needle-128k", "args": {"limit": 1}},
                ],
                "benchmarks": [{"workload": "8k1k-v1", "concurrency": [1, 2]}],
            }
        )
    )
    return path


def test_recipe_hash_and_user_overrides(tmp_path):
    path = source_spec(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "argv": ["vllm", "serve", "fixture/model", "-tp", "2"],
                "env": {"VLLM_TEST": "recipe"},
            }
        )
    )
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "recipe": {
                    "path": "recipe.json",
                    "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                },
                "serve": {
                    "config": {"tensor-parallel-size": 4},
                    "env": {"VLLM_TEST": "custom"},
                },
            }
        )
    )
    resolved = resolve(path, [override])
    assert resolved["serve"]["config"]["tensor-parallel-size"] == 4
    assert resolved["serve"]["env"]["VLLM_TEST"] == "custom"
    assert resolved["serve"]["env"]["VLLM_GPU_SYNC_CHECK"] is None
    recipe.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        resolve(path, [override])


def test_build_request_pins_branch_before_submitting(monkeypatch):
    calls = []

    def service(request, timeout):
        calls.append(request)
        return io.BytesIO(
            json.dumps(
                {"sha": "b" * 40}
                if len(calls) == 1
                else {"web_url": "https://buildkite.com/fixture/byog/1"}
            ).encode()
        )

    monkeypatch.setattr("tools.byog.request_build.urllib.request.urlopen", service)
    result = request_build(
        "benchislett/vllm",
        "feature/test",
        "byog",
        "fixture",
        "github-secret",
        "buildkite-secret",
        "byog-images",
        "arm64",
        "arm-cpu",
    )
    assert calls[0].full_url.endswith("feature%2Ftest")
    payload = json.loads(calls[1].data)
    assert payload["commit"] == "b" * 40
    assert payload["env"]["BYOG_ARCH"] == "arm64"
    assert payload["env"]["BYOG_BUILD_QUEUE"] == "arm-cpu"
    assert "secret" not in json.dumps(payload)
    assert result["source_commit"] == "b" * 40


@pytest.mark.parametrize("fail", [False, True])
def test_runner_preserves_results_and_stops_its_server(tmp_path, monkeypatch, fail):
    # Exercise the actual orchestration/process lifecycle with a tiny external
    # vllm CLI service fixture. Client HTTP correctness is tested separately.
    executable = tmp_path / "vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        r"""
import http.server,json,os,sys,yaml
from pathlib import Path
args=sys.argv[1:]
if args[0]=='serve':
    config=yaml.safe_load(Path(args[args.index('--config')+1]).read_text())
    Path(os.environ['BYOG_TEST_PID']).write_text(str(os.getpid()))
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"choices":[{"message":{"content":"1"}}]}).encode())
        def do_GET(self):
            self.send_response(200); self.end_headers()
            body={'data':[{'id':config['served-model-name']}]}
            self.wfile.write(json.dumps(body).encode())
    http.server.HTTPServer(('127.0.0.1',config['port']),Handler).serve_forever()
else:
    if args[0]=='eval':
        report=Path(args[args.index('--save-results')+1])
        artifacts=Path(args[args.index('--work-dir')+1])
        artifacts.mkdir()
        (artifacts/'predictions.jsonl').write_text('{"answer":"fixture"}\n')
    else:
        report=Path(args[args.index('--result-dir')+1])
        report=report/args[args.index('--result-filename')+1]
    failed=os.environ.get('BYOG_TEST_FAIL')=='1' and args[0]=='eval'
    content={'schema':'byog.measurement/v1', 'metrics':{},'settings':{},'records':{}}
    content['status']='partial' if failed else 'completed'
    report.write_text(json.dumps(content))
    sys.exit(1 if failed else 0)
"""
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("BYOG_TEST_PID", str(tmp_path / "server.pid"))
    monkeypatch.setenv("BYOG_TEST_FAIL", str(int(fail)))
    monkeypatch.setenv("BYOG_DATA_DIR", str(tmp_path))

    class Hub:
        def snapshot_download(self, *args, **kwargs):
            assert kwargs["revision"] == "a" * 40
            return str(tmp_path)

    monkeypatch.setattr("vllm.transformers_utils.repo_utils.hf_api", lambda: Hub())
    spec = resolve(source_spec(tmp_path), [])
    output = tmp_path / "result"
    assert run(spec, output, False) == int(fail)
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == ("partial" if fail else "completed")
    assert len(result["jobs"]) == 4
    planned = json.loads((output / "plan.json").read_text())["jobs"]
    assert "--dataset-args" in planned[0]["command"]
    assert "--dataset-args" not in planned[1]["command"]
    assert all(job["status"] == "completed" for job in result["jobs"][2:])
    assert (output / "bundle.tar.gz").exists()
    assert (output / "eval-0-gsm8k.json").exists()
    assert (output / "eval-1-mrcr-v2-8needle-128k/predictions.jsonl").exists()
    pid = int((tmp_path / "server.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert (
            hashlib.sha256((output / name).read_bytes()).hexdigest() == entry["sha256"]
        )
