# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real EvalScope adapters with tiny local datasets and an HTTP server."""

import asyncio
import json
import sys
from typing import Any

import pytest
from aiohttp import web


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_eval_cli_multiple_adapters_and_partial_results(tmp_path, fail):
    rows: dict[str, dict[str, Any]] = {
        "gsm8k": {"question": "six times seven?", "answer": "#### 42"},
        "aime25": {"problem": "AIME fixture: six times seven?", "answer": "42"},
        "gpqa_diamond": {
            "Question": "GPQA fixture: which is correct?",
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong1",
            "Incorrect Answer 2": "wrong2",
            "Incorrect Answer 3": "wrong3",
        },
    }
    rows["openai_mrcr"] = {
        "prompt": json.dumps([{"role": "user", "content": "MRCR fixture"}]),
        "answer": "PREFIXanswer",
        "random_string_to_prepend": "PREFIX",
        "n_needles": 8,
    }
    dataset_args = {}
    for name, row in rows.items():
        path = tmp_path / name
        path.mkdir()
        split = "train" if name in ("gpqa_diamond", "openai_mrcr") else "test"
        (path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        dataset_args[name] = {
            "local_path": str(path),
            "few_shot_num": 0,
            "subset_list": ["default"],
        }
    config = tmp_path / "eval.yaml"
    config.write_text(
        json.dumps(
            {
                "model_name": "fixture",
                "datasets": {name: {"metric_threshold": 0.9} for name in rows},
                "server_args": "--tensor-parallel-size 4",
                "startup_max_wait_seconds": 1800,
                "generation_config": {
                    "temperature": 0,
                    "max_tokens": 128,
                    "retries": 1,
                },
                "dataset_args": dataset_args,
                "dataset_hub": "local",
                "api_key": "secret-fixture-key",
            }
        )
    )
    received = []

    async def response(request):
        payload = await request.json()
        received.append(payload)
        prompt = str(payload["messages"])
        if fail and "AIME fixture" in prompt:
            return web.json_response(
                {"error": {"message": "fixture failure"}}, status=400
            )
        answer = "\\boxed{42}"
        if "GPQA fixture" in prompt:
            import regex as re

            match = re.search(r"([A-D])\) correct", prompt)
            if match is None:
                match = re.search(r"([A-D])\. correct", prompt)
            assert match is not None, prompt
            answer = f"ANSWER: {match.group(1)}"
        if "MRCR fixture" in prompt:
            answer = "PREFIXanswer"
        return web.json_response(
            {
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", response)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "eval",
            "--eval-config",
            str(config),
            "--api-url",
            f"http://127.0.0.1:{port}/v1",
            "--eval-batch-size",
            "1",
            "--work-dir",
            str(tmp_path / "output"),
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=120)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert process.returncode == int(fail), output.decode()
    finally:
        await runner.cleanup()
    report = json.loads((tmp_path / "output/result.json").read_text())
    assert report["status"] == ("partial" if fail else "completed")
    assert report["schema"] == "byog.measurement/v1"
    assert report["server"] is None
    assert report["metrics"]["gsm8k"]["score"] == 1
    assert report["metrics"]["gpqa_diamond"]["score"] == 1
    assert report["metrics"]["openai_mrcr"]["score"] == 1
    if not fail:
        assert report["metrics"]["aime25"]["score"] == 1
    assert bool(report["records"]["errors"]) == fail
    assert report["records"]["artifacts"]
    assert "secret-fixture-key" not in json.dumps(report)
    assert all(p["max_tokens"] == 128 and p["temperature"] == 0 for p in received)
    assert report["settings"]["tasks"]["gsm8k"]["eval_batch_size"] == 1


def test_mrcr_presets_filter_needles_and_context(tmp_path):
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    from vllm.evals.runner import PRESETS

    path = tmp_path / "mrcr"
    path.mkdir()
    (path / "train.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "prompt": json.dumps([{"role": "user", "content": " x" * length}]),
                    "answer": "PREFIXanswer",
                    "random_string_to_prepend": "PREFIX",
                    "n_needles": needles,
                }
            )
            + "\n"
            for needles, length in [
                (2, 100000),
                (8, 65536),
                (8, 100000),
                (8, 131072),
                (8, 200000),
                (8, 262144),
                (8, 262145),
            ]
        )
    )
    for name, preset in PRESETS.items():
        task = TaskConfig(
            model="fixture",
            api_url="http://localhost/v1",
            datasets=["openai_mrcr"],
            dataset_hub="local",
            dataset_args={
                "openai_mrcr": {
                    "local_path": str(path),
                    "extra_params": preset["extra_params"],
                }
            },
        )
        adapter = get_benchmark("openai_mrcr", task)
        data = adapter.load()
        samples = list(data[0]["default"])
        expected = [100000, 131072] if name.endswith("128k") else [200000, 262144]
        assert [s.metadata["raw_input_tok_cnt"] for s in samples] == expected
        assert all(s.metadata["n_needles"] == 8 for s in samples)
