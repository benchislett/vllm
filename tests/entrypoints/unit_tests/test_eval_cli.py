# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real EvalScope adapters with tiny local datasets and an HTTP server."""

import asyncio
import json
import subprocess
from typing import Any

import pytest
from aiohttp import web


@pytest.mark.asyncio
async def test_eval_cli_multiple_adapters_and_partial_results(tmp_path):
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
        values = [row]
        if name == "aime25":
            values.append({**row, "problem": "AIME transport failure"})
        if name == "openai_mrcr":
            values = [
                {
                    **row,
                    "n_needles": needles,
                    "prompt": json.dumps(
                        [{"role": "user", "content": "MRCR fixture " + " x" * length}]
                    ),
                }
                for needles, length in (
                    (2, 100000),
                    (8, 100),
                    (8, 100000),
                    (8, 200000),
                    (8, 300000),
                )
            ]
        (path / f"{split}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
        dataset_args[name] = {
            "local_path": str(path),
            "subset_list": ["default"],
        }
        if name == "gsm8k":
            (path / "train.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "question": f"demonstration-{i}: 1+1?",
                            "answer": "Two. #### 2",
                        }
                    )
                    + "\n"
                    for i in range(4)
                )
            )
        else:
            dataset_args[name]["few_shot_num"] = 0
    from tests.evals.gsm8k.gsm8k_eval import gsm8k_task

    gsm_config = gsm8k_task({"model_name": "fixture", "max_tokens": 128})
    config = tmp_path / "eval.yaml"
    config.write_text(
        json.dumps(
            {
                "model_name": "fixture",
                "datasets": {
                    name: {"metric_threshold": 0.9}
                    for name in (
                        "gsm8k",
                        "aime25",
                        "gpqa_diamond",
                        "mrcr-v2-8needle-128k",
                        "mrcr-v2-8needle-256k",
                    )
                },
                "limit": 2,
                "server_args": "--tensor-parallel-size 4",
                "startup_max_wait_seconds": 1800,
                "generation_config": {
                    **gsm_config["generation_config"],
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
        if "AIME transport failure" in prompt:
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
        from tests.evals.utils import run_evaluation

        config_value = json.loads(config.read_text())
        config_value["eval_batch_size"] = 1
        with pytest.raises(subprocess.CalledProcessError) as failure:
            await asyncio.to_thread(
                run_evaluation,
                config_value,
                f"http://127.0.0.1:{port}/v1",
                tmp_path / "output",
                timeout=120,
            )
        assert failure.value.returncode == 1
    finally:
        await runner.cleanup()
    report = json.loads((tmp_path / "output/result.json").read_text())
    assert report["status"] == "partial"
    gsm_requests = [
        p
        for p in received
        if "six times seven?" in str(p["messages"]) and "AIME" not in str(p["messages"])
    ]
    assert len(gsm_requests) == 1
    prompt = str(gsm_requests[0]["messages"])
    assert all(f"demonstration-{i}" in prompt for i in range(4))
    assert "boxed" in prompt
    assert gsm_requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert report["schema"] == "byog.measurement/v1"
    assert report["server"] is None
    for name in (
        "gsm8k",
        "gpqa_diamond",
        "aime25",
        "mrcr-v2-8needle-128k",
        "mrcr-v2-8needle-256k",
    ):
        assert report["metrics"][name] == {"score": 1, "num": 1}
    assert set(report["records"]["errors"]) == {"aime25"}
    assert report["records"]["artifacts"]
    assert "secret-fixture-key" not in json.dumps(report)
    assert all(p["max_tokens"] == 128 and p["temperature"] == 0 for p in received)
    assert report["settings"]["tasks"]["gsm8k"]["eval_batch_size"] == 1
