# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K client tests use real HTTP and local data; no inference engine needed."""

import asyncio
import hashlib
import json
import sys

import pytest
from aiohttp import web

from vllm.evals.gsm8k import _build_gsm8k_prompts, _score_gsm8k


def test_packaged_evaluator_preserves_prompts_and_scoring():
    from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k as legacy
    from vllm.evals.gsm8k import evaluate_gsm8k

    assert legacy is evaluate_gsm8k
    data = (
        [{"question": "1+1?", "answer": "Two. #### 2"}],
        [{"question": "6*7?", "answer": "#### 42"}],
    )
    prompts, labels = _build_gsm8k_prompts(1, 1, data=data)
    assert prompts == ["Question: 1+1?\nAnswer: Two. #### 2\n\nQuestion: 6*7?\nAnswer:"]
    assert labels == [42]
    result = _score_gsm8k(["The answer is 42."], [10], labels, 1, 128, 2.0)
    assert result["accuracy"] == 1.0
    assert result["invalid_rate"] == 0.0
    assert result["tokens_per_second"] == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize("chat,fail", [(True, False), (False, True)])
async def test_eval_cli_records_protocol_dataset_and_transport_failures(
    tmp_path, chat, fail
):
    row = {"question": "What is six times seven?", "answer": "#### 42"}
    for split in ("train", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(
                json.dumps({**row, "question": f"{row['question']} item{i}"}) + "\n"
                for i in range(2)
            )
        )
    config = tmp_path / "eval.yaml"
    config.write_text(
        "num-questions: 99\nnum-shots: 1\nmax-tokens: 128\ntemperature: 0.0\n"
    )
    received = []

    async def response(request):
        payload = await request.json()
        received.append(payload)
        if fail and "item1" in payload.get("prompt", ""):
            return web.Response(status=503, text="fixture unavailable")
        choice = (
            {"message": {"content": "42", "reasoning": "999"}}
            if chat
            else {"text": "42"}
        )
        return web.json_response(
            {"choices": [choice], "usage": {"completion_tokens": 7}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions" if chat else "/v1/completions", response)
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
            "gsm8k",
            "--config",
            str(config),
            "--data-dir",
            str(tmp_path),
            "--num-questions",
            "2",
            "--max-concurrency",
            "1",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--report-format",
            "byog",
            "--save-results",
            str(tmp_path / "result.json"),
        ]
        if chat:
            command += [
                "--use-chat-completions",
                "--chat-template-kwargs",
                '{"enable_thinking": false}',
            ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert process.returncode == (1 if fail else 0), output.decode()
    finally:
        await runner.cleanup()
    report = json.loads((tmp_path / "result.json").read_text())
    assert len(received) == 2
    assert all(p["max_tokens"] == 128 and p["temperature"] == 0.0 for p in received)
    assert ("stop" not in received[0]) if chat else ("stop" in received[0])
    if chat:
        assert received[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert report["schema"] == "byog.measurement/v1"
    assert report["server"] is None
    assert report["status"] == ("partial" if fail else "completed")
    assert report["metrics"]["accuracy"] == (0.5 if fail else 1.0)
    assert report["metrics"]["failed_requests"] == int(fail)
    assert (
        report["settings"]["dataset"]["sha256"]["test"]
        == hashlib.sha256((tmp_path / "test.jsonl").read_bytes()).hexdigest()
    )
    records = report["records"]["questions"]
    assert len(records) == 2
    assert records[0]["correct"] and records[0]["completion_tokens"] == 7
    assert (records[1]["error"] is not None) == fail
