# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for workload resolution and the measurement interchange format."""

import json

import pytest


@pytest.mark.asyncio
async def test_cli_preserves_partial_chat_report(tmp_path):
    """Exercise parsing, request generation, metrics and file output without GPUs."""
    import asyncio
    import os
    import sys

    from aiohttp import web
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import TokenizersBackend

    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(128)}}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    TokenizersBackend(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        tokenizer_dir
    )
    received = []

    async def completions(request):
        payload = await request.json()
        received.append(payload)
        if len(received) == 4:
            return web.Response(status=500, text="injected request failure")
        events = [
            {"choices": [{"delta": {"content": "word1 "}}]},
            {"choices": [{"delta": {"content": "word2"}}]},
            {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2}},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return web.Response(
            text=body + "data: [DONE]\n\n", content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    cache_enabled = False

    async def metrics(request):
        return web.Response(
            text=(
                "vllm:cache_config_info{"
                f'enable_prefix_caching="{cache_enabled}",engine="0"}} 1\n'
                'vllm:prefix_cache_hits_total{engine="0"} 0\n'
            )
        )

    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        client = (
            [os.environ["VLLM_TEST_RUST_BENCH"]]
            if os.environ.get("VLLM_TEST_RUST_BENCH")
            else [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve"]
        )
        command = [
            *client,
            "--model",
            str(tokenizer_dir),
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--workload",
            "8k1k-v1",
            "--max-concurrency",
            "1",
            "--ready-check-timeout-sec",
            "5",
            "--report-format",
            "byog",
            "--result-dir",
            str(tmp_path),
            "--result-filename",
            "report.json",
            "--header",
            "Authorization=secret",
        ]

        async def invoke(*extra):
            process = await asyncio.create_subprocess_exec(
                *command,
                *extra,
                env={**os.environ, "VLLM_USE_RUST_BENCH": "0", "HF_HUB_OFFLINE": "1"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            return process.returncode, output.decode()

        code, output = await invoke("--input-len", "8")
        assert code != 0 and "input-len" in output and not received
        cache_enabled = True
        code, output = await invoke()
        assert code != 0 and "prefix-caching" in output and not received
        cache_enabled = False
        code, output = await invoke()
        assert code == 0, output
    finally:
        await runner.cleanup()
    report_text = (tmp_path / "report.json").read_text()
    report = json.loads(report_text)
    assert "secret" not in report_text
    assert len(received) == 22  # readiness, one warmup, max(20, 10 * 1) measured
    assert all(
        819 <= p["max_completion_tokens"] <= 1229
        and "messages" in p
        and "prompt" not in p
        and p["ignore_eos"]
        for p in received
    )
    assert report["schema"] == "byog.measurement/v1"
    assert report["status"] == "partial"
    assert report["metrics"]["completed"] == 19
    assert len(report["records"]["errors"]) == 20
    assert report["settings"]["workload"]["resolved"]["random_input_len"] == 8192
