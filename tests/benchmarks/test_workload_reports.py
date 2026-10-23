# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for workload resolution and the measurement interchange format."""

import json

import pytest

from vllm.benchmarks.serve import add_cli_args
from vllm.benchmarks.workloads import resolve_workload
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse(*options):
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(list(options))
    resolve_workload(args)
    return args


@pytest.mark.parametrize("preset_first", [True, False])
def test_explicit_legacy_defaults_override_preset_in_either_order(preset_first):
    preset = ["--workload", "8k1k-v1", "--max-concurrency", "16"]
    overrides = ["--random-output-len", "128", "--num-warmups", "0"]
    args = parse(*(preset + overrides if preset_first else overrides + preset))
    assert args.random_input_len == 8192
    assert args.random_output_len == 128
    assert args.num_warmups == 0
    assert args.backend == "openai-chat"
    assert args.random_range_ratio == 0.2


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
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--model",
            str(tokenizer_dir),
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--workload",
            "8k1k-v1",
            "--max-concurrency",
            "1",
            "--input-len",
            "8",
            "--output-len",
            "2",
            "--random-range-ratio",
            "0",
            "--num-prompts",
            "3",
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
        assert process.returncode == 0, output.decode()
    finally:
        await runner.cleanup()
    report_text = (tmp_path / "report.json").read_text()
    report = json.loads(report_text)
    assert "secret" not in report_text
    assert len(received) == 5  # readiness request, one warmup, three measured
    assert all(
        p["max_completion_tokens"] == 2
        and "messages" in p
        and "prompt" not in p
        and p["ignore_eos"]
        for p in received
    )
    assert report["schema"] == "byog.measurement/v1"
    assert report["status"] == "partial"
    assert report["metrics"]["completed"] == 2
    assert len(report["records"]["errors"]) == 3
    assert report["settings"]["workload"]["modified"]
    assert report["settings"]["workload"]["resolved"]["random_input_len"] == 8
