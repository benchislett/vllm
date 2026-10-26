# MRCR Long-Context Accuracy Evaluation

Smoke test for long-context behavior using OpenAI's public [`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) dataset. The model sees a long chat with several near-duplicate "needles" and must reproduce a specific earlier assistant turn verbatim, prepended with a random anti-guessing string.

**Scoring:** if the response doesn't start with `random_string_to_prepend`, score is 0; otherwise the prefix is stripped and the mean `SequenceMatcher.ratio()` against the reference answer is reported.

## Usage

```bash
# Pytest (spawns the server)
pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
    --config-list-file=configs/models-small.txt

# Standalone (choose a model with sufficient context for the preset)
# Start MODEL with sufficient context for 128k input plus output/template overhead.
vllm serve MODEL --port 8000
vllm eval mrcr-v2-8needle-128k --model MODEL \
    --api-url http://localhost:8000/v1 --work-dir results/mrcr
```

## Configuration

```yaml
model_name: "Qwen/Qwen3-0.6B"
# Per-needle thresholds catch bucket-specific regressions (sliding window,
# chunked prefill, prefix cache) that an aggregate can hide. A scalar
# (e.g. `match_ratio_threshold: 0.20`) is also accepted and checked against
# the mean match ratio.
match_ratio_threshold:
  2: 0.30
  4: 0.15
  8: 0.10
num_samples: 30
needles: [2, 4, 8]
# max_prompt_tokens: 32768       # Optional; defaults to server max_model_len - max_tokens - 256
max_tokens: 2048
concurrency: 8
server_args: "--max-model-len 32768 --reasoning-parser qwen3"
```

## Notes

- Install `pip install "vllm[eval]"`. The test invokes `vllm eval openai_mrcr`
  separately for each needle count, balances `num_samples` across those invocations,
  and retains per-needle threshold checks and artifacts.
- `max_prompt_tokens` defaults to the advertised server context minus output
  budget and 256 tokens. EvalScope counts message content with `o200k_base`,
  not the server tokenizer or chat template. Set a lower explicit cap when needed
  to leave sufficient model-specific context headroom.
- Sample limits apply after needle/length filtering. EvalScope can load the full
  dataset snapshot, unlike the former streaming-shard helper.
- The adapter changes sampling/token counting and may normalize whitespace when
  scoring. Historical thresholds are retained but must be revalidated on the
  configured model; this migration does not establish score equivalence.
- Reasoning models need the appropriate server reasoning parser. The default
  `extra_body` disables thinking through `chat_template_kwargs.enable_thinking`.
- The standalone presets select disjoint 64–128k and 128–256k input bins and
  require a sufficiently large server context. See the
  [evaluation guide](../../../docs/benchmarking/evaluation.md).
