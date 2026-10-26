# GSM8K Accuracy Evaluation

GSM8K uses the standard EvalScope four-shot prompt and boxed-answer scorer.
Server tests invoke `vllm eval gsm8k` through the shared evaluation launcher;
the server applies the model's chat template. Offline engine tests use the same
EvalScope adapter with `LLM.chat()`. The raw-completions/last-integer scorer and
standalone legacy script have been removed.

Install `pip install 'vllm[eval]'`, then run:

```bash
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.yaml

# Or evaluate an already-running server:
vllm eval gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
    --api-url http://localhost:8000/v1 --dataset-hub huggingface \
    --generation-config '{"max_tokens":1024,"temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
    --work-dir results/gsm8k
```

## Model-test configuration

```yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct
accuracy_threshold: 0.54  # Historical value; revalidate after this migration.
num_questions: 1319
max_tokens: 1024
temperature: 0
chat_template_kwargs:
  enable_thinking: false
server_args: "--max-model-len 4096 --tensor-parallel-size 2"
```

The test defaults to 1,319 questions, 1,024 output tokens, greedy sampling, seed
42, concurrency 32 and thinking disabled through the template argument. Model
configs can override generation settings and server arguments. Four-shot prompt
construction and answer extraction belong to EvalScope; `num_fewshot`,
`gen_prefix`, and `use_chat_completions` are no longer model-config switches.
Use a checkpoint with a chat template. The two base Mixtral MoE configurations
now select the Instruct sibling for that reason. The two scheduled DeepSeek-V2
nightly scripts likewise use the Chat checkpoint. Scheduled scripts keep their
256-token output budget and read scores from the standardized report.

This changes prompts, generation mode, scoring, and some default settings.
**Historical accuracy and speculative-acceptance thresholds remain unchanged
but require GPU revalidation.** They are not claimed to be equivalent baselines.
Choose model-specific output budgets and thinking settings for reproductions.

Server tests retain reports and raw artifacts under pytest's output directory.
The offloading test retains its two-run/cache-reset sequence. Offline speculative
checks retain their engine and acceptance assertions, but now generate through
`LLM.chat()` without legacy completion stop strings. No external server is
started for those offline checks. See the
[evaluation guide](../../../docs/benchmarking/evaluation.md).
