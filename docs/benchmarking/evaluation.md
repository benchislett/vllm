# Evaluating a running server

`vllm eval` runs [EvalScope](https://github.com/modelscope/evalscope) against an
existing OpenAI-compatible chat server. Install `pip install 'vllm[eval]'`
(EvalScope 1.14.0). Evaluation dependencies are loaded only when running an eval;
the client needs no GPU and does not start the server.

```bash
vllm eval gsm8k gpqa_diamond aime25 --model YOUR_SERVED_MODEL \
    --api-url http://localhost:8000/v1 --eval-batch-size 32 \
    --generation-config '{"max_tokens":32768,"temperature":0.7,"top_p":0.8}' \
    --work-dir results/math-science
```

EvalScope owns dataset loading, prompts, answer extraction and scoring. Other
registered EvalScope dataset names work through the same interface. GSM8K uses
EvalScope's default 4-shot prompt and boxed-answer scorer; it is **not directly
comparable** to the older five-shot, last-integer evaluator in
`tests/evals/gsm8k/gsm8k_eval.py`, which remains unchanged. GPQA-Diamond uses the
198-question diamond subset, and AIME25 uses the 2025 competition questions.
Default dataset sources and few-shot settings come from the installed adapter.

Choose sampling, thinking settings and output budgets for the model. Use
`generation_config.extra_body.chat_template_kwargs` for model-specific chat
options. `--limit 32` makes a smoke test, not a reference score. The MRCR presets apply limits after needle/context filtering; generic
EvalScope datasets retain their adapter's limit behavior. A successful exit means a complete scored run, not that an
accuracy threshold was met. No LLM judge is configured by this launcher.

## MRCR v2, eight needles

```bash
vllm eval mrcr-v2-8needle-128k mrcr-v2-8needle-256k \
    --model YOUR_SERVED_MODEL --api-url http://localhost:8000/v1 \
    --generation-config '{"max_tokens":4096,"temperature":0}' \
    --work-dir results/mrcr
```

These presets use EvalScope's `openai_mrcr` adapter for OpenAI's
[multi-round co-reference benchmark](https://huggingface.co/datasets/openai/mrcr)
(often called MRCR v2), with `needle_count: [8]`. The 128k preset selects input
lengths **(65,536, 131,072]**; 256k selects **(131,072, 262,144]**. They are disjoint
input-length bins, not exact-length synthetic prompts or cumulative “up to”
scores. EvalScope counts message content with `o200k_base`, without server chat
framing. The serving model's tokenizer can yield different lengths; configure
sufficient context space for its prompt **and output**. The adapter's detailed
score bins include output tokens and can differ from these input filters.

Each preset runs in its own artifact directory, so both lengths can be selected
in one invocation. `--dataset-args` can override the underlying `openai_mrcr`
adapter's `extra_params`; overrides are recorded, so an overridden run should
not be presented as an unchanged preset result.

## Reusing evaluation configs

`--eval-config` accepts EvalScope task YAML, including `datasets`, `dataset_args`,
`generation_config`, `seed`, `repeats`, `limit` and `eval_batch_size`. It also
accepts the existing Qwen accuracy configuration shape:

```yaml
model_name: Qwen/Qwen3.8-Flash-Next-FP8
datasets:
  gsm8k: {metric_threshold: 0.98, tolerance: 0.02}
  aime25: {metric_threshold: 0.90, tolerance: 0.05}
eval_batch_size: 32
generation_config:
  seed: 1236
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  max_tokens: 32768
  extra_body:
    chat_template_kwargs: {enable_thinking: true}
```

```bash
vllm eval --eval-config tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8.yaml \
    --api-url http://localhost:8000/v1 --work-dir results/qwen
```

The launcher reads `model_name` and dataset names from these CI configs.
`server_args`, `env`, startup timeouts and metric thresholds belong to the CI
caller and are not executed or enforced here. Explicit CLI arguments replace
the corresponding config field; positional names replace the dataset list.
JSON options replace the whole object rather than merging it. The launcher
always uses fresh remote chat evaluations; cached prediction reuse is rejected.
Use `OPENAI_API_KEY` for authentication, or the EvalScope config's `api_key`.

Dataset sources are not automatically pinned. For reproducibility, set
`dataset_args.DATASET.dataset_revision` (or `hf_dataset_revision` with
`dataset_hub: huggingface`) to an immutable revision. A local dataset can be
selected with `dataset_hub: local` and `dataset_args.DATASET.local_path`, using the adapter's row schema
and EvalScope's split naming. A frozen data snapshot is needed for offline runs.

## Reports and failures

`--work-dir` must be empty. Without it, a unique directory is created under
`eval_results/`. EvalScope keeps its configs, evaluation fingerprints, per-sample
predictions, reviews, and native reports in per-evaluation subdirectories.
`result.json` contains the shared `byog.measurement/v1` envelope: EvalScope's
version, resolved task settings, per-evaluation score and count, full native
reports, artifact SHA256 hashes, and completion status. API keys are masked in
the task snapshot. `--save-results PATH` selects another summary path;
`--report-format evalscope` writes native reports there instead.

Copy the work directory with the summary: large per-sample records remain in
EvalScope's artifact files. Treat both as raw evaluation data when sharing.
A failed, empty or incomplete evaluation produces a partial BYOG report and a
nonzero exit code; subsequent evaluations still run. Valid wrong answers do not
make a run partial. Setup/import errors can fail before a report exists.
Remote server provenance remains unknown. This command does not certify model
correctness, upload results, or update PRs.

## Existing accuracy tests

The Qwen GSM8K/AIME25, GPT-OSS GPQA-Diamond, and MRCR tests launch this command
through `tests/evals/utils.py`, retaining each invocation's config and artifacts.
GPT-OSS retains eight repeats, reasoning effort and its 32k output budget. MRCR
runs each needle count separately, balancing the sample budget and preserving
per-needle threshold assertions; it limits samples after filtering.

A small MRCR adapter subclass enables EvalScope's post-filter sample limiting;
its prompts and scoring remain EvalScope-owned. The GPQA and MRCR adapter
protocols change with this migration. Qwen also updates from EvalScope 1.10.0
to the common 1.14.0 dependency. Their historical
thresholds are retained, not recalibrated here: rerun model accuracy before
claiming equivalence to the old baselines. MRCR now uses EvalScope's `o200k_base`
content length instead of the serving model's `/tokenize` count; configure
`max_prompt_tokens` with adequate server context headroom. EvalScope may load a
full cached dataset snapshot rather than the previous streaming subset.

The legacy `tests/evals/gsm8k/` pathway (including offloading tests) is deliberately
unchanged. Migrating it to a standardized eval may be desirable later, but its
raw-completions/five-shot/scoring protocol needs separate accuracy validation.
