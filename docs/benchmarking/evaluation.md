# Evaluating a running server

`vllm eval gsm8k` runs the existing GSM8K few-shot evaluator against a server.
The client does not need a GPU and does not start or configure the server.

```bash
vllm eval gsm8k --base-url http://localhost:8000 \
    --use-chat-completions --num-questions 1319 --num-shots 5 \
    --max-tokens 1024 --temperature 0 --max-concurrency 32 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --report-format byog --save-results results/gsm8k.json
```

Choose chat/completion mode, thinking settings, sampling and output budget for
**the model being evaluated**. The command above is an example for a chat model
that supports `enable_thinking`; it is not a universal accuracy reference.
A quick smoke run can use `--num-questions 32`, but its score is not directly
comparable to a full evaluation. The selected questions are the first N test
rows; few-shot examples are the first K train rows, matching the existing
vLLM GSM8K evaluator.

Without `--use-chat-completions`, the evaluator uses raw completions and the
existing stop strings. Chat mode sends no stop strings and scores final answer
content, excluding separately returned reasoning. Temperature defaults to zero
for completions and to the server default for chat. `--temperature`, `--top-p`,
`--top-k`, `--reasoning-effort` and `--chat-template-kwargs` can make those settings
explicit. The scorer preserves the existing last-integer extraction behavior;
this is not an interchangeable implementation of every GSM8K evaluation protocol.

## Configuration and datasets

Client settings can be supplied as YAML:

```yaml
num-questions: 1319
num-shots: 5
use-chat-completions: true
max-tokens: 1024
temperature: 0
max-concurrency: 32
chat-template-kwargs:
  enable_thinking: false
```

Use `vllm eval gsm8k --config eval.yaml`. Explicit command-line options take
precedence over the file. This is a **client** config, not a server recipe or
an existing pytest model config containing `server_args` and accuracy thresholds.

By default, train and test JSONL files come from `openai/grade-school-math` at
commit `3101c7d5072418e28b9008a6636bde82a006892c`. Downloads are cached separately
by URL and published atomically. `--dataset-revision` accepts another full
40-character commit SHA. For offline execution, pass `--data-dir DIR` containing
`train.jsonl` and `test.jsonl`. Local datasets are identified as local rather
than claiming the default revision. Every saved result records both content
SHA256 hashes; questions must fit the selected dataset, and few-shot examples
must fit the train split.

## Reports and failures

`--save-results PATH` writes metrics, request settings, dataset identity and
per-question records. `--report-format byog` requires an output path and wraps
these in the shared `byog.measurement/v1` format: settings and their hash,
metrics, records, and `completed` or `partial` status. Server provenance is
unknown to a remote client. This produces evidence for a suite runner or results
service; it does not upload data, comment on a PR or certify model correctness.

Per-question records retain prompts, final answers, expected/predicted values,
correctness, completion-token usage, and transport/protocol errors. Usage is
zero when the server omits it, preserving the existing evaluator's accounting.
Request failures count against overall accuracy and are separately counted as
`failed_requests`; they produce a partial report and a nonzero exit code after
the report is saved. A valid but incorrect answer does not make the run partial.
No accuracy threshold is implied by a successful exit. Dataset/setup errors fail
before producing an evaluation report. Review raw content before sharing it.

The legacy `python tests/evals/gsm8k/gsm8k_eval.py ...` command delegates to the
same launcher. Existing Python imports and the offline evaluator remain
available at their original test-module path. Programmatic calls retain their
legacy defaults; the CLI now uses pinned datasets, records individual results
and returns a nonzero exit status on request failures.
