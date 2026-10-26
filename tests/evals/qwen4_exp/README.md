# Qwen4Exp accuracy evaluation

This suite starts a Qwen3.8-Flash-Next-FP8 OpenAI-compatible server once and
uses EvalScope to evaluate GSM8K and AIME25.

```bash
# B200
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
  --config-list-file=configs/models-b200.txt

# H200
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
  --config-list-file=configs/models-h200.txt
```

The test launches `vllm eval --eval-config` with the same config and retains its
BYOG/native artifacts under pytest's temporary output directory. Install
`pip install "vllm[eval]"` for the pinned EvalScope client.

This migration pins EvalScope 1.14.0 (previous CI used 1.10.0). Historical
thresholds remain unchanged and need model accuracy revalidation; see the
[evaluation guide](../../../docs/benchmarking/evaluation.md#existing-accuracy-tests).
