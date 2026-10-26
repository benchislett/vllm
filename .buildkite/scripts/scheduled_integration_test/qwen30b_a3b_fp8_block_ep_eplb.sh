#!/usr/bin/env bash
set -euxo pipefail

uv pip install --system "evalscope[openai_mrcr]==1.14.0"

# args: [THRESHOLD] [NUM_QUESTIONS] [START_PORT] [DATA_PARALLEL_SIZE] [TENSOR_PARALLEL_SIZE]
THRESHOLD=${1:-0.8}
NUM_Q=${2:-1319}
PORT=${3:-8020}
DATA_PARALLEL_SIZE=${4:-2}
TENSOR_PARALLEL_SIZE=${5:-2}
OUT_DIR=${OUT_DIR:-/tmp/vllm-scheduled}
mkdir -p "${OUT_DIR}"

wait_for_server() {
  local port=$1
  timeout 600 bash -c '
    until curl -sf "http://127.0.0.1:'"$port"'/health" > /dev/null; do
      sleep 1
    done'
}

MODEL="Qwen/Qwen3-30B-A3B-FP8"
# Set BACKENDS based on platform
if command -v rocm-smi &> /dev/null || [[ -d /opt/rocm ]] || [[ -n "${ROCM_PATH:-}" ]]; then
  # ROCm platform
  BACKENDS=("allgather_reducescatter")
  # Disable MOE padding for ROCm since it is causing eplb to fail
  export VLLM_ROCM_MOE_PADDING=0
else
  # Non-ROCm platform (CUDA/other)
  BACKENDS=("deepep_high_throughput" "deepep_low_latency")
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for BACK in "${BACKENDS[@]}"; do
  VLLM_DEEP_GEMM_WARMUP=skip \
  vllm serve "$MODEL" \
    --enforce-eager \
    --enable-eplb \
    --all2all-backend "$BACK" \
    --eplb-config '{"window_size":10, "step_interval":100, "num_redundant_experts":0, "log_balancedness":true, "use_async":false}' \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --enable-expert-parallel \
    --trust-remote-code \
    --max-model-len 2048 \
    --port "$PORT" &
  SERVER_PID=$!
  wait_for_server "$PORT"

  TAG=$(echo "$MODEL" | tr '/: \\n' '_____')
  OUT="${OUT_DIR}/${TAG}_${BACK}.json"
  vllm eval gsm8k --model "$MODEL" --api-url "http://127.0.0.1:${PORT}/v1" \
    --limit "${NUM_Q}" --dataset-hub huggingface \
    --generation-config '{"max_tokens":256,"temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
    --work-dir "$(mktemp -d "${OUT_DIR}/gsm8k-XXXXXX")" --save-results "${OUT}"
  python3 - <<PY
import json; acc=json.load(open('${OUT}'))['metrics']['gsm8k']['score']
print(f"${MODEL} ${BACK}: accuracy {acc:.3f}")
assert acc >= ${THRESHOLD}, f"${MODEL} ${BACK} accuracy {acc}"
PY

  cleanup
  SERVER_PID=
  sleep 1
  PORT=$((PORT+1))
done
