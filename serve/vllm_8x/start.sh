#!/usr/bin/env bash
# Start 8 one-GPU vLLM workers behind vllm-router.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/vllm_8x"

POLICY="${1:-round_robin}"
NUM_REPLICAS="${NUM_REPLICAS:-8}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-9000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29000}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-81920}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
RENDERER_NUM_WORKERS="${RENDERER_NUM_WORKERS:-1}"
if [[ -z "${HF_OVERRIDES:-}" && "$MAX_MODEL_LEN" -gt 40960 ]]; then
  HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":40960},"max_position_embeddings":81920}'
fi
LOAD_FORMAT="${LOAD_FORMAT:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

check_port() {
  local port="$1"
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: port $port already in use. Run serve/vllm_8x/stop.sh first." >&2
    exit 1
  fi
}

for ((i=0; i<NUM_REPLICAS; i++)); do check_port "$((WORKER_BASE_PORT + i))"; done
check_port "$ROUTER_PORT"
check_port "$PROMETHEUS_PORT"

worker_urls=()
worker_pids=()
for ((i=0; i<NUM_REPLICAS; i++)); do
  port=$((WORKER_BASE_PORT + i))
  worker_urls+=("http://127.0.0.1:$port")
  cmd=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$port"
    --tensor-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --tokenizer-mode "$TOKENIZER_MODE"
    --api-server-count "$API_SERVER_COUNT"
    --renderer-num-workers "$RENDERER_NUM_WORKERS"
    --disable-uvicorn-access-log
  )
  [[ -n "${HF_OVERRIDES:-}" ]] && cmd+=(--hf-overrides "$HF_OVERRIDES")
  [[ -n "$LOAD_FORMAT" ]] && cmd+=(--load-format "$LOAD_FORMAT")
  if [[ "$ENABLE_PREFIX_CACHING" == "0" ]]; then
    cmd+=(--no-enable-prefix-caching)
  else
    cmd+=(--enable-prefix-caching)
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_FASTOKENS=%q ' "$i" "${VLLM_USE_FASTOKENS:-0}"
    printf '%q ' "${cmd[@]}"
    echo
  else
    echo "worker $i GPU=$i port=$port"
    CUDA_VISIBLE_DEVICES="$i" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_FASTOKENS="${VLLM_USE_FASTOKENS:-0}" \
      "${cmd[@]}" > "$LOG_DIR/worker_${i}.log" 2>&1 &
    worker_pids+=("$!")
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  router_cmd=(vllm-router --host 0.0.0.0 --port "$ROUTER_PORT" --worker-urls "${worker_urls[@]}" --policy "$POLICY" --prometheus-port "$PROMETHEUS_PORT")
  [[ "$POLICY" == "consistent_hash" ]] && router_cmd+=(--request-id-headers x-correlation-id)
  printf '%q ' "${router_cmd[@]}"
  echo
  exit 0
fi

printf '%s\n' "${worker_pids[@]}" > "$LOG_DIR/worker.pids"

deadline=$(( $(date +%s) + 900 ))
for url in "${worker_urls[@]}"; do
  until curl -fsS --max-time 5 "$url/v1/models" 2>/dev/null | grep -q "$MODEL"; do
    if (( $(date +%s) > deadline )); then
      echo "ERROR: $url not ready. See $LOG_DIR/worker_*.log" >&2
      exit 1
    fi
    sleep 3
  done
  echo "$url ready"
done

router_cmd=(
  vllm-router
  --host 0.0.0.0
  --port "$ROUTER_PORT"
  --worker-urls "${worker_urls[@]}"
  --policy "$POLICY"
  --prometheus-port "$PROMETHEUS_PORT"
)
[[ "$POLICY" == "consistent_hash" ]] && router_cmd+=(--request-id-headers x-correlation-id)
"${router_cmd[@]}" > "$LOG_DIR/router.log" 2>&1 &
echo "$!" > "$LOG_DIR/router.pid"

for _ in {1..60}; do
  if curl -fsS --max-time 5 "http://127.0.0.1:$ROUTER_PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
    echo "vllm-router ready: http://127.0.0.1:$ROUTER_PORT"
    exit 0
  fi
  sleep 1
done

echo "ERROR: router failed to become ready. Tail:" >&2
tail -40 "$LOG_DIR/router.log" >&2
exit 1
