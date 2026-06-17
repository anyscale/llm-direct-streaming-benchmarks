#!/usr/bin/env bash
# Start vLLM-router P/D disaggregation with 4 prefill DP ranks and 4 decode DP ranks.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/vllm_pd4p4d"

POLICY_RAW="${1:-${PROXY_POLICY:-consistent_hash}}"
case "$POLICY_RAW" in
  pow2) POLICY="power_of_two" ;;
  prefix_aware) POLICY="cache_aware" ;;
  random|round_robin|cache_aware|power_of_two|consistent_hash) POLICY="$POLICY_RAW" ;;
  *) echo "ERROR: unknown policy $POLICY_RAW" >&2; exit 2 ;;
esac

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-81920}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
RENDERER_NUM_WORKERS="${RENDERER_NUM_WORKERS:-1}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"max_position_embeddings\":$MAX_MODEL_LEN,\"model_max_length\":$MAX_MODEL_LEN}}"

PREFILL_DP="${PREFILL_DP:-4}"
DECODE_DP="${DECODE_DP:-4}"
PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"
PROXY_PORT="${PROXY_PORT:-8000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29000}"
PREFILL_BASE_PORT="${PREFILL_BASE_PORT:-9100}"
DECODE_BASE_PORT="${DECODE_BASE_PORT:-9200}"
PREFILL_RPC_PORT="${PREFILL_RPC_PORT:-13345}"
DECODE_RPC_PORT="${DECODE_RPC_PORT:-13346}"
PREFILL_NIXL_BASE="${PREFILL_NIXL_BASE:-5600}"
DECODE_NIXL_BASE="${DECODE_NIXL_BASE:-5700}"
PREFILL_DP_MASTER_PORT="${PREFILL_DP_MASTER_PORT:-29400}"
DECODE_DP_MASTER_PORT="${DECODE_DP_MASTER_PORT:-29500}"

IFS=',' read -r -a prefill_gpu_arr <<< "$PREFILL_GPUS"
IFS=',' read -r -a decode_gpu_arr <<< "$DECODE_GPUS"
(( ${#prefill_gpu_arr[@]} == PREFILL_DP )) || { echo "PREFILL_GPUS count must equal PREFILL_DP" >&2; exit 2; }
(( ${#decode_gpu_arr[@]} == DECODE_DP )) || { echo "DECODE_GPUS count must equal DECODE_DP" >&2; exit 2; }

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$LOG_DIR/vllm_cache}"
mkdir -p "$VLLM_CACHE_ROOT"

check_port() {
  local port="$1"
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: port $port already in use. Run serve/vllm_pd4p4d/stop.sh first." >&2
    exit 1
  fi
}

ports=("$PROXY_PORT" "$PROMETHEUS_PORT" "$PREFILL_RPC_PORT" "$DECODE_RPC_PORT" "$PREFILL_DP_MASTER_PORT" "$DECODE_DP_MASTER_PORT")
for ((i=0; i<PREFILL_DP; i++)); do ports+=("$((PREFILL_BASE_PORT + i))"); done
for ((i=0; i<DECODE_DP; i++)); do ports+=("$((DECODE_BASE_PORT + i))"); done
for port in "${ports[@]}"; do check_port "$port"; done

prefill_urls=()
decode_urls=()
for ((i=0; i<PREFILL_DP; i++)); do prefill_urls+=("http://127.0.0.1:$((PREFILL_BASE_PORT + i))"); done
for ((i=0; i<DECODE_DP; i++)); do decode_urls+=("http://127.0.0.1:$((DECODE_BASE_PORT + i))"); done

launch_rank() {
  local role="$1" dp_size="$2" dp_rank="$3" gpu="$4" port="$5" rpc_port="$6" nixl_base="$7" kv_role="$8" master_port="$9"
  local kv_cfg="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$kv_role\"}"
  local cmd=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$port"
    --data-parallel-size "$dp_size"
    --data-parallel-rank "$dp_rank"
    --data-parallel-address 127.0.0.1
    --data-parallel-rpc-port "$rpc_port"
    --enable-expert-parallel
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --tokenizer-mode "$TOKENIZER_MODE"
    --api-server-count "$API_SERVER_COUNT"
    --renderer-num-workers "$RENDERER_NUM_WORKERS"
    --hf-overrides "$HF_OVERRIDES"
    --enable-prefix-caching
    --trust-remote-code
    --disable-uvicorn-access-log
    --kv-transfer-config "$kv_cfg"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_NIXL_SIDE_CHANNEL_PORT=%q VLLM_DP_MASTER_PORT=%q ' "$gpu" "$nixl_base" "$master_port"
    printf '%q ' "${cmd[@]}"
    echo
  else
    echo "$role rank=$dp_rank gpu=$gpu port=$port"
    setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
      VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_base" VLLM_DP_MASTER_PORT="$master_port" \
      "${cmd[@]}" > "$LOG_DIR/${role}_dp${dp_rank}.log" 2>&1 &
    echo "$!" >> "$LOG_DIR/worker.pids"
  fi
}

rm -f "$LOG_DIR/worker.pids"
for ((i=0; i<PREFILL_DP; i++)); do
  launch_rank prefill "$PREFILL_DP" "$i" "${prefill_gpu_arr[$i]}" "$((PREFILL_BASE_PORT + i))" "$PREFILL_RPC_PORT" "$PREFILL_NIXL_BASE" "${PREFILL_KV_ROLE:-kv_producer}" "$PREFILL_DP_MASTER_PORT"
done
for ((i=0; i<DECODE_DP; i++)); do
  launch_rank decode "$DECODE_DP" "$i" "${decode_gpu_arr[$i]}" "$((DECODE_BASE_PORT + i))" "$DECODE_RPC_PORT" "$DECODE_NIXL_BASE" "${DECODE_KV_ROLE:-kv_consumer}" "$DECODE_DP_MASTER_PORT"
done

router_cmd=(
  vllm-router
  --host 0.0.0.0
  --port "$PROXY_PORT"
  --vllm-pd-disaggregation
  --kv-connector nixl
  --policy "$POLICY"
  --prometheus-port "$PROMETHEUS_PORT"
  --request-timeout-secs "${REQUEST_TIMEOUT_SECS:-1800}"
  --log-dir "$LOG_DIR/router_runtime"
)
for url in "${prefill_urls[@]}"; do router_cmd+=(--prefill "$url"); done
for url in "${decode_urls[@]}"; do router_cmd+=(--decode "$url"); done
[[ "$POLICY" == "consistent_hash" ]] && router_cmd+=(--request-id-headers x-correlation-id)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${router_cmd[@]}"
  echo
  exit 0
fi

deadline=$(( $(date +%s) + 1500 ))
for url in "${prefill_urls[@]}" "${decode_urls[@]}"; do
  until curl -fsS --max-time 5 "$url/v1/models" 2>/dev/null | grep -q "$MODEL"; do
    if (( $(date +%s) > deadline )); then
      echo "ERROR: $url not ready. See $LOG_DIR/*.log" >&2
      exit 1
    fi
    sleep 3
  done
  echo "$url ready"
done

"${router_cmd[@]}" > "$LOG_DIR/proxy.log" 2>&1 &
echo "$!" > "$LOG_DIR/proxy.pid"
for _ in {1..60}; do
  if curl -fsS --max-time 5 "http://127.0.0.1:$PROXY_PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
    echo "vLLM P/D router ready: http://127.0.0.1:$PROXY_PORT"
    exit 0
  fi
  sleep 1
done
tail -40 "$LOG_DIR/proxy.log" >&2
exit 1
