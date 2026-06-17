#!/usr/bin/env bash
# Run the 4-prefill/4-decode Wide-EP P/D agentic trials.
#
# Both backends use their stock P/D path with one config across all
# concurrencies: vLLM-router with --vllm-pd-disaggregation, and Ray Serve LLM
# with the public build_pd_openai_app builder plus the tokenize-once
# optimization (ray-project/ray#64049). Decode runs with no per-replica
# max-ongoing cap so admission is never the bottleneck.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

backend=""
concurrency=(64 128 256)
dry_run=0
no_start=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --concurrency)
      concurrency=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do concurrency+=("$1"); shift; done
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --no-start)
      no_start=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$backend" in
  ray) start_script="serve/ray_pd4p4d/start.sh"; stop_script="serve/ray_pd4p4d/stop.sh"; policy="consistent_hash" ;;
  vllm) start_script="serve/vllm_pd4p4d/start.sh"; stop_script="serve/vllm_pd4p4d/stop.sh"; policy="consistent_hash" ;;
  *) echo "Usage: $0 --backend ray|vllm [--concurrency 64 ...]" >&2; exit 2 ;;
esac

export MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export URL="${URL:-http://127.0.0.1:8000}"
export BASE_DATASET="${BASE_DATASET:-datasets/agentic_loop_2200s_seed42/dataset.jsonl}"
export RESULTS_DIR="${RESULTS_DIR:-artifacts/reruns/agentic_pd4p4d/$backend/$(date -u +%Y%m%d-%H%M%S)}"
export SESSION_HEADER="${SESSION_HEADER:-x-correlation-id}"
export PREFILL_DP="${PREFILL_DP:-4}"
export DECODE_DP="${DECODE_DP:-4}"
export PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
export DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"
# Single config across all concurrencies. Decode has no per-replica max-ongoing
# cap (10000 is effectively unbounded) so admission is never the bottleneck.
export PREFILL_MAX_ONGOING="${PREFILL_MAX_ONGOING:-256}"
export DECODE_MAX_ONGOING="${DECODE_MAX_ONGOING:-10000}"
export PREFILL_KV_ROLE="${PREFILL_KV_ROLE:-kv_producer}"
export DECODE_KV_ROLE="${DECODE_KV_ROLE:-kv_consumer}"

# Ray serves P/D through the public build_pd_openai_app builder with the
# tokenize-once optimization (ray-project/ray#64049). aiperf runs its own
# warmup, so the Serve-level prewarm is left off.
if [[ "$backend" == "ray" ]]; then
  export PD_TOKENIZE_ONCE="${PD_TOKENIZE_ONCE:-1}"
  export PD_PREWARM="${PD_PREWARM:-0}"
fi

stack_up=0
cleanup() {
  if [[ "$stack_up" == "1" && "$no_start" != "1" && "$dry_run" != "1" && "${KEEP_STACK_UP:-0}" != "1" ]]; then
    "$stop_script" || true
    stack_up=0
  fi
}
trap cleanup EXIT

run_group() {
  local label="$1"
  local concurrencies="$2"
  local decode_max_ongoing="$3"
  local prefill_kv_role="$4"
  local decode_kv_role="$5"

  [[ -z "$concurrencies" ]] && return 0

  export CONCURRENCIES="$concurrencies"
  export DECODE_MAX_ONGOING="$decode_max_ongoing"
  export PREFILL_KV_ROLE="$prefill_kv_role"
  export DECODE_KV_ROLE="$decode_kv_role"

  echo
  echo "P/D 4p4d group: $label; concurrencies=$CONCURRENCIES; decode_max_ongoing=$DECODE_MAX_ONGOING; kv=$PREFILL_KV_ROLE/$DECODE_KV_ROLE"
  if [[ "$no_start" != "1" ]]; then
    echo "Starting $backend P/D 4p4d stack with policy=$policy"
    DRY_RUN="$dry_run" MODEL="$MODEL" "$start_script" "$policy"
    if [[ "$dry_run" != "1" ]]; then
      stack_up=1
    fi
  fi

  DRY_RUN="$dry_run" ./bench/_run_agentic_profile_sweep.sh
  cleanup
}

run_group "default" "${concurrency[*]}" "$DECODE_MAX_ONGOING" "$PREFILL_KV_ROLE" "$DECODE_KV_ROLE"
