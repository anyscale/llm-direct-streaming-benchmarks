#!/usr/bin/env bash
# Run the standard 8x replica agentic multi-turn trials.

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
  ray) start_script="serve/ray_8x/start.sh"; stop_script="serve/ray_8x/stop.sh"; policy="consistent_hash" ;;
  vllm) start_script="serve/vllm_8x/start.sh"; stop_script="serve/vllm_8x/stop.sh"; policy="consistent_hash" ;;
  *) echo "Usage: $0 --backend ray|vllm [--concurrency 64 ...]" >&2; exit 2 ;;
esac

export MODEL="${MODEL:-Qwen/Qwen3-0.6B-FP8}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
# fastokens on vLLM 0.22.0 is the native VLLM_USE_FASTOKENS env toggle (swaps the
# HF fast-tokenizer Rust BPE backend, ~25x faster); the tokenizer stays a normal
# HF fast tokenizer so streaming detok is unaffected.
export VLLM_USE_FASTOKENS="${VLLM_USE_FASTOKENS:-1}"
export URL="${URL:-http://127.0.0.1:8000}"
export BASE_DATASET="${BASE_DATASET:-datasets/agentic_loop_2200s_seed42/dataset.jsonl}"
export RESULTS_DIR="${RESULTS_DIR:-artifacts/reruns/agentic_8x/$backend/$(date -u +%Y%m%d-%H%M%S)}"
export CONCURRENCIES="${concurrency[*]}"
export SESSION_HEADER="${SESSION_HEADER:-x-correlation-id}"

if [[ "$no_start" != "1" ]]; then
  echo "Starting $backend 8x stack with policy=$policy"
  DRY_RUN="$dry_run" MODEL="$MODEL" "$start_script" "$policy"
fi

cleanup() {
  if [[ "$no_start" != "1" && "$dry_run" != "1" && "${KEEP_STACK_UP:-0}" != "1" ]]; then
    "$stop_script" || true
  fi
}
trap cleanup EXIT

DRY_RUN="$dry_run" ./bench/_run_agentic_profile_sweep.sh
