#!/usr/bin/env bash
# Run the synthetic random-prompt 8x replica trials.
#
# Methodology: maximum throughput (REQUEST_RATE=inf) with a discarded
# warm-up pass. The first run after a server starts pays a one-time cold start
# (CUDA graph capture + cache warmup), so we run the full sweep once to warm the
# engines (discarded), then run the identical sweep again and keep that second
# reading. Set WARMUP=0 (or pass --no-warmup) to skip the warm-up pass.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

backend=""
workloads=(8000:50 50:500)
concurrency=(64 128 256)
dry_run=0
no_start=0
use_pregenerated=1
synthetic_dataset_dir="datasets/synthetic_8x"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --workloads)
      workloads=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do workloads+=("$1"); shift; done
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
    --no-warmup)
      WARMUP=0
      shift
      ;;
    --synthetic-dataset-dir)
      synthetic_dataset_dir="$2"
      shift 2
      ;;
    --no-pregenerated)
      use_pregenerated=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$backend" in
  ray) start_script="serve/ray_8x/start.sh"; stop_script="serve/ray_8x/stop.sh"; policy="round_robin" ;;
  vllm) start_script="serve/vllm_8x/start.sh"; stop_script="serve/vllm_8x/stop.sh"; policy="round_robin" ;;
  *) echo "Usage: $0 --backend ray|vllm [--workloads 8000:50 ...] [--concurrency 64 ...] [--no-warmup]" >&2; exit 2 ;;
esac

MODEL="${MODEL:-Qwen/Qwen3-0.6B-FP8}"
URL="${URL:-http://127.0.0.1:8000}"
NUM_PROMPTS="${NUM_PROMPTS:-4096}"
RESULTS_DIR="${RESULTS_DIR:-artifacts/reruns/synthetic_8x/$backend/$(date -u +%Y%m%d-%H%M%S)}"
WARMUP="${WARMUP:-1}"
SYN_MAX_MODEL_LEN="${SYN_MAX_MODEL_LEN:-10000}"
SYN_GPU_MEM_UTIL="${SYN_GPU_MEM_UTIL:-0.95}"
SYN_MAX_NUM_SEQS="${SYN_MAX_NUM_SEQS:-128}"
SYN_MAX_NUM_BATCHED_TOKENS="${SYN_MAX_NUM_BATCHED_TOKENS:-16384}"
SYN_KV_CACHE_DTYPE="${SYN_KV_CACHE_DTYPE:-fp8}"
SYN_LOAD_FORMAT="${SYN_LOAD_FORMAT:-dummy}"
SYN_ENABLE_PREFIX_CACHING="${SYN_ENABLE_PREFIX_CACHING:-0}"

run_cmd() {
  if [[ "$dry_run" == "1" ]]; then
    printf '%q ' "$@"
    echo
  else
    "$@"
  fi
}

run_sweep() {  # $1 = output base dir for this pass
  local out_base="$1"
  local workload isl osl out_dir dataset_path c result_file
  for workload in "${workloads[@]}"; do
    IFS=: read -r isl osl <<< "$workload"
    out_dir="$out_base/isl${isl}_osl${osl}"
    mkdir -p "$out_dir"
    dataset_path="$synthetic_dataset_dir/isl${isl}_osl${osl}_n${NUM_PROMPTS}_seed0.jsonl"
    for c in "${concurrency[@]}"; do
      result_file="$out_dir/c${c}.json"
      local cmd=(
        vllm bench serve
        --backend openai
        --base-url "$URL"
        --model "$MODEL"
        --num-prompts "$NUM_PROMPTS"
        --max-concurrency "$c"
        # max throughput by default; burstiness=inf => constant (deterministic
        # 1/rate) arrivals, matching aiperf's --arrival-pattern constant.
        --request-rate "${REQUEST_RATE:-inf}"
        --burstiness "${BURSTINESS:-inf}"
        --percentile-metrics ttft,tpot,itl,e2el
        --save-result
        --result-filename "$result_file"
      )
      if [[ "$use_pregenerated" == "1" && -f "$dataset_path" ]]; then
        cmd+=(
          --dataset-name custom
          --dataset-path "$dataset_path"
          --custom-output-len "$osl"
          --skip-chat-template
          --disable-shuffle
          --ignore-eos
        )
      else
        cmd+=(
          --dataset-name random
          --random-input-len "$isl"
          --random-output-len "$osl"
        )
      fi
      # SAVE_DETAILED=1 keeps per-request arrays in the result JSON; off by default.
      [[ "${SAVE_DETAILED:-0}" == "1" ]] && cmd+=(--save-detailed)
      echo "Synthetic workload=$workload c=$c -> $result_file"
      run_cmd "${cmd[@]}"
    done
  done
}

if [[ "$no_start" != "1" ]]; then
  echo "Starting $backend 8x stack with policy=$policy"
  DRY_RUN="$dry_run" \
    MODEL="$MODEL" \
    MAX_MODEL_LEN="$SYN_MAX_MODEL_LEN" \
    GPU_MEM_UTIL="$SYN_GPU_MEM_UTIL" \
    MAX_NUM_SEQS="$SYN_MAX_NUM_SEQS" \
    MAX_NUM_BATCHED_TOKENS="$SYN_MAX_NUM_BATCHED_TOKENS" \
    KV_CACHE_DTYPE="$SYN_KV_CACHE_DTYPE" \
    LOAD_FORMAT="$SYN_LOAD_FORMAT" \
    ENABLE_PREFIX_CACHING="$SYN_ENABLE_PREFIX_CACHING" \
    "$start_script" "$policy"
fi

cleanup() {
  if [[ "$no_start" != "1" && "$dry_run" != "1" && "${KEEP_STACK_UP:-0}" != "1" ]]; then
    "$stop_script" || true
  fi
}
trap cleanup EXIT

# Discard-first / take-second reading: warm the engines with one full sweep
# (cold start = CUDA graph capture + cache warmup), then record the second.
if [[ "$WARMUP" == "1" && "$dry_run" != "1" ]]; then
  echo "### warm-up pass (discarded) ###"
  run_sweep "${WARMUP_SCRATCH:-/tmp/synthetic_8x_warmup/$backend}"
fi
echo "### recorded reading ###"
run_sweep "$RESULTS_DIR"

echo "Results: $RESULTS_DIR"
