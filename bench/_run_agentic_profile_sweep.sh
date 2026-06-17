#!/usr/bin/env bash
# Common AIPerf sweep for the agentic Mooncake trace.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODEL:?set MODEL}"
: "${TOKENIZER:?set TOKENIZER}"
: "${URL:?set URL}"
: "${BASE_DATASET:?set BASE_DATASET}"
: "${RESULTS_DIR:?set RESULTS_DIR}"

CONCURRENCIES="${CONCURRENCIES:-64 128 256}"
SESSIONS_PER_C="${SESSIONS_PER_C:-8}"
WARMUP_PER_C="${WARMUP_PER_C:-2}"
WORKERS_MAX="${WORKERS_MAX:-1024}"
SESSION_HEADER="${SESSION_HEADER:-x-correlation-id}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
if [[ "$AIPERF_BIN" == "aiperf" && -x ".venv/bin/aiperf" ]]; then
  AIPERF_BIN=".venv/bin/aiperf"
fi

if [[ ! -f "$BASE_DATASET" ]]; then
  echo "ERROR: dataset not found: $BASE_DATASET" >&2
  echo "Run ./scripts/generate_agentic_dataset.sh first." >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v "$AIPERF_BIN" >/dev/null 2>&1; then
  echo "ERROR: $AIPERF_BIN not found. Install aiperf==0.8.0 or set AIPERF_BIN." >&2
  exit 1
fi

supports_session_header=0
if "$AIPERF_BIN" profile --help 2>&1 | grep -q -- "--session-header"; then
  supports_session_header=1
elif [[ "${SESSION_HEADER,,}" != "x-correlation-id" ]]; then
  echo "WARNING: $AIPERF_BIN does not support --session-header; using built-in X-Correlation-ID header." >&2
fi

BASE_SESSIONS="$(python3 -c "import json; print(len({json.loads(l)['session_id'] for l in open('$BASE_DATASET') if l.strip()}))")"
SLICE_DIR="$(dirname "$BASE_DATASET")/_slices"
mkdir -p "$SLICE_DIR" "$RESULTS_DIR"

echo "Base dataset: $BASE_DATASET ($BASE_SESSIONS sessions)"

for c in $CONCURRENCIES; do
  needed=$(( SESSIONS_PER_C * c ))
  if (( needed > BASE_SESSIONS )); then
    echo "ERROR: c=$c needs $needed sessions, but base has $BASE_SESSIONS" >&2
    exit 1
  fi

  slice="$SLICE_DIR/c${c}_n${needed}.jsonl"
  ramp="$(python3 -c "print(max(5, min(60, int(0.25 * $c))))")"
  warmup=$(( WARMUP_PER_C * c ))
  artifact_dir="$RESULTS_DIR/c${c}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' python3 scripts/slice_dataset.py --src "$BASE_DATASET" --dst "$slice" --num-sessions "$needed"
    echo
  else
    python3 scripts/slice_dataset.py --src "$BASE_DATASET" --dst "$slice" --num-sessions "$needed"
  fi

  cmd=(
    "$AIPERF_BIN" profile
    --model "$MODEL"
    --tokenizer "$TOKENIZER"
    --url "$URL"
    --endpoint-type chat
    --streaming
    --input-file "$slice"
    --custom-dataset-type mooncake_trace
    --isl-block-size 64
    --extra-inputs ignore_eos:true
    --concurrency "$c"
    --concurrency-ramp-duration "$ramp"
    --request-rate "${REQUEST_RATE:-500}"
    --arrival-pattern "${ARRIVAL_PATTERN:-constant}"
    --warmup-request-count "$warmup"
    --workers-max "$WORKERS_MAX"
    --ui simple
    --artifact-dir "$artifact_dir"
    --server-metrics-formats json csv jsonl
    --slice-duration "${SLICE_DURATION:-60}"
  )
  if [[ "$supports_session_header" == "1" ]]; then
    cmd+=(--session-header "$SESSION_HEADER")
  fi

  if [[ -n "${SERVER_METRICS_URLS:-}" ]]; then
    read -r -a metrics_urls <<< "$SERVER_METRICS_URLS"
    cmd+=(--server-metrics "${metrics_urls[@]}")
  fi
  if [[ -n "${AIPERF_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$AIPERF_EXTRA_ARGS"
    cmd+=("${extra_args[@]}")
  fi

  echo
  echo "=== c=$c sessions=$needed ramp=${ramp}s warmup=$warmup ==="
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
done

echo "Sweep complete: $RESULTS_DIR"
