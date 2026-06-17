#!/usr/bin/env bash
# Generate the deterministic agentic-code dataset used by the benchmark sweeps.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_SESSIONS="${NUM_SESSIONS:-2200}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-datasets/agentic_loop_${NUM_SESSIONS}s_seed${SEED}}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
if [[ "$AIPERF_BIN" == "aiperf" && -x ".venv/bin/aiperf" ]]; then
  AIPERF_BIN=".venv/bin/aiperf"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "$AIPERF_BIN" synthesize agentic-code \
    --config configs/agentic_loop.json \
    --num-sessions "$NUM_SESSIONS" \
    --seed "$SEED" \
    --output datasets
  echo
  echo "Would normalize latest generated directory to $OUT_DIR"
  exit 0
fi

if ! command -v "$AIPERF_BIN" >/dev/null 2>&1; then
  echo "ERROR: $AIPERF_BIN not found. Install aiperf==0.8.0 or set AIPERF_BIN." >&2
  exit 1
fi

mkdir -p datasets
before="$(mktemp)"
find datasets -mindepth 1 -maxdepth 1 -type d -print | sort > "$before"

"$AIPERF_BIN" synthesize agentic-code \
  --config configs/agentic_loop.json \
  --num-sessions "$NUM_SESSIONS" \
  --seed "$SEED" \
  --output datasets

latest="$(comm -13 "$before" <(find datasets -mindepth 1 -maxdepth 1 -type d -print | sort) | head -1)"
rm -f "$before"
if [[ -z "$latest" ]]; then
  latest="$(ls -1dt datasets/*/ | head -1)"
fi

rm -rf "$OUT_DIR"
mv "$latest" "$OUT_DIR"
python scripts/validate_dataset.py "$OUT_DIR/dataset.jsonl" --write-manifest "$OUT_DIR/repro_manifest.json"

echo "Dataset ready: $OUT_DIR/dataset.jsonl"
