#!/usr/bin/env bash
# Start Ray Serve LLM 8x direct-streaming deployment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/ray_8x"

POLICY="${1:-round_robin}"
HAPROXY_PORT="${HAPROXY_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B-FP8}"

export RAY_SERVE_ENABLE_HA_PROXY=1
export RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1
export ROUTING_POLICY="$POLICY"
export MODEL

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if ss -ltn "sport = :$HAPROXY_PORT" 2>/dev/null | grep -q LISTEN; then
  echo "ERROR: port $HAPROXY_PORT already in use. Run serve/ray_8x/stop.sh first." >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1 python launch.py
  exit 0
fi

if ! ray status >/dev/null 2>&1; then
  echo "ERROR: ray is not running. Start it with 'ray start --head'." >&2
  exit 1
fi

nohup python launch.py > "$LOG_DIR/launch.log" 2>&1 &
echo "$!" > "$LOG_DIR/launch.pid"
echo "Ray Serve launch pid=$(cat "$LOG_DIR/launch.pid")"

deadline=$(( $(date +%s) + 1200 ))
until curl -fsS --max-time 5 "http://127.0.0.1:$HAPROXY_PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; do
  if (( $(date +%s) > deadline )); then
    echo "ERROR: $MODEL did not register. Tail:" >&2
    tail -80 "$LOG_DIR/launch.log" >&2
    exit 1
  fi
  if ! kill -0 "$(cat "$LOG_DIR/launch.pid")" 2>/dev/null; then
    echo "ERROR: launch.py exited. Tail:" >&2
    tail -80 "$LOG_DIR/launch.log" >&2
    exit 1
  fi
  sleep 5
done

echo "Ray Serve ready: http://127.0.0.1:$HAPROXY_PORT"
