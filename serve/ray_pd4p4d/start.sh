#!/usr/bin/env bash
# Start Ray Serve LLM P/D 4p4d direct-streaming deployment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/ray_pd4p4d"

POLICY="${1:-consistent_hash}"
HAPROXY_PORT="${HAPROXY_PORT:-8000}"
MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"

export RAY_SERVE_ENABLE_HA_PROXY=1
export RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1
export RAY_SERVE_THROUGHPUT_OPTIMIZED="${RAY_SERVE_THROUGHPUT_OPTIMIZED:-1}"
export RAY_SERVE_HAPROXY_TCP_NODELAY="${RAY_SERVE_HAPROXY_TCP_NODELAY:-1}"
export RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY="${RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY:-0}"
export RAY_SERVE_SESSION_ID_HEADER_KEY="${RAY_SERVE_SESSION_ID_HEADER_KEY:-x-correlation-id}"
export ROUTING_POLICY="$POLICY"
export MODEL
export PREFILL_DP="${PREFILL_DP:-4}"
export DECODE_DP="${DECODE_DP:-4}"
export PREFILL_MAX_ONGOING="${PREFILL_MAX_ONGOING:-256}"
export DECODE_MAX_ONGOING="${DECODE_MAX_ONGOING:-10000}"
export PREFILL_KV_ROLE="${PREFILL_KV_ROLE:-kv_producer}"
export DECODE_KV_ROLE="${DECODE_KV_ROLE:-kv_consumer}"
# launch.py uses the public build_pd_openai_app builder; aiperf does its own
# warmup, so the Serve-level prewarm is off.
export PD_PREWARM="${PD_PREWARM:-0}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if ss -ltn "sport = :$HAPROXY_PORT" 2>/dev/null | grep -q LISTEN; then
  echo "ERROR: port $HAPROXY_PORT already in use. Run serve/ray_pd4p4d/stop.sh first." >&2
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
echo "Ray P/D launch pid=$(cat "$LOG_DIR/launch.pid")"

deadline=$(( $(date +%s) + 1800 ))
until curl -fsS --max-time 5 "http://127.0.0.1:$HAPROXY_PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; do
  if (( $(date +%s) > deadline )); then
    echo "ERROR: $MODEL did not register. Tail:" >&2
    tail -100 "$LOG_DIR/launch.log" >&2
    exit 1
  fi
  if ! kill -0 "$(cat "$LOG_DIR/launch.pid")" 2>/dev/null; then
    echo "ERROR: launch.py exited. Tail:" >&2
    tail -100 "$LOG_DIR/launch.log" >&2
    exit 1
  fi
  sleep 5
done

echo "Ray P/D ready: http://127.0.0.1:$HAPROXY_PORT"
