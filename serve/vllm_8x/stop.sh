#!/usr/bin/env bash
# Stop the vLLM 8x stack started by start.sh.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/vllm_8x"

LOG_DIR="logs"

stop_pid() {
  local pid="$1"
  local label="$2"
  [[ -z "$pid" ]] && return 0
  echo "Stopping $label pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
}

if [[ -f "$LOG_DIR/router.pid" ]]; then
  stop_pid "$(cat "$LOG_DIR/router.pid")" router
  rm -f "$LOG_DIR/router.pid"
fi
if [[ -f "$LOG_DIR/worker.pids" ]]; then
  while read -r pid; do stop_pid "$pid" worker; done < "$LOG_DIR/worker.pids"
  rm -f "$LOG_DIR/worker.pids"
fi

if [[ "${VLLM_STOP_STRAY:-0}" == "1" ]]; then
  pkill -9 -f '^vllm-router([[:space:]]|$)' 2>/dev/null || true
  pkill -9 -f 'vllm serve' 2>/dev/null || true
fi

echo "vllm_8x stopped"

