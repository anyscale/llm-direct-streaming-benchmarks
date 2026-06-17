#!/usr/bin/env bash
# Stop the vLLM P/D 4p4d stack.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/vllm_pd4p4d"

LOG_DIR="logs"

collect_descendants() {
  local parent="$1"
  local child
  pgrep -P "$parent" 2>/dev/null | while read -r child; do
    [[ -z "$child" ]] && continue
    echo "$child"
    collect_descendants "$child"
  done
}

stop_pid() {
  local pid="$1"
  local label="$2"
  [[ -z "$pid" ]] && return 0
  echo "Stopping $label pid=$pid"
  mapfile -t descendants < <(collect_descendants "$pid")
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$pgid" ]]; then kill -TERM -- "-$pgid" 2>/dev/null || true; fi
  kill -TERM "$pid" 2>/dev/null || true
  ((${#descendants[@]} > 0)) && kill -TERM "${descendants[@]}" 2>/dev/null || true
  sleep 2
  if [[ -n "$pgid" ]]; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi
  kill -KILL "$pid" 2>/dev/null || true
  ((${#descendants[@]} > 0)) && kill -KILL "${descendants[@]}" 2>/dev/null || true
}

if [[ -f "$LOG_DIR/proxy.pid" ]]; then
  stop_pid "$(cat "$LOG_DIR/proxy.pid")" proxy
  rm -f "$LOG_DIR/proxy.pid"
fi
if [[ -f "$LOG_DIR/worker.pids" ]]; then
  while read -r pid; do stop_pid "$pid" vllm-rank; done < "$LOG_DIR/worker.pids"
  rm -f "$LOG_DIR/worker.pids"
fi

if [[ "${VLLM_STOP_STRAY:-0}" == "1" ]]; then
  pkill -9 -f '^vllm-router([[:space:]]|$)' 2>/dev/null || true
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'EngineCore|ApiServer|VLLM::Worker' 2>/dev/null || true
fi

echo "vllm_pd4p4d stopped"

