#!/usr/bin/env bash
# Stop Ray Serve P/D deployments started by serve/ray_pd4p4d/start.sh.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/serve/ray_pd4p4d"

LOG_DIR="logs"

serve shutdown -y 2>/dev/null || true

if [[ -f "$LOG_DIR/launch.pid" ]]; then
  pid="$(cat "$LOG_DIR/launch.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$LOG_DIR/launch.pid"
fi

pkill -f 'serve/ray_pd4p4d/launch.py' 2>/dev/null || true
pkill -f 'haproxy.*serve' 2>/dev/null || true
echo "ray_pd4p4d stopped"

