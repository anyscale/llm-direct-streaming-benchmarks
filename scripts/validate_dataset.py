#!/usr/bin/env python3
"""Validate a Mooncake trace JSONL dataset and optionally write a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = 0
    sessions: dict[str, int] = {}
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            sha.update(raw)
            row = json.loads(raw)
            sid = row["session_id"]
            sessions[sid] = sessions.get(sid, 0) + 1
            rows += 1
    return {
        "path": str(path),
        "rows": rows,
        "sessions": len(sessions),
        "sha256": sha.hexdigest(),
        "min_turns_per_session": min(sessions.values()) if sessions else 0,
        "max_turns_per_session": max(sessions.values()) if sessions else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    info = summarize(args.dataset)
    print(json.dumps(info, indent=2))
    if args.write_manifest:
        payload = {
            **info,
            "generator": "aiperf synthesize agentic-code",
            "aiperf_version": "0.8.0",
            "num_sessions_requested": 2200,
            "seed": 42,
            "config": "configs/agentic_loop.json",
            "model": "microsoft/Phi-tiny-MoE-instruct for P/D; Qwen/Qwen3-0.6B-FP8 for 8x",
            "tokenizer": "same as model unless overridden",
        }
        args.write_manifest.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

