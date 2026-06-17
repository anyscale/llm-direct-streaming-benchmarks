#!/usr/bin/env python3
"""Slice a Mooncake-trace dataset.jsonl to the first N sessions.

Sessions are atomic in aiperf's multi-turn replay, so we slice by session_id —
keep all rows whose session_id is among the first N unique session_ids
encountered. Preserves ordering, leaves turn structure intact.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def slice_dataset(src: Path, dst: Path, num_sessions: int) -> int:
    kept_ids: dict[str, None] = {}  # ordered set
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["session_id"]
            if sid not in kept_ids:
                if len(kept_ids) >= num_sessions:
                    continue
                kept_ids[sid] = None
            fout.write(line)
    return len(kept_ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--num-sessions", type=int, required=True)
    args = p.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n = slice_dataset(args.src, args.dst, args.num_sessions)
    print(f"Wrote {n} sessions to {args.dst}")


if __name__ == "__main__":
    main()
