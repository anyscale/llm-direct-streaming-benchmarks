#!/usr/bin/env python3
"""Validate that all frozen artifact points needed by the figures are present."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


CONCURRENCIES = [64, 128, 256]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def csv_metric(path: Path, metric: str) -> float:
    for row in csv.reader(io.StringIO(path.read_text())):
        if row and row[0].split("(")[0].strip() == metric:
            return float(row[1])
    raise KeyError(f"{metric!r} not found in {path}")


def check_synthetic(root: Path) -> int:
    count = 0
    for system in ("ray", "vllm"):
        for workload in ("isl8000_osl50", "isl50_osl500"):
            for c in CONCURRENCIES:
                raw = load_json(root / "synthetic_8x" / system / workload / f"c{c}.json")
                for key in ("mean_ttft_ms", "mean_tpot_ms", "request_throughput"):
                    if key not in raw:
                        raise KeyError(f"{key} missing for {system}/{workload}/c{c}")
                count += 1
    return count


def check_agentic_8x(root: Path) -> int:
    count = 0
    for system in ("ray", "vllm"):
        for c in CONCURRENCIES:
            path = root / "agentic_8x" / system / f"c{c}" / "profile_export_aiperf.csv"
            csv_metric(path, "Time to First Token")
            csv_metric(path, "Inter Token Latency")
            csv_metric(path, "Request Throughput")
            count += 1
    return count


def check_pd4p4d(root: Path) -> int:
    count = 0
    for system in ("ray", "vllm"):
        for c in CONCURRENCIES:
            raw = load_json(
                root
                / "agentic_pd4p4d"
                / system
                / f"c{c}"
                / "profile_export_aiperf.json"
            )
            for key in ("time_to_first_token", "inter_token_latency", "output_token_throughput"):
                if "avg" not in raw[key]:
                    raise KeyError(f"{key}.avg missing for {system}/pd4p4d/c{c}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    synthetic = check_synthetic(args.artifacts)
    agentic = check_agentic_8x(args.artifacts)
    pd = check_pd4p4d(args.artifacts)
    print(f"validated synthetic_8x points: {synthetic}")
    print(f"validated agentic_8x points: {agentic}")
    print(f"validated agentic_pd4p4d points: {pd}")


if __name__ == "__main__":
    main()

