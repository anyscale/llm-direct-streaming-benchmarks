#!/usr/bin/env python3
"""Compare rerun benchmark artifacts against the frozen baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONCURRENCIES = [64, 128, 256]
SYNTHETIC_WORKLOADS = ["isl8000_osl50", "isl50_osl500"]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def avg(raw: dict, key: str) -> float:
    value = raw[key]
    if isinstance(value, dict):
        return float(value["avg"])
    return float(value)


def pct_delta(candidate: float, baseline: float) -> float:
    return ((candidate - baseline) / baseline) * 100.0


def normalize_workload(value: str) -> str:
    if value.startswith("isl"):
        return value
    isl, osl = value.split(":", 1)
    return f"isl{isl}_osl{osl}"


def selected_concurrencies(values: list[int] | None) -> list[int]:
    return values or CONCURRENCIES


def synthetic_rows(
    baseline_root: Path,
    candidate_root: Path,
    system: str,
    workloads: list[str] | None = None,
    concurrencies: list[int] | None = None,
):
    selected_workloads = workloads or SYNTHETIC_WORKLOADS
    for workload in selected_workloads:
        for c in selected_concurrencies(concurrencies):
            base = load_json(baseline_root / "synthetic_8x" / system / workload / f"c{c}.json")
            cand = load_json(candidate_root / workload / f"c{c}.json")
            for metric, key in [
                ("ttft_ms", "mean_ttft_ms"),
                ("tpot_ms", "mean_tpot_ms"),
                ("req_s", "request_throughput"),
            ]:
                yield workload, c, metric, float(base[key]), float(cand[key])


def agentic_rows(
    baseline_root: Path,
    candidate_root: Path,
    system: str,
    kind: str,
    concurrencies: list[int] | None = None,
):
    if kind == "agentic_8x":
        metrics = [
            ("ttft_ms", "time_to_first_token"),
            ("tpot_ms", "inter_token_latency"),
            ("req_s", "request_throughput"),
        ]
    else:
        metrics = [
            ("ttft_ms", "time_to_first_token"),
            ("tpot_ms", "inter_token_latency"),
            ("out_tok_s", "output_token_throughput"),
        ]

    for c in selected_concurrencies(concurrencies):
        base = load_json(
            baseline_root / kind / system / f"c{c}" / "profile_export_aiperf.json"
        )
        cand = load_json(candidate_root / f"c{c}" / "profile_export_aiperf.json")
        for label, key in metrics:
            yield kind, c, label, avg(base, key), avg(cand, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["synthetic_8x", "agentic_8x", "agentic_pd4p4d"], required=True)
    parser.add_argument("--system", choices=["ray", "vllm"], required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts"))
    parser.add_argument("--workloads", nargs="+", help="Synthetic workloads, e.g. 8000:50")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        choices=CONCURRENCIES,
        help="Limit comparison to one or more concurrency levels.",
    )
    parser.add_argument("--warn-pct", type=float, default=15.0)
    parser.add_argument("--fail-pct", type=float, default=30.0)
    args = parser.parse_args()

    if args.kind == "synthetic_8x":
        workloads = (
            [normalize_workload(workload) for workload in args.workloads]
            if args.workloads
            else None
        )
        rows = list(
            synthetic_rows(
                args.baseline,
                args.candidate,
                args.system,
                workloads,
                args.concurrency,
            )
        )
    else:
        rows = list(
            agentic_rows(
                args.baseline,
                args.candidate,
                args.system,
                args.kind,
                args.concurrency,
            )
        )

    worst = 0.0
    warned = False
    print(f"{'case':<18} {'c':>4} {'metric':<10} {'baseline':>12} {'candidate':>12} {'delta':>9} status")
    print("-" * 82)
    for case, c, metric, baseline, candidate in rows:
        delta = pct_delta(candidate, baseline)
        worst = max(worst, abs(delta))
        status = "ok"
        if abs(delta) > args.fail_pct:
            status = "FAIL"
        elif abs(delta) > args.warn_pct:
            status = "warn"
            warned = True
        print(
            f"{case:<18} {c:>4} {metric:<10} "
            f"{baseline:>12.3f} {candidate:>12.3f} {delta:>8.1f}% {status}"
        )

    print("-" * 82)
    print(f"worst_abs_delta={worst:.1f}% warn_pct={args.warn_pct:.1f}% fail_pct={args.fail_pct:.1f}%")
    if any(abs(pct_delta(candidate, baseline)) > args.fail_pct for _, _, _, baseline, candidate in rows):
        raise SystemExit(1)
    if warned:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
