#!/usr/bin/env python3
"""Plot the two Ray Serve LLM vs vLLM-router comparison figures."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.ticker import FuncFormatter


CONCURRENCIES = [64, 128, 256]
RAY_COLOR = "#1f77b4"
VLLM_COLOR = "#ff7f0e"

SYSTEMS = [
    {"key": "ray", "label": "Ray Serve LLM", "color": RAY_COLOR, "marker": "o"},
    {"key": "vllm", "label": "vLLM-router", "color": VLLM_COLOR, "marker": "s"},
]

WORKLOAD_ROWS = [
    (
        "isl8000_osl50",
        "Prefill-heavy\n(ISL=8000, OSL=50)\nround-robin routing",
    ),
    (
        "isl50_osl500",
        "Decode-heavy\n(ISL=50, OSL=500)\nround-robin routing",
    ),
    (
        "agentic",
        "Coding-agent multi-turn\n(real prompt traces)\nKV-aware routing",
    ),
]

METRIC_COLUMNS = [
    ("ttft", "Mean TTFT (ms)", "lower is better"),
    ("tpot", "Mean TPOT (ms)", "lower is better"),
    ("throughput", "Throughput (req/s)", "higher is better"),
]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# aiperf CSV column indices: Metric,avg,min,max,sum,p1,p5,p10,p25,p50,...
_AIPERF_AVG = 1


def read_aiperf_csv_metric(path: Path, metric_name: str, col: int = _AIPERF_AVG) -> float:
    text = path.read_text()
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        name = row[0].split("(")[0].strip()
        if name == metric_name:
            return float(row[col])
    raise KeyError(f"{metric_name!r} not found in {path}")


def load_synthetic_point(artifacts: Path, system_key: str, workload: str, c: int,
                         override_dir: Path | None = None) -> dict:
    # override_dir points at a rerun results dir laid out as <dir>/<workload>/c<N>.json
    path = (override_dir / workload / f"c{c}.json" if override_dir is not None
            else artifacts / "synthetic_8x" / system_key / workload / f"c{c}.json")
    raw = load_json(path)
    return {
        "ttft": raw["mean_ttft_ms"],
        "tpot": raw["mean_tpot_ms"],
        "throughput": raw["request_throughput"],
    }


def load_agentic_8x_point(artifacts: Path, system_key: str, c: int,
                          override_dir: Path | None = None) -> dict:
    # override_dir points at a rerun results dir laid out as <dir>/c<N>/profile_export_aiperf.csv
    path = (override_dir / f"c{c}" / "profile_export_aiperf.csv" if override_dir is not None
            else artifacts / "agentic_8x" / system_key / f"c{c}" / "profile_export_aiperf.csv")
    return {
        "ttft": read_aiperf_csv_metric(path, "Time to First Token"),
        "tpot": read_aiperf_csv_metric(path, "Inter Token Latency"),
        "throughput": read_aiperf_csv_metric(path, "Request Throughput"),
    }


def load_workload_series(
    artifacts: Path, row_key: str, system_key: str,
    synth_dirs: dict[str, Path] | None = None, ag_dirs: dict[str, Path] | None = None,
) -> dict[int, dict[str, float]]:
    synth_dirs = synth_dirs or {}
    ag_dirs = ag_dirs or {}
    out = {}
    for c in CONCURRENCIES:
        if row_key == "agentic":
            out[c] = load_agentic_8x_point(artifacts, system_key, c, ag_dirs.get(system_key))
        else:
            out[c] = load_synthetic_point(artifacts, system_key, row_key, c, synth_dirs.get(system_key))
    return out


def load_pd4p4d_point(artifacts: Path, system_key: str, c: int) -> dict:
    raw = load_json(
        artifacts
        / "agentic_pd4p4d"
        / system_key
        / f"c{c}"
        / "profile_export_aiperf.json"
    )
    return {
        "ttft": raw["time_to_first_token"]["avg"],
        "tpot": raw["inter_token_latency"]["avg"],
        "output_toks": raw["output_token_throughput"]["avg"],
    }


def fmt_value(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 20:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def deoverlap(ax, texts: list) -> None:
    """De-overlap value labels on an axis (adjustText), nudging mostly
    vertically with a thin leader line when a label is pushed off its point."""
    if texts:
        adjust_text(
            texts, ax=ax, only_move={"text": "y", "static": "y", "explode": "y"},
            arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.6},
        )


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors="#444", which="both", length=4)
    ax.yaxis.grid(True, color="#e6e6e6", linestyle="-", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:g}")
    )


def plot_series(ax, xs: list[int], ys: list[float], system: dict, label: bool) -> list:
    ax.plot(
        xs,
        ys,
        color=system["color"],
        marker=system["marker"],
        markersize=8,
        markerfacecolor=system["color"],
        markeredgecolor="white",
        markeredgewidth=1.4,
        linewidth=2.4,
        label=system["label"] if label else None,
        zorder=3,
    )
    # Return value-label text objects; deoverlap() de-overlaps them per-axis.
    return [
        ax.text(x, y, fmt_value(y), fontsize=8.5, color=system["color"],
                fontweight="bold", ha="center", va="center", zorder=5,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.9})
        for x, y in zip(xs, ys)
    ]


def build_3x3_figure(artifacts: Path, synth_dirs: dict[str, Path] | None = None,
                     ag_dirs: dict[str, Path] | None = None) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 12),
        gridspec_kw={"hspace": 0.42, "wspace": 0.22},
    )

    for row_idx, (row_key, row_title) in enumerate(WORKLOAD_ROWS):
        per_system = {
            system["key"]: load_workload_series(
                artifacts, row_key, system["key"], synth_dirs, ag_dirs
            )
            for system in SYSTEMS
        }
        for col_idx, (metric_key, metric_title, direction) in enumerate(METRIC_COLUMNS):
            ax = axes[row_idx, col_idx]
            texts = []
            for system in SYSTEMS:
                data = per_system[system["key"]]
                xs = CONCURRENCIES
                ys = [float(data[c][metric_key]) for c in xs]
                texts += plot_series(ax, xs, ys, system, label=(row_idx == 0 and col_idx == 0))

            ax.set_xticks(CONCURRENCIES)
            ax.set_xticklabels([str(c) for c in CONCURRENCIES])
            ax.set_xlim(CONCURRENCIES[0] - 12, CONCURRENCIES[-1] + 12)
            _, ymax = ax.get_ylim()
            ax.set_ylim(bottom=0, top=ymax * 1.22 if ymax > 0 else 1)
            style_axis(ax)
            deoverlap(ax, texts)

            if row_idx == 0:
                ax.set_title(f"{metric_title}\n({direction})", fontsize=11.5, pad=10)
            if row_idx == 2:
                ax.set_xlabel("Max concurrency", fontsize=10.5, color="#333")
            if col_idx == 0:
                ax.set_ylabel(
                    row_title,
                    fontsize=11,
                    fontweight="bold",
                    color="#1a1a1a",
                    labelpad=10,
                )

    fig.suptitle(
        "Performance Comparison Across Workloads",
        fontsize=17,
        fontweight="bold",
        y=0.995,
        color="#0d1117",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(SYSTEMS),
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.07,
        top=0.90,
        hspace=0.55,
        wspace=0.22,
    )
    return fig


def build_pd4p4d_figure(artifacts: Path) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6.1),
        gridspec_kw={"wspace": 0.22},
    )
    columns = [
        ("ttft", "Mean TTFT (ms)", "lower is better"),
        ("tpot", "Mean TPOT (ms)", "lower is better"),
        ("output_toks", "Output tok/s", "higher is better"),
    ]
    per_system = {
        system["key"]: {
            c: load_pd4p4d_point(artifacts, system["key"], c) for c in CONCURRENCIES
        }
        for system in SYSTEMS
    }

    for col_idx, (metric_key, metric_title, direction) in enumerate(columns):
        ax = axes[col_idx]
        texts = []
        for system in SYSTEMS:
            data = per_system[system["key"]]
            xs = CONCURRENCIES
            ys = [data[c][metric_key] for c in xs]
            texts += plot_series(ax, xs, ys, system, label=(col_idx == 0))

        ax.set_title(f"{metric_title}\n({direction})", fontsize=11.5, pad=10)
        ax.set_xticks(CONCURRENCIES)
        ax.set_xticklabels([str(c) for c in CONCURRENCIES])
        ax.set_xlim(CONCURRENCIES[0] - 12, CONCURRENCIES[-1] + 12)
        ax.set_xlabel("Max concurrency", fontsize=10.5, color="#333")
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax * 1.22 if ymax > 0 else 1)
        style_axis(ax)
        deoverlap(ax, texts)

    fig.suptitle(
        "Agentic 4p4d P/D Wide-EP Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.985,
        color="#0d1117",
    )
    fig.text(
        0.5,
        0.925,
        "Ray Serve LLM vs vLLM-router.",
        ha="center",
        va="top",
        fontsize=10,
        color="#444",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=len(SYSTEMS),
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.12,
        top=0.72,
        wspace=0.22,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--ray-synthetic-dir", type=Path, default=None,
                        help="rerun dir for the Ray synthetic rows (<dir>/isl<I>_osl<O>/c<N>.json)")
    parser.add_argument("--ray-agentic-dir", type=Path, default=None,
                        help="rerun dir for the Ray agentic row (<dir>/c<N>/profile_export_aiperf.csv)")
    parser.add_argument("--vllm-synthetic-dir", type=Path, default=None,
                        help="rerun dir for the vLLM synthetic rows (same layout as --ray-synthetic-dir)")
    parser.add_argument("--vllm-agentic-dir", type=Path, default=None,
                        help="rerun dir for the vLLM agentic row (same layout as --ray-agentic-dir)")
    parser.add_argument("--out-3x3-name", default="fig_3x3_overview_ray_vllm.png")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    synth_dirs = {k: v for k, v in
                  {"ray": args.ray_synthetic_dir, "vllm": args.vllm_synthetic_dir}.items()
                  if v is not None}
    ag_dirs = {k: v for k, v in
               {"ray": args.ray_agentic_dir, "vllm": args.vllm_agentic_dir}.items()
               if v is not None}

    out_3x3 = args.out / args.out_3x3_name
    fig = build_3x3_figure(args.artifacts, synth_dirs, ag_dirs)
    fig.savefig(out_3x3, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_3x3}")

    # The companion 4p4d figure uses its own frozen data; only regenerate it in
    # the default (no-override) mode so an override run touches just the 3x3.
    if not synth_dirs and not ag_dirs:
        out_pd = args.out / "fig_4p4d_agentic_ray_vllm.png"
        fig = build_pd4p4d_figure(args.artifacts)
        fig.savefig(out_pd, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out_pd}")


if __name__ == "__main__":
    main()
