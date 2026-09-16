from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
MANIFEST = ROOT / "run_manifest.csv"

BUDGET = 24
FAILED_REPEAT = 5

DESIGN_METRICS = [
    ("final_tau_min", "Residence Time", "min"),
    ("final_flow_rate_mL_min", "Flow Rate", "mL/min"),
    ("final_tubing_ID_mm", "Tubing ID", "mm"),
    ("final_BPR_bar", "BPR", "bar"),
    ("final_reactor_volume_mL", "Reactor Volume", "mL"),
]

EXECUTION_METRICS = [
    ("runtime_s", "Runtime", "s"),
    ("llm_call_count", "LLM Calls", "count"),
    ("total_tokens", "Total Tokens", "tokens"),
]


def _read_rows() -> list[dict]:
    rows = list(csv.DictReader(MANIFEST.open()))
    filtered: list[dict] = []
    for row in rows:
        if int(row["candidate_budget"]) != BUDGET:
            continue
        filtered.append(row)
    return filtered


def _completed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["status"] == "completed"]


def _failed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["status"] != "completed"]


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(rows: list[dict], metrics: list[tuple[str, str, str]]) -> list[dict]:
    out: list[dict] = []
    for key, label, unit in metrics:
        values = [_float(row, key) for row in rows]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        cv = (std / mean * 100.0) if mean else math.nan
        out.append(
            {
                "metric_key": key,
                "metric_label": label,
                "unit": unit,
                "n": len(values),
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "cv_percent": round(cv, 6) if not math.isnan(cv) else "",
            }
        )
    return out


def _metric_long(rows: list[dict], metrics: list[tuple[str, str, str]]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        repeat_index = int(row["repeat_index"])
        for key, label, unit in metrics:
            out.append(
                {
                    "repeat_index": repeat_index,
                    "metric_key": key,
                    "metric_label": label,
                    "unit": unit,
                    "value": _float(row, key),
                }
            )
    return out


def _scientific_off(ax) -> None:
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)


def _plot_small_multiples(
    rows: list[dict],
    metrics: list[tuple[str, str, str]],
    title: str,
    subtitle: str,
    png_name: str,
    svg_name: str,
    csv_name: str,
) -> None:
    summary_rows = _metric_summary(rows, metrics)
    _write_csv(
        VIS_DIR / csv_name,
        ["metric_key", "metric_label", "unit", "n", "mean", "std", "min", "max", "cv_percent"],
        summary_rows,
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.6), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, label, unit), summary in zip(axes, metrics, summary_rows):
        xs = list(range(1, len(rows) + 1))
        ys = [_float(row, metric) for row in rows]
        mean = float(summary["mean"])
        std = float(summary["std"])

        ax.scatter(xs, ys, s=36, color="#1f4e79", zorder=3, label="Repeat")
        ax.axhline(mean, color="#d95f02", linewidth=2.0, zorder=2, label="Mean")
        ax.fill_between(
            [0.6, len(rows) + 0.4],
            [mean - std, mean - std],
            [mean + std, mean + std],
            color="#fdb863",
            alpha=0.28,
            zorder=1,
            label="Mean ± SD",
        )
        ax.set_xlim(0.6, len(rows) + 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs])
        ax.set_title(f"{label}\nmean={mean:.3g}, sd={std:.3g}", fontsize=10, pad=10)
        ax.set_xlabel("Repeat")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        _scientific_off(ax)

    fig.suptitle(title, fontsize=15, y=1.04)
    fig.text(0.5, 0.99, subtitle, ha="center", va="top", fontsize=10)
    fig.savefig(VIS_DIR / png_name, dpi=220, bbox_inches="tight")
    fig.savefig(VIS_DIR / svg_name, bbox_inches="tight")
    plt.close(fig)


def _plot_cv(summary_rows: list[dict], png_name: str, svg_name: str, csv_name: str) -> None:
    _write_csv(
        VIS_DIR / csv_name,
        ["metric_key", "metric_label", "unit", "n", "mean", "std", "min", "max", "cv_percent"],
        summary_rows,
    )

    labels = [row["metric_label"] for row in summary_rows]
    values = [float(row["cv_percent"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color="#4daf4a", edgecolor="#2c7a2c", linewidth=1.0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Budget 24 Variability Across Design and Execution Metrics")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    plt.xticks(rotation=25, ha="right")
    fig.savefig(VIS_DIR / png_name, dpi=220, bbox_inches="tight")
    fig.savefig(VIS_DIR / svg_name, bbox_inches="tight")
    plt.close(fig)


def _plot_overview_table(rows: list[dict], png_name: str, svg_name: str, csv_name: str) -> None:
    fieldnames = [
        "repeat_index",
        "status",
        "runtime_s",
        "llm_call_count",
        "total_tokens",
        "final_tau_min",
        "final_flow_rate_mL_min",
        "final_tubing_ID_mm",
        "final_BPR_bar",
        "final_reactor_volume_mL",
    ]
    _write_csv(VIS_DIR / csv_name, fieldnames, rows)

    fig, ax = plt.subplots(figsize=(12.5, 2.8 + 0.35 * len(rows)), constrained_layout=True)
    ax.axis("off")
    table_data = [fieldnames] + [[row.get(col, "") for col in fieldnames] for row in rows]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for col in range(len(fieldnames)):
        table[(0, col)].set_facecolor("#d9e8f5")
        table[(0, col)].set_text_props(weight="bold")
    for i, row in enumerate(rows, start=1):
        if row["status"] != "completed":
            for col in range(len(fieldnames)):
                table[(i, col)].set_facecolor("#f9d5d3")
    ax.set_title("Budget 24 Repeat-Level Overview", fontsize=13, pad=12)
    fig.savefig(VIS_DIR / png_name, dpi=220, bbox_inches="tight")
    fig.savefig(VIS_DIR / svg_name, bbox_inches="tight")
    plt.close(fig)


def _write_manifest(all_rows: list[dict], completed_rows: list[dict], failed_rows: list[dict]) -> None:
    payload = {
        "source_manifest": str(MANIFEST),
        "candidate_budget": BUDGET,
        "total_rows_found": len(all_rows),
        "completed_rows_used_for_summary_figures": len(completed_rows),
        "failed_rows_excluded_from_summary_figures": len(failed_rows),
        "failed_repeat_indices": [int(row["repeat_index"]) for row in failed_rows],
        "notes": [
            "Summary figures use only completed repeats.",
            "Repeat 5 is excluded from mean/std plots because the benchmark run failed with final metric fields blank in run_manifest.csv.",
        ],
    }
    (VIS_DIR / "budget24_visualization_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = _read_rows()
    completed_rows = _completed_rows(all_rows)
    failed_rows = _failed_rows(all_rows)

    raw_rows = [
        {
            "repeat_index": row["repeat_index"],
            "status": row["status"],
            "runtime_s": row["runtime_s"],
            "llm_call_count": row["llm_call_count"],
            "total_tokens": row["total_tokens"],
            "final_tau_min": row["final_tau_min"],
            "final_flow_rate_mL_min": row["final_flow_rate_mL_min"],
            "final_tubing_ID_mm": row["final_tubing_ID_mm"],
            "final_BPR_bar": row["final_BPR_bar"],
            "final_reactor_volume_mL": row["final_reactor_volume_mL"],
        }
        for row in all_rows
    ]

    completed_export = [
        {
            "repeat_index": row["repeat_index"],
            "status": row["status"],
            "runtime_s": row["runtime_s"],
            "llm_call_count": row["llm_call_count"],
            "total_tokens": row["total_tokens"],
            "final_tau_min": row["final_tau_min"],
            "final_flow_rate_mL_min": row["final_flow_rate_mL_min"],
            "final_tubing_ID_mm": row["final_tubing_ID_mm"],
            "final_BPR_bar": row["final_BPR_bar"],
            "final_reactor_volume_mL": row["final_reactor_volume_mL"],
        }
        for row in completed_rows
    ]

    _write_csv(
        VIS_DIR / "budget24_all_repeats_raw.csv",
        list(raw_rows[0].keys()),
        raw_rows,
    )
    _write_csv(
        VIS_DIR / "budget24_completed_repeats_raw.csv",
        list(completed_export[0].keys()),
        completed_export,
    )

    _write_csv(
        VIS_DIR / "budget24_design_metrics_long.csv",
        ["repeat_index", "metric_key", "metric_label", "unit", "value"],
        _metric_long(completed_rows, DESIGN_METRICS),
    )
    _write_csv(
        VIS_DIR / "budget24_execution_metrics_long.csv",
        ["repeat_index", "metric_key", "metric_label", "unit", "value"],
        _metric_long(completed_rows, EXECUTION_METRICS),
    )

    _plot_small_multiples(
        completed_rows,
        DESIGN_METRICS,
        title="Budget 24 Design Variables Across Completed Repeats",
        subtitle="Points = repeats, line = mean, band = ±1 SD | n=4 completed repeats",
        png_name="budget24_design_mean_std.png",
        svg_name="budget24_design_mean_std.svg",
        csv_name="budget24_design_mean_std.csv",
    )
    _plot_small_multiples(
        completed_rows,
        EXECUTION_METRICS,
        title="Budget 24 Execution Metrics Across Completed Repeats",
        subtitle="Points = repeats, line = mean, band = ±1 SD | n=4 completed repeats",
        png_name="budget24_execution_mean_std.png",
        svg_name="budget24_execution_mean_std.svg",
        csv_name="budget24_execution_mean_std.csv",
    )

    combined_summary = _metric_summary(completed_rows, DESIGN_METRICS + EXECUTION_METRICS)
    _plot_cv(
        combined_summary,
        png_name="budget24_metric_cv_percent.png",
        svg_name="budget24_metric_cv_percent.svg",
        csv_name="budget24_metric_cv_percent.csv",
    )
    _plot_overview_table(
        raw_rows,
        png_name="budget24_repeat_overview.png",
        svg_name="budget24_repeat_overview.svg",
        csv_name="budget24_repeat_overview.csv",
    )
    _write_manifest(all_rows, completed_rows, failed_rows)


if __name__ == "__main__":
    main()
