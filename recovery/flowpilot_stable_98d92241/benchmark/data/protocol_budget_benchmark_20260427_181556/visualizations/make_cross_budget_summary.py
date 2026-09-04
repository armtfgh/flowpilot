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

BUDGETS = [1, 6, 12, 24]

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

ALL_METRICS = DESIGN_METRICS + EXECUTION_METRICS


def _read_rows() -> list[dict]:
    return list(csv.DictReader(MANIFEST.open()))


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


def _scientific_off(ax) -> None:
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)


def _cross_budget_summary(rows: list[dict], metrics: list[tuple[str, str, str]]) -> list[dict]:
    out: list[dict] = []
    for budget in BUDGETS:
        budget_rows = [row for row in rows if int(row["candidate_budget"]) == budget]
        for key, label, unit in metrics:
            values = [_float(row, key) for row in budget_rows]
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            cv = (std / mean * 100.0) if mean else math.nan
            out.append(
                {
                    "candidate_budget": budget,
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


def _metric_panel_plot(
    summary_rows: list[dict],
    metrics: list[tuple[str, str, str]],
    title: str,
    subtitle: str,
    png_name: str,
    svg_name: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.8), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    budgets = BUDGETS
    for ax, (metric_key, label, unit) in zip(axes, metrics):
        rows = [row for row in summary_rows if row["metric_key"] == metric_key]
        rows.sort(key=lambda row: int(row["candidate_budget"]))
        means = [float(row["mean"]) for row in rows]
        stds = [float(row["std"]) for row in rows]
        ns = [int(row["n"]) for row in rows]

        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            fmt="o-",
            linewidth=2.0,
            markersize=6.5,
            color="#1f4e79",
            ecolor="#d95f02",
            elinewidth=1.6,
            capsize=4,
            capthick=1.3,
        )
        ax.set_xticks(budgets)
        ax.set_xlabel("Candidate Budget")
        ax.set_ylabel(unit)
        ax.set_title(label, fontsize=10, pad=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        _scientific_off(ax)

        for x, y, sd, n in zip(budgets, means, stds, ns):
            ax.text(x, y + (0.02 * max(means) if max(means) else 0.02), f"n={n}", ha="center", va="bottom", fontsize=8)
            ax.text(x, y - (0.06 * max(means) if max(means) else 0.04), f"sd={sd:.2g}", ha="center", va="top", fontsize=7, color="#6a3d9a")

    fig.suptitle(title, fontsize=15, y=1.04)
    fig.text(0.5, 0.99, subtitle, ha="center", va="top", fontsize=10)
    fig.savefig(VIS_DIR / png_name, dpi=220, bbox_inches="tight")
    fig.savefig(VIS_DIR / svg_name, bbox_inches="tight")
    plt.close(fig)


def _plot_cv_heatmap(summary_rows: list[dict], png_name: str, svg_name: str) -> None:
    labels = [row["metric_label"] for row in summary_rows if int(row["candidate_budget"]) == 1]
    matrix = []
    for budget in BUDGETS:
        row = []
        for metric_label in labels:
            rec = next(
                item for item in summary_rows
                if int(item["candidate_budget"]) == budget and item["metric_label"] == metric_label
            )
            row.append(float(rec["cv_percent"]) if rec["cv_percent"] != "" else 0.0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10.6, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(len(BUDGETS)))
    ax.set_yticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Candidate Budget")
    ax.set_title("Relative Variability (CV%) Across Budgets and Metrics")
    for i, budget in enumerate(BUDGETS):
        for j, label in enumerate(labels):
            ax.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="CV (%)")
    fig.savefig(VIS_DIR / png_name, dpi=220, bbox_inches="tight")
    fig.savefig(VIS_DIR / svg_name, bbox_inches="tight")
    plt.close(fig)


def _plot_completion_overview(all_rows: list[dict], png_name: str, svg_name: str, csv_name: str) -> None:
    table_rows = []
    for budget in BUDGETS:
        budget_rows = [row for row in all_rows if int(row["candidate_budget"]) == budget]
        completed = sum(1 for row in budget_rows if row["status"] == "completed")
        failed = sum(1 for row in budget_rows if row["status"] != "completed")
        table_rows.append(
            {
                "candidate_budget": budget,
                "total_runs": len(budget_rows),
                "completed_runs": completed,
                "failed_runs": failed,
            }
        )
    _write_csv(VIS_DIR / csv_name, list(table_rows[0].keys()), table_rows)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    xs = range(len(BUDGETS))
    completed_vals = [row["completed_runs"] for row in table_rows]
    failed_vals = [row["failed_runs"] for row in table_rows]
    ax.bar(xs, completed_vals, color="#4daf4a", label="Completed")
    ax.bar(xs, failed_vals, bottom=completed_vals, color="#e41a1c", label="Failed")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Candidate Budget")
    ax.set_ylabel("Run Count")
    ax.set_title("Cross-Budget Completion Overview")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.savefig(VIS_DIR / png_name, dpi=220, bbox_inches="tight")
    fig.savefig(VIS_DIR / svg_name, bbox_inches="tight")
    plt.close(fig)


def _write_manifest(all_rows: list[dict], completed_rows: list[dict], failed_rows: list[dict]) -> None:
    payload = {
        "source_manifest": str(MANIFEST),
        "budgets_included": BUDGETS,
        "total_rows_found": len(all_rows),
        "completed_rows_used_for_mean_std_figures": len(completed_rows),
        "failed_rows_excluded_from_mean_std_figures": len(failed_rows),
        "failed_runs": [
            {
                "candidate_budget": int(row["candidate_budget"]),
                "repeat_index": int(row["repeat_index"]),
                "status": row["status"],
            }
            for row in failed_rows
        ],
        "notes": [
            "Cross-budget mean/std figures use completed runs only.",
            "Budget 24 repeat 5 is excluded from mean/std calculations because it failed with blank final metric fields in run_manifest.csv.",
        ],
    }
    (VIS_DIR / "cross_budget_visualization_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = _read_rows()
    completed_rows = _completed_rows(all_rows)
    failed_rows = _failed_rows(all_rows)

    raw_export = [
        {
            "candidate_budget": row["candidate_budget"],
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
        if int(row["candidate_budget"]) in BUDGETS
    ]
    _write_csv(VIS_DIR / "cross_budget_all_repeats_raw.csv", list(raw_export[0].keys()), raw_export)

    completed_export = [
        {
            "candidate_budget": row["candidate_budget"],
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
        if int(row["candidate_budget"]) in BUDGETS
    ]
    _write_csv(VIS_DIR / "cross_budget_completed_repeats_raw.csv", list(completed_export[0].keys()), completed_export)

    summary_rows = _cross_budget_summary(completed_rows, ALL_METRICS)
    _write_csv(
        VIS_DIR / "cross_budget_mean_std_summary.csv",
        ["candidate_budget", "metric_key", "metric_label", "unit", "n", "mean", "std", "min", "max", "cv_percent"],
        summary_rows,
    )

    _metric_panel_plot(
        summary_rows,
        DESIGN_METRICS,
        title="Cross-Budget Comparison of Design Variables",
        subtitle="y = mean over completed repeats, error bar = ±1 SD",
        png_name="cross_budget_design_mean_std.png",
        svg_name="cross_budget_design_mean_std.svg",
    )
    _metric_panel_plot(
        summary_rows,
        EXECUTION_METRICS,
        title="Cross-Budget Comparison of Execution Metrics",
        subtitle="y = mean over completed repeats, error bar = ±1 SD",
        png_name="cross_budget_execution_mean_std.png",
        svg_name="cross_budget_execution_mean_std.svg",
    )
    _plot_cv_heatmap(
        summary_rows,
        png_name="cross_budget_cv_heatmap.png",
        svg_name="cross_budget_cv_heatmap.svg",
    )
    _plot_completion_overview(
        [row for row in all_rows if int(row["candidate_budget"]) in BUDGETS],
        png_name="cross_budget_completion_overview.png",
        svg_name="cross_budget_completion_overview.svg",
        csv_name="cross_budget_completion_overview.csv",
    )
    _write_manifest(all_rows, completed_rows, failed_rows)


if __name__ == "__main__":
    main()
