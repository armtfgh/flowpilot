from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = BASE_DIR.parent
REPLACEMENT_DIR = Path(
    "/home/amirreza/Documents/codes/Flow Agent/benchmark/data/"
    "protocol_budget_benchmark_20260428_102728"
)

ORIGINAL_MANIFEST = ORIGINAL_DIR / "run_manifest.csv"
REPLACEMENT_MANIFEST = REPLACEMENT_DIR / "run_manifest.csv"

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: str) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _cv_percent(mean_value: float, std_value: float) -> float:
    if mean_value == 0:
        return 0.0
    return 100.0 * std_value / mean_value


def _merge_rows() -> tuple[list[dict[str, str]], dict[str, object]]:
    original_rows = _read_csv(ORIGINAL_MANIFEST)
    replacement_rows = _read_csv(REPLACEMENT_MANIFEST)
    if len(replacement_rows) != 1:
        raise RuntimeError("Expected exactly one replacement run row.")
    replacement = replacement_rows[0]

    merged: list[dict[str, str]] = []
    replaced_count = 0
    failed_originals: list[dict[str, str]] = []
    for row in original_rows:
        if row["candidate_budget"] == "24" and row["repeat_index"] == "5":
            failed_originals.append(row)
            if row["status"] == "failed":
                replaced = dict(replacement)
                replaced["repeat_index"] = "5"
                merged.append(replaced)
                replaced_count += 1
            else:
                merged.append(row)
        else:
            merged.append(row)

    meta = {
        "original_row_count": len(original_rows),
        "replacement_row_count": len(replacement_rows),
        "replaced_count": replaced_count,
        "failed_original_rows": failed_originals,
        "replacement_source_run": replacement,
    }
    return merged, meta


def _group_stats(rows: list[dict[str, str]], metrics: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    completed = [r for r in rows if r["status"] == "completed"]
    by_budget: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in completed:
        by_budget[int(row["candidate_budget"])].append(row)

    summary: list[dict[str, object]] = []
    for budget in BUDGETS:
        budget_rows = by_budget[budget]
        for metric_key, metric_label, unit in metrics:
            values = [_to_float(r[metric_key]) for r in budget_rows]
            mean_value = _mean(values)
            std_value = _std(values)
            summary.append(
                {
                    "candidate_budget": budget,
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "unit": unit,
                    "n": len(values),
                    "mean": mean_value,
                    "std": std_value,
                    "min": min(values),
                    "max": max(values),
                    "cv_percent": _cv_percent(mean_value, std_value),
                }
            )
    return summary


def _plot_metric_grid(summary_rows: list[dict[str, object]], metrics: list[tuple[str, str, str]], out_stem: str) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.8 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_label, unit) in zip(axes, metrics):
        rows = [r for r in summary_rows if r["metric_key"] == metric_key]
        rows.sort(key=lambda r: int(r["candidate_budget"]))
        x = [int(r["candidate_budget"]) for r in rows]
        y = [float(r["mean"]) for r in rows]
        yerr = [float(r["std"]) for r in rows]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            capsize=4,
            lw=2,
            markersize=6,
            color="#1f4e79",
            ecolor="#4f81bd",
        )
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("Candidate Budget")
        ax.set_ylabel(f"{metric_label} ({unit})" if unit else metric_label)
        ax.set_title(f"{metric_label}: mean ± SD across repeats")
        ax.grid(True, axis="y", alpha=0.25)

    png_path = BASE_DIR / f"{out_stem}.png"
    svg_path = BASE_DIR / f"{out_stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_cv_heatmap(summary_rows: list[dict[str, object]], metrics: list[tuple[str, str, str]], out_stem: str) -> None:
    metric_labels = [label for _, label, _ in metrics]
    matrix = []
    for metric_key, _, _ in metrics:
        row = []
        for budget in BUDGETS:
            item = next(r for r in summary_rows if r["metric_key"] == metric_key and int(r["candidate_budget"]) == budget)
            row.append(float(item["cv_percent"]))
        matrix.append(row)

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.55 * len(metrics))))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(BUDGETS)), labels=[str(b) for b in BUDGETS])
    ax.set_yticks(np.arange(len(metric_labels)), labels=metric_labels)
    ax.set_xlabel("Candidate Budget")
    ax.set_title("Coefficient of Variation (%) across budgets")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label="CV (%)")
    fig.tight_layout()
    fig.savefig(BASE_DIR / f"{out_stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(BASE_DIR / f"{out_stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    merged_rows, merge_meta = _merge_rows()
    merged_rows.sort(key=lambda r: (int(r["candidate_budget"]), int(r["repeat_index"])))

    completed_rows = [r for r in merged_rows if r["status"] == "completed"]
    completion_rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        rows = [r for r in merged_rows if int(r["candidate_budget"]) == budget]
        completion_rows.append(
            {
                "candidate_budget": budget,
                "completed_runs": sum(1 for r in rows if r["status"] == "completed"),
                "failed_runs": sum(1 for r in rows if r["status"] != "completed"),
                "total_runs": len(rows),
            }
        )

    design_summary = _group_stats(merged_rows, DESIGN_METRICS)
    execution_summary = _group_stats(merged_rows, EXECUTION_METRICS)
    all_summary = design_summary + execution_summary

    _write_csv(
        BASE_DIR / "cross_budget_repaired_all_repeats_raw.csv",
        merged_rows,
        fieldnames=list(merged_rows[0].keys()),
    )
    _write_csv(
        BASE_DIR / "cross_budget_repaired_completed_repeats_raw.csv",
        completed_rows,
        fieldnames=list(completed_rows[0].keys()),
    )
    _write_csv(
        BASE_DIR / "cross_budget_repaired_mean_std_summary.csv",
        all_summary,
        fieldnames=[
            "candidate_budget",
            "metric_key",
            "metric_label",
            "unit",
            "n",
            "mean",
            "std",
            "min",
            "max",
            "cv_percent",
        ],
    )
    _write_csv(
        BASE_DIR / "cross_budget_repaired_completion_overview.csv",
        completion_rows,
        fieldnames=["candidate_budget", "completed_runs", "failed_runs", "total_runs"],
    )

    _plot_metric_grid(design_summary, DESIGN_METRICS, "cross_budget_repaired_design_mean_std")
    _plot_metric_grid(execution_summary, EXECUTION_METRICS, "cross_budget_repaired_execution_mean_std")
    _plot_cv_heatmap(all_summary, DESIGN_METRICS + EXECUTION_METRICS, "cross_budget_repaired_cv_heatmap")

    manifest = {
        "source_benchmark_dir": str(ORIGINAL_DIR),
        "replacement_benchmark_dir": str(REPLACEMENT_DIR),
        "replacement_policy": "Replace original failed budget_24 repeat_05 with completed replacement budget_24 repeat_01.",
        "budgets": BUDGETS,
        "completed_rows_used_for_mean_std_figures": len(completed_rows),
        "failed_rows_after_repair": sum(1 for r in merged_rows if r["status"] != "completed"),
        "merge_meta": merge_meta,
    }
    (BASE_DIR / "cross_budget_repaired_visualization_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
