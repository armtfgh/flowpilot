from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "matrix_manifest.csv"
OUTDIR = ROOT / "visualizations"

UPSTREAM_ORDER = ["claude", "gpt4o", "gpt4omini"]
COUNCIL_ORDER = ["claude", "gpt4o", "gpt4omini"]
DISPLAY = {
    "claude": "Claude",
    "gpt4o": "GPT-4o",
    "gpt4omini": "GPT-4o mini",
}


def read_rows() -> list[dict]:
    with MANIFEST.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_completed_rows(rows: list[dict]) -> list[dict]:
    completed = []
    for row in rows:
        item = dict(row)
        for key in (
            "runtime_s",
            "llm_call_count",
            "total_tokens",
            "final_tau_min",
            "final_flow_rate_mL_min",
            "final_tubing_ID_mm",
            "final_BPR_bar",
            "final_reactor_volume_mL",
            "llm_event_count",
        ):
            item[key] = as_float(item.get(key, ""))
        completed.append(item)
    return completed


def matrix_for(rows: list[dict], key: str, *, status_mode: bool = False) -> np.ndarray:
    mat = np.full((len(UPSTREAM_ORDER), len(COUNCIL_ORDER)), np.nan, dtype=float)
    for row in rows:
        u = row["upstream_bundle"]
        c = row["council_bundle"]
        if u not in UPSTREAM_ORDER or c not in COUNCIL_ORDER:
            continue
        i = UPSTREAM_ORDER.index(u)
        j = COUNCIL_ORDER.index(c)
        if status_mode:
            mat[i, j] = 1.0 if row["status"] == "completed" else 0.0
        else:
            val = row.get(key)
            if val is not None:
                mat[i, j] = float(val)
    return mat


def export_matrix_csv(path: Path, mat: np.ndarray) -> None:
    rows = []
    for i, u in enumerate(UPSTREAM_ORDER):
        row = {"upstream_bundle": u}
        for j, c in enumerate(COUNCIL_ORDER):
            value = mat[i, j]
            row[c] = "" if np.isnan(value) else value
        rows.append(row)
    write_csv(path, ["upstream_bundle"] + COUNCIL_ORDER, rows)


def draw_heatmap(
    mat: np.ndarray,
    title: str,
    cbar_label: str,
    out_png: Path,
    out_svg: Path,
    *,
    fmt: str = ".1f",
    cmap: str = "YlGnBu",
    fail_text: str = "FAIL",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plot_mat = np.ma.masked_invalid(mat)
    im = ax.imshow(plot_mat, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(COUNCIL_ORDER)))
    ax.set_xticklabels([DISPLAY[c] for c in COUNCIL_ORDER], rotation=0)
    ax.set_yticks(range(len(UPSTREAM_ORDER)))
    ax.set_yticklabels([DISPLAY[u] for u in UPSTREAM_ORDER])
    ax.set_xlabel("Council Model")
    ax.set_ylabel("Upstream Bundle")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isnan(value):
                text = fail_text
                color = "black"
            else:
                text = format(value, fmt)
                color = "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def draw_status_heatmap(mat: np.ndarray, out_png: Path, out_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COUNCIL_ORDER)))
    ax.set_xticklabels([DISPLAY[c] for c in COUNCIL_ORDER])
    ax.set_yticks(range(len(UPSTREAM_ORDER)))
    ax.set_yticklabels([DISPLAY[u] for u in UPSTREAM_ORDER])
    ax.set_xlabel("Council Model")
    ax.set_ylabel("Upstream Bundle")
    ax.set_title("Completion Status")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            status = "OK" if mat[i, j] >= 0.5 else "FAIL"
            ax.text(j, i, status, ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Completed = 1, Failed = 0")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def draw_cost_bars(rows: list[dict], out_png: Path, out_svg: Path) -> None:
    completed = [r for r in rows if r["status"] == "completed"]
    labels = [f"{DISPLAY[r['upstream_bundle']]}\n{DISPLAY[r['council_bundle']]}" for r in completed]
    runtime = [r["runtime_s"] for r in completed]
    calls = [r["llm_call_count"] for r in completed]
    tokens_k = [r["total_tokens"] / 1000.0 for r in completed]

    x = np.arange(len(completed))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.bar(x - width, runtime, width, label="Runtime (s)")
    ax.bar(x, calls, width, label="LLM calls")
    ax.bar(x + width, tokens_k, width, label="Tokens (k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mixed units")
    ax.set_title("Cost / Execution Comparison")
    ax.legend()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    rows_typed = build_completed_rows(rows)

    raw_csv = OUTDIR / "model_matrix_raw.csv"
    completed_csv = OUTDIR / "model_matrix_completed.csv"
    write_csv(raw_csv, list(rows[0].keys()), rows)
    write_csv(
        completed_csv,
        list(rows_typed[0].keys()),
        [r for r in rows_typed if r["status"] == "completed"],
    )

    status_mat = matrix_for(rows_typed, "status", status_mode=True)
    export_matrix_csv(OUTDIR / "status_matrix.csv", status_mat)
    draw_status_heatmap(
        status_mat,
        OUTDIR / "status_heatmap.png",
        OUTDIR / "status_heatmap.svg",
    )

    metrics = [
        ("runtime_s", "Runtime Heatmap", "Seconds", ".0f", "YlGnBu"),
        ("llm_call_count", "LLM Call Count", "Calls", ".0f", "YlGnBu"),
        ("total_tokens", "Total Token Usage", "Tokens", ".0f", "YlGnBu"),
        ("final_tau_min", "Final Residence Time", "Minutes", ".1f", "YlGnBu"),
        ("final_flow_rate_mL_min", "Final Flow Rate", "mL/min", ".3f", "YlGnBu"),
        ("final_tubing_ID_mm", "Final Tubing ID", "mm", ".2f", "YlGnBu"),
        ("final_BPR_bar", "Final BPR", "bar", ".1f", "YlGnBu"),
        ("final_reactor_volume_mL", "Final Reactor Volume", "mL", ".3f", "YlGnBu"),
    ]

    summary_rows = []
    for key, title, label, fmt, cmap in metrics:
        mat = matrix_for(rows_typed, key)
        export_matrix_csv(OUTDIR / f"{key}_matrix.csv", mat)
        draw_heatmap(
            mat,
            title,
            label,
            OUTDIR / f"{key}_heatmap.png",
            OUTDIR / f"{key}_heatmap.svg",
            fmt=fmt,
            cmap=cmap,
        )
        for row in rows_typed:
            summary_rows.append(
                {
                    "metric": key,
                    "upstream_bundle": row["upstream_bundle"],
                    "council_bundle": row["council_bundle"],
                    "status": row["status"],
                    "value": row.get(key),
                }
            )

    write_csv(
        OUTDIR / "metric_long_table.csv",
        ["metric", "upstream_bundle", "council_bundle", "status", "value"],
        summary_rows,
    )

    draw_cost_bars(
        rows_typed,
        OUTDIR / "cost_comparison_bars.png",
        OUTDIR / "cost_comparison_bars.svg",
    )

    manifest = {
        "source_manifest": str(MANIFEST),
        "rows_total": len(rows),
        "rows_completed": sum(1 for r in rows if r["status"] == "completed"),
        "rows_failed": sum(1 for r in rows if r["status"] != "completed"),
        "upstream_order": UPSTREAM_ORDER,
        "council_order": COUNCIL_ORDER,
        "generated_files": sorted(p.name for p in OUTDIR.iterdir()),
    }
    with (OUTDIR / "visualization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
