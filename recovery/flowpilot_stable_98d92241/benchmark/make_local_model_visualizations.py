from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_UPSTREAM_ORDER = ["claude", "gemma"]
DEFAULT_COUNCIL_ORDER = ["claude", "gemma"]
DISPLAY = {
    "claude": "Claude",
    "gemma": "Gemma 31B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualizations for a local model benchmark.")
    parser.add_argument("experiment_dir", help="Path to local_model_benchmark_* directory")
    return parser.parse_args()


def read_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
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


def normalize_rows(rows: list[dict]) -> list[dict]:
    out = []
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
        out.append(item)
    return out


def matrix_for(rows: list[dict], key: str, upstream_order: list[str], council_order: list[str], *, status_mode: bool = False) -> np.ndarray:
    mat = np.full((len(upstream_order), len(council_order)), np.nan, dtype=float)
    for row in rows:
        u = row["upstream_bundle"]
        c = row["council_bundle"]
        if u not in upstream_order or c not in council_order:
            continue
        i = upstream_order.index(u)
        j = council_order.index(c)
        if status_mode:
            mat[i, j] = 1.0 if row["status"] == "completed" else 0.0
        else:
            val = row.get(key)
            if val is not None:
                mat[i, j] = float(val)
    return mat


def export_matrix_csv(path: Path, mat: np.ndarray, upstream_order: list[str], council_order: list[str]) -> None:
    rows = []
    for i, u in enumerate(upstream_order):
        row = {"upstream_bundle": u}
        for j, c in enumerate(council_order):
            value = mat[i, j]
            row[c] = "" if np.isnan(value) else value
        rows.append(row)
    write_csv(path, ["upstream_bundle"] + council_order, rows)


def draw_heatmap(mat: np.ndarray, title: str, cbar_label: str, out_png: Path, out_svg: Path, upstream_order: list[str], council_order: list[str], *, fmt: str = ".1f", cmap: str = "YlGnBu", fail_text: str = "NA") -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    plot_mat = np.ma.masked_invalid(mat)
    im = ax.imshow(plot_mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(council_order)))
    ax.set_xticklabels([DISPLAY.get(c, c) for c in council_order])
    ax.set_yticks(range(len(upstream_order)))
    ax.set_yticklabels([DISPLAY.get(u, u) for u in upstream_order])
    ax.set_xlabel("Council Model")
    ax.set_ylabel("Upstream Model")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            text = fail_text if np.isnan(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def draw_status_heatmap(mat: np.ndarray, out_png: Path, out_svg: Path, upstream_order: list[str], council_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    im = ax.imshow(mat, cmap=plt.get_cmap("RdYlGn"), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(council_order)))
    ax.set_xticklabels([DISPLAY.get(c, c) for c in council_order])
    ax.set_yticks(range(len(upstream_order)))
    ax.set_yticklabels([DISPLAY.get(u, u) for u in upstream_order])
    ax.set_xlabel("Council Model")
    ax.set_ylabel("Upstream Model")
    ax.set_title("Completion Status")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            status = "OK" if mat[i, j] >= 0.5 else "NA"
            ax.text(j, i, status, ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Completed = 1")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def draw_cost_bars(rows: list[dict], out_png: Path, out_svg: Path) -> None:
    completed = [r for r in rows if r["status"] == "completed"]
    labels = [f"{DISPLAY.get(r['upstream_bundle'], r['upstream_bundle'])}\n{DISPLAY.get(r['council_bundle'], r['council_bundle'])}" for r in completed]
    runtime = [r["runtime_s"] for r in completed]
    calls = [r["llm_call_count"] for r in completed]
    tokens_k = [r["total_tokens"] / 1000.0 for r in completed]

    x = np.arange(len(completed))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.bar(x - width, runtime, width, label="Runtime (s)")
    ax.bar(x, calls, width, label="LLM calls")
    ax.bar(x + width, tokens_k, width, label="Tokens (k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mixed units")
    ax.set_title("Local Benchmark Cost Comparison")
    ax.legend()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_dir).resolve()
    manifest_path = root / "matrix_manifest.csv"
    outdir = root / "visualizations"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(manifest_path)
    rows_typed = normalize_rows(rows)
    upstream_order = [u for u in DEFAULT_UPSTREAM_ORDER if any(r["upstream_bundle"] == u for r in rows)]
    council_order = [c for c in DEFAULT_COUNCIL_ORDER if any(r["council_bundle"] == c for r in rows)]

    write_csv(outdir / "local_model_matrix_raw.csv", list(rows[0].keys()), rows)
    write_csv(outdir / "local_model_matrix_completed.csv", list(rows_typed[0].keys()), [r for r in rows_typed if r["status"] == "completed"])

    status_mat = matrix_for(rows_typed, "status", upstream_order, council_order, status_mode=True)
    export_matrix_csv(outdir / "status_matrix.csv", status_mat, upstream_order, council_order)
    draw_status_heatmap(status_mat, outdir / "status_heatmap.png", outdir / "status_heatmap.svg", upstream_order, council_order)

    metrics = [
        ("runtime_s", "Runtime", "Seconds", ".0f"),
        ("llm_call_count", "LLM Call Count", "Calls", ".0f"),
        ("total_tokens", "Total Tokens", "Tokens", ".0f"),
        ("final_tau_min", "Final Residence Time", "Minutes", ".1f"),
        ("final_flow_rate_mL_min", "Final Flow Rate", "mL/min", ".3f"),
        ("final_tubing_ID_mm", "Final Tubing ID", "mm", ".2f"),
        ("final_BPR_bar", "Final BPR", "bar", ".1f"),
        ("final_reactor_volume_mL", "Final Reactor Volume", "mL", ".3f"),
    ]

    long_rows = []
    for key, title, label, fmt in metrics:
        mat = matrix_for(rows_typed, key, upstream_order, council_order)
        export_matrix_csv(outdir / f"{key}_matrix.csv", mat, upstream_order, council_order)
        draw_heatmap(mat, title, label, outdir / f"{key}_heatmap.png", outdir / f"{key}_heatmap.svg", upstream_order, council_order, fmt=fmt)
        for row in rows_typed:
            long_rows.append(
                {
                    "metric": key,
                    "upstream_bundle": row["upstream_bundle"],
                    "council_bundle": row["council_bundle"],
                    "status": row["status"],
                    "value": row.get(key),
                }
            )

    write_csv(outdir / "metric_long_table.csv", ["metric", "upstream_bundle", "council_bundle", "status", "value"], long_rows)
    draw_cost_bars(rows_typed, outdir / "cost_comparison_bars.png", outdir / "cost_comparison_bars.svg")

    manifest = {
        "source_manifest": str(manifest_path),
        "rows_total": len(rows),
        "rows_completed": sum(1 for r in rows if r["status"] == "completed"),
        "upstream_order": upstream_order,
        "council_order": council_order,
        "generated_files": sorted(p.name for p in outdir.iterdir()),
    }
    with (outdir / "visualization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
