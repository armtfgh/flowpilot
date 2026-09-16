from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "claude",
    "gpt4o",
    "gpt4omini",
    "gemma",
    "gpt4omini_rescued",
    "gemma_rescued",
]
DISPLAY = {
    "claude": "Claude",
    "gpt4o": "GPT-4o",
    "gpt4omini": "GPT-4o mini",
    "gemma": "Gemma",
    "gpt4omini_rescued": "GPT-4o mini\nrescued",
    "gemma_rescued": "Gemma\nrescued",
}
COLORS = {
    "claude": "#355070",
    "gpt4o": "#b56576",
    "gpt4omini": "#6d597a",
    "gemma": "#2a9d8f",
    "gpt4omini_rescued": "#9d4edd",
    "gemma_rescued": "#f4a261",
}
MARKERS = {
    "claude": "o",
    "gpt4o": "s",
    "gpt4omini": "^",
    "gemma": "D",
}
QUALITY_SCORE = {
    "screen_required": 0.0,
    "deintensified_longer_than_batch": 1.0,
    "batch_equivalent_tau": 2.0,
    "some_intensification_but_below_target": 3.0,
    "meets_3x_rescue_target": 4.0,
    "validated": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualizations for model benchmark folders.")
    parser.add_argument("experiment_dirs", nargs="+", help="Benchmark directories containing matrix_manifest.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def as_bool(value: object) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def ordered(values: set[str]) -> list[str]:
    known = [x for x in MODEL_ORDER if x in values]
    unknown = sorted(values - set(known))
    return known + unknown


def display(name: str) -> str:
    return DISPLAY.get(name, name)


def load_json(path_value: str) -> dict:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_rows(root: Path) -> list[dict]:
    manifest_rows = read_csv(root / "matrix_manifest.csv")
    audit_rows = read_csv(root / "post_run_audit.csv")
    quality_rows = read_csv(root / "final_quality_audit.csv")

    audit_by_key = {(r.get("upstream") or r.get("upstream_bundle"), r.get("council") or r.get("council_bundle")): r for r in audit_rows}
    quality_by_key = {(r.get("upstream") or r.get("upstream_bundle"), r.get("council") or r.get("council_bundle")): r for r in quality_rows}
    rows: list[dict] = []

    for row in manifest_rows:
        item = dict(row)
        key = (item.get("upstream_bundle"), item.get("council_bundle"))
        audit = audit_by_key.get(key, {})
        quality = quality_by_key.get(key, {})

        for name in (
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
            item[name] = as_float(item.get(name))

        item["engine_validated"] = as_bool(audit.get("engine_validated"))
        item["screen_required"] = as_bool(audit.get("screen_required"))
        item["critical"] = as_float(audit.get("critical"))
        item["total_checks"] = as_float(audit.get("total_checks"))
        item["screen_count"] = as_float(audit.get("screen_count"))
        item["concerns"] = audit.get("concerns", "")
        item["concern_count"] = 0 if not item["concerns"] else len(str(item["concerns"]).split(";"))

        batch_time = as_float(quality.get("batch_time_min")) or 15.0
        item["batch_time_min"] = batch_time
        tau = item.get("final_tau_min")
        item["batch_over_tau_reduction_factor"] = as_float(quality.get("batch_over_tau_reduction_factor"))
        if item["batch_over_tau_reduction_factor"] is None and tau:
            item["batch_over_tau_reduction_factor"] = batch_time / tau

        quality_flag = quality.get("quality_flag")
        if not quality_flag:
            if item["screen_required"]:
                quality_flag = "screen_required"
            elif item["engine_validated"]:
                quality_flag = "validated"
            else:
                quality_flag = "not_validated"
        item["quality_flag"] = quality_flag
        item["quality_score"] = QUALITY_SCORE.get(str(quality_flag), np.nan)

        run_summary = load_json(str(item.get("run_summary_path") or ""))
        stage_ms = run_summary.get("stage_durations_ms", {})
        for stage in (
            "council_stage_2_domain_scoring",
            "council_stage_3_5_preselection_refinement",
            "council_stage_4_chief_selection",
            "council_stage_6_dfmea",
        ):
            item[f"{stage}_s"] = as_float(stage_ms.get(stage, 0.0)) / 1000.0 if stage_ms else 0.0
        rows.append(item)
    return rows


def matrix_for(rows: list[dict], key: str, upstreams: list[str], councils: list[str]) -> np.ndarray:
    mat = np.full((len(upstreams), len(councils)), np.nan, dtype=float)
    for row in rows:
        u = row["upstream_bundle"]
        c = row["council_bundle"]
        if u not in upstreams or c not in councils:
            continue
        val = row.get(key)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        val = as_float(val)
        if val is not None:
            mat[upstreams.index(u), councils.index(c)] = val
    return mat


def export_matrix(path: Path, mat: np.ndarray, upstreams: list[str], councils: list[str]) -> None:
    rows = []
    for i, upstream in enumerate(upstreams):
        row = {"upstream_bundle": upstream}
        for j, council in enumerate(councils):
            row[council] = "" if np.isnan(mat[i, j]) else mat[i, j]
        rows.append(row)
    write_csv(path, ["upstream_bundle"] + councils, rows)


def heatmap(
    mat: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    out_stem: Path,
    upstreams: list[str],
    councils: list[str],
    fmt: str = ".1f",
    cmap: str = "YlGnBu",
    vmin: float | None = None,
    vmax: float | None = None,
    nan_text: str = "NA",
) -> None:
    width = max(7.0, 1.15 * len(councils) + 2.0)
    height = max(4.4, 0.75 * len(upstreams) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(councils)))
    ax.set_xticklabels([display(c) for c in councils], rotation=20, ha="right")
    ax.set_yticks(range(len(upstreams)))
    ax.set_yticklabels([display(u) for u in upstreams])
    ax.set_xlabel("Council model")
    ax.set_ylabel("Upstream model")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            text = nan_text if np.isnan(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def status_heatmap(rows: list[dict], outdir: Path, upstreams: list[str], councils: list[str]) -> None:
    mat = np.full((len(upstreams), len(councils)), np.nan, dtype=float)
    text = [["" for _ in councils] for _ in upstreams]
    for row in rows:
        i = upstreams.index(row["upstream_bundle"])
        j = councils.index(row["council_bundle"])
        if row["engine_validated"] is True:
            mat[i, j] = 2
            text[i][j] = "valid"
        elif row["screen_required"] is True:
            mat[i, j] = 1
            text[i][j] = "screen"
        else:
            mat[i, j] = 0
            text[i][j] = "fail"
    fig, ax = plt.subplots(figsize=(max(7, len(councils) * 1.2 + 2), max(4.4, len(upstreams) * 0.75 + 2)), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(councils)))
    ax.set_xticklabels([display(c) for c in councils], rotation=20, ha="right")
    ax.set_yticks(range(len(upstreams)))
    ax.set_yticklabels([display(u) for u in upstreams])
    ax.set_xlabel("Council model")
    ax.set_ylabel("Upstream model")
    ax.set_title("Validation Status")
    for i in range(len(upstreams)):
        for j in range(len(councils)):
            ax.text(j, i, text[i][j] or "NA", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["fail", "screen", "valid"])
    fig.savefig(outdir / "validation_status_heatmap.png", dpi=220)
    fig.savefig(outdir / "validation_status_heatmap.svg")
    plt.close(fig)
    export_matrix(outdir / "validation_status_matrix.csv", mat, upstreams, councils)


def cost_bars(rows: list[dict], outdir: Path) -> None:
    labels = [f"{display(r['upstream_bundle'])}\n{display(r['council_bundle'])}" for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.7), 5.4), constrained_layout=True)
    ax.bar(x - width, [r["runtime_s"] or 0 for r in rows], width, label="runtime (s)", color="#355070")
    ax.bar(x, [r["llm_call_count"] or 0 for r in rows], width, label="LLM calls", color="#b56576")
    ax.bar(x + width, [(r["total_tokens"] or 0) / 1000 for r in rows], width, label="tokens (k)", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mixed units")
    ax.set_title("Cost Comparison")
    ax.legend()
    fig.savefig(outdir / "cost_comparison_bars.png", dpi=220)
    fig.savefig(outdir / "cost_comparison_bars.svg")
    plt.close(fig)


def regime_scatter(rows: list[dict], outdir: Path) -> None:
    usable = [r for r in rows if r.get("final_flow_rate_mL_min") is not None and r.get("final_tau_min") is not None]
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    out_rows = []
    for row in usable:
        u = row["upstream_bundle"]
        c = row["council_bundle"]
        size = 50 + min((row.get("final_reactor_volume_mL") or 0) * 5, 900)
        ax.scatter(
            row["final_flow_rate_mL_min"],
            row["final_tau_min"],
            s=size,
            c=COLORS.get(u, "#777777"),
            marker=MARKERS.get(c, "o"),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.82,
        )
        ax.annotate(f"{display(u).replace(chr(10), ' ')}/{display(c).replace(chr(10), ' ')}", (row["final_flow_rate_mL_min"], row["final_tau_min"]), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        out_rows.append({
            "upstream_bundle": u,
            "council_bundle": c,
            "final_flow_rate_mL_min": row["final_flow_rate_mL_min"],
            "final_tau_min": row["final_tau_min"],
            "final_reactor_volume_mL": row["final_reactor_volume_mL"],
            "quality_flag": row["quality_flag"],
        })
    ax.axhline(15, color="#444444", linestyle="--", linewidth=1.0, label="batch time 15 min")
    ax.axhline(5, color="#777777", linestyle=":", linewidth=1.0, label="3x target tau 5 min")
    ax.set_xlabel("Final flow rate (mL/min)")
    ax.set_ylabel("Final residence time (min)")
    ax.set_title("Final Design Regime Map")
    ax.legend(loc="best")
    write_csv(outdir / "design_regime_scatter.csv", list(out_rows[0]), out_rows)
    fig.savefig(outdir / "design_regime_scatter.png", dpi=220)
    fig.savefig(outdir / "design_regime_scatter.svg")
    plt.close(fig)


def cost_quality_scatter(rows: list[dict], outdir: Path) -> None:
    usable = [r for r in rows if r.get("total_tokens") is not None and r.get("quality_score") is not None and not np.isnan(r.get("quality_score"))]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    out_rows = []
    for row in usable:
        u = row["upstream_bundle"]
        c = row["council_bundle"]
        ax.scatter(
            (row["total_tokens"] or 0) / 1000,
            row["quality_score"],
            s=60 + (row.get("llm_call_count") or 0) * 1.4,
            c=COLORS.get(u, "#777777"),
            marker=MARKERS.get(c, "o"),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.84,
        )
        ax.annotate(f"{display(u).replace(chr(10), ' ')}/{display(c).replace(chr(10), ' ')}", ((row["total_tokens"] or 0) / 1000, row["quality_score"]), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        out_rows.append({
            "upstream_bundle": u,
            "council_bundle": c,
            "total_tokens_k": (row["total_tokens"] or 0) / 1000,
            "quality_score": row["quality_score"],
            "quality_flag": row["quality_flag"],
            "llm_call_count": row["llm_call_count"],
        })
    ax.set_xlabel("Total tokens (k)")
    ax.set_ylabel("Quality score")
    ax.set_title("Cost vs Validation/Intensification Quality")
    ax.set_yticks(sorted(set(QUALITY_SCORE.values())))
    write_csv(outdir / "cost_quality_scatter.csv", list(out_rows[0]), out_rows)
    fig.savefig(outdir / "cost_quality_scatter.png", dpi=220)
    fig.savefig(outdir / "cost_quality_scatter.svg")
    plt.close(fig)


def stage_breakdown(rows: list[dict], outdir: Path) -> None:
    stages = [
        ("council_stage_2_domain_scoring_s", "domain scoring"),
        ("council_stage_3_5_preselection_refinement_s", "refinement"),
        ("council_stage_4_chief_selection_s", "chief"),
        ("council_stage_6_dfmea_s", "DFMEA"),
    ]
    labels = [f"{display(r['upstream_bundle'])}\n{display(r['council_bundle'])}" for r in rows]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.7), 5.5), constrained_layout=True)
    out_rows = []
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f"]
    for (key, label), color in zip(stages, colors):
        vals = np.array([r.get(key) or 0.0 for r in rows])
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
        for row, val in zip(rows, vals):
            out_rows.append({
                "upstream_bundle": row["upstream_bundle"],
                "council_bundle": row["council_bundle"],
                "stage": key,
                "stage_label": label,
                "seconds": val,
            })
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Council Stage Runtime Breakdown")
    ax.legend()
    write_csv(outdir / "stage_runtime_breakdown.csv", list(out_rows[0]), out_rows)
    fig.savefig(outdir / "stage_runtime_breakdown.png", dpi=220)
    fig.savefig(outdir / "stage_runtime_breakdown.svg")
    plt.close(fig)


def generate(root: Path) -> None:
    root = root.resolve()
    outdir = root / "visualizations"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(root)
    if not rows:
        raise SystemExit(f"No rows found in {root / 'matrix_manifest.csv'}")

    upstreams = ordered({r["upstream_bundle"] for r in rows})
    councils = ordered({r["council_bundle"] for r in rows})

    write_csv(outdir / "model_matrix_enriched.csv", list(rows[0]), rows)
    status_heatmap(rows, outdir, upstreams, councils)

    metrics = [
        ("runtime_s", "Runtime", "seconds", ".0f", "YlGnBu"),
        ("llm_call_count", "LLM Call Count", "calls", ".0f", "YlGnBu"),
        ("total_tokens", "Total Tokens", "tokens", ".0f", "YlGnBu"),
        ("final_tau_min", "Final Residence Time", "min", ".2f", "YlOrRd"),
        ("batch_over_tau_reduction_factor", "Batch/Tau Intensification Factor", "batch time / tau", ".2f", "YlGn"),
        ("final_flow_rate_mL_min", "Final Flow Rate", "mL/min", ".3f", "PuBuGn"),
        ("final_tubing_ID_mm", "Final Tubing ID", "mm", ".2f", "BuPu"),
        ("final_BPR_bar", "Final BPR", "bar", ".1f", "Greens"),
        ("final_reactor_volume_mL", "Final Reactor Volume", "mL", ".3f", "Oranges"),
        ("engine_validated", "Engine Validated", "validated = 1", ".0f", "RdYlGn"),
        ("screen_required", "Screen Required", "screen = 1", ".0f", "Reds"),
        ("critical", "Critical Audit Count", "critical checks", ".0f", "Reds"),
        ("concern_count", "Post-run Concern Count", "concerns", ".0f", "Reds"),
        ("quality_score", "Quality Score", "higher is better", ".0f", "RdYlGn"),
    ]

    long_rows = []
    for key, title, label, fmt, cmap in metrics:
        mat = matrix_for(rows, key, upstreams, councils)
        export_matrix(outdir / f"{key}_matrix.csv", mat, upstreams, councils)
        heatmap(
            mat,
            title=title,
            cbar_label=label,
            out_stem=outdir / f"{key}_heatmap",
            upstreams=upstreams,
            councils=councils,
            fmt=fmt,
            cmap=cmap,
            vmin=0 if key in {"engine_validated", "screen_required", "quality_score"} else None,
            vmax=5 if key == "quality_score" else (1 if key in {"engine_validated", "screen_required"} else None),
        )
        for row in rows:
            long_rows.append({
                "metric": key,
                "upstream_bundle": row["upstream_bundle"],
                "council_bundle": row["council_bundle"],
                "status": row.get("status"),
                "value": row.get(key),
            })
    write_csv(outdir / "metric_long_table.csv", list(long_rows[0]), long_rows)

    cost_bars(rows, outdir)
    regime_scatter(rows, outdir)
    cost_quality_scatter(rows, outdir)
    stage_breakdown(rows, outdir)

    manifest = {
        "source_dir": str(root),
        "rows_total": len(rows),
        "upstream_order": upstreams,
        "council_order": councils,
        "generated_files": sorted(p.name for p in outdir.iterdir()),
    }
    with (outdir / "visualization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    for path in args.experiment_dirs:
        generate(Path(path))


if __name__ == "__main__":
    main()
