from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model_matrix_raw.csv"
OUTDIR = ROOT
UPSTREAM_ORDER = ["claude", "gpt4o", "gpt4omini"]
COUNCIL_ORDER = ["claude", "gpt4o", "gpt4omini"]
DISPLAY = {
    "claude": "Claude",
    "gpt4o": "GPT-4o",
    "gpt4omini": "GPT-4o mini",
}
UP_COLORS = {"claude": "#355070", "gpt4o": "#b56576", "gpt4omini": "#6d597a"}
CO_MARKERS = {"claude": "o", "gpt4o": "s", "gpt4omini": "^"}


def read_rows() -> list[dict]:
    with SOURCE.open("r", encoding="utf-8") as handle:
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched = []
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
        item["completed"] = item["status"] == "completed"
        if item["completed"]:
            run_summary = load_json(Path(item["run_summary_path"]))
            stage_durations = run_summary.get("stage_durations_ms", {})
            item["stage2_s"] = stage_durations.get("council_stage_2_domain_scoring", 0.0) / 1000.0
            item["stage35_s"] = stage_durations.get("council_stage_3_5_preselection_refinement", 0.0) / 1000.0
            item["stage4_s"] = stage_durations.get("council_stage_4_chief_selection", 0.0) / 1000.0
            item["stage6_s"] = stage_durations.get("council_stage_6_dfmea", 0.0) / 1000.0
        else:
            item["stage2_s"] = None
            item["stage35_s"] = None
            item["stage4_s"] = None
            item["stage6_s"] = None

        if not item["completed"]:
            item["quality_proxy"] = 0.0
        else:
            score = 100.0
            if item["final_BPR_bar"] is not None and item["final_BPR_bar"] <= 0.1:
                score -= 40.0
            if item["final_reactor_volume_mL"] is not None and item["final_reactor_volume_mL"] > 25.0:
                score -= 20.0
            if item["final_tubing_ID_mm"] is not None and item["final_tubing_ID_mm"] < 0.5:
                score -= 10.0
            item["quality_proxy"] = max(score, 0.0)
        enriched.append(item)
    return enriched


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", dpi=220)
    fig.savefig(OUTDIR / f"{stem}.svg")
    plt.close(fig)


def final_regime_scatter(rows: list[dict]) -> None:
    completed = [r for r in rows if r["completed"]]
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    out_rows = []
    for row in completed:
        size = 60 + (row["final_reactor_volume_mL"] or 0) * 6
        ax.scatter(
            row["final_flow_rate_mL_min"],
            row["final_tau_min"],
            s=size,
            c=UP_COLORS[row["upstream_bundle"]],
            marker=CO_MARKERS[row["council_bundle"]],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.6,
        )
        label = f"{DISPLAY[row['upstream_bundle']]}/{DISPLAY[row['council_bundle']]}"
        ax.annotate(label, (row["final_flow_rate_mL_min"], row["final_tau_min"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        out_rows.append({
            "upstream_bundle": row["upstream_bundle"],
            "council_bundle": row["council_bundle"],
            "final_flow_rate_mL_min": row["final_flow_rate_mL_min"],
            "final_tau_min": row["final_tau_min"],
            "final_reactor_volume_mL": row["final_reactor_volume_mL"],
        })
    ax.set_xlabel("Final Flow Rate (mL/min)")
    ax.set_ylabel("Final Residence Time (min)")
    ax.set_title("Final Design Regime Map")
    write_csv(OUTDIR / "model_regime_scatter.csv", list(out_rows[0].keys()), out_rows)
    save_fig(fig, "model_regime_scatter")


def grouped_effects(rows: list[dict]) -> None:
    completed = [r for r in rows if r["completed"]]
    metrics = [("runtime_s", "runtime (s)"), ("total_tokens", "tokens"), ("final_tau_min", "tau"), ("final_reactor_volume_mL", "V_R")]
    out_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, label) in zip(axes, metrics):
        upstream_means = []
        council_means = []
        for u in ["claude", "gpt4o"]:
            vals = [r[key] for r in completed if r["upstream_bundle"] == u]
            upstream_means.append(sum(vals) / len(vals))
            out_rows.append({"group_type": "upstream", "group_name": u, "metric": key, "mean": upstream_means[-1]})
        for c in COUNCIL_ORDER:
            vals = [r[key] for r in completed if r["council_bundle"] == c]
            if vals:
                council_means.append(sum(vals) / len(vals))
                out_rows.append({"group_type": "council", "group_name": c, "metric": key, "mean": council_means[-1]})
            else:
                council_means.append(np.nan)
        x1 = np.arange(2)
        x2 = np.arange(3) + 3.2
        ax.bar(x1, upstream_means, color=[UP_COLORS["claude"], UP_COLORS["gpt4o"]], alpha=0.8)
        ax.bar(x2, council_means, color=["#355070", "#b56576", "#6d597a"], alpha=0.8)
        ax.set_xticks(list(x1) + list(x2))
        ax.set_xticklabels(["U Claude", "U 4o", "C Claude", "C 4o", "C 4o mini"], rotation=20)
        ax.set_title(label)
    write_csv(OUTDIR / "model_grouped_effects.csv", ["group_type", "group_name", "metric", "mean"], out_rows)
    save_fig(fig, "model_grouped_effects")


def stage_runtime_breakdown(rows: list[dict]) -> None:
    completed = [r for r in rows if r["completed"]]
    labels = [f"{DISPLAY[r['upstream_bundle']]}\n{DISPLAY[r['council_bundle']]}" for r in completed]
    stages = [("stage2_s", "Stage 2"), ("stage35_s", "Stage 3.5"), ("stage4_s", "Stage 4"), ("stage6_s", "Stage 6")]
    out_rows = []
    x = np.arange(len(completed))
    bottom = np.zeros(len(completed))
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f"]
    fig, ax = plt.subplots(figsize=(11.2, 5.4), constrained_layout=True)
    for (key, label), color in zip(stages, colors):
        vals = [r[key] for r in completed]
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += np.array(vals)
        for r, v in zip(completed, vals):
            out_rows.append({
                "upstream_bundle": r["upstream_bundle"],
                "council_bundle": r["council_bundle"],
                "stage": key,
                "label": label,
                "seconds": v,
            })
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Council Stage Runtime Breakdown")
    ax.legend()
    write_csv(OUTDIR / "model_stage_runtime_breakdown.csv", ["upstream_bundle", "council_bundle", "stage", "label", "seconds"], out_rows)
    save_fig(fig, "model_stage_runtime_breakdown")


def cost_quality_scatter(rows: list[dict]) -> None:
    completed = [r for r in rows if r["completed"]]
    fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    out_rows = []
    for row in completed:
        ax.scatter(
            row["total_tokens"] / 1000.0,
            row["quality_proxy"],
            s=70 + row["llm_call_count"] * 2,
            c=UP_COLORS[row["upstream_bundle"]],
            marker=CO_MARKERS[row["council_bundle"]],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
        )
        label = f"{DISPLAY[row['upstream_bundle']]}/{DISPLAY[row['council_bundle']]}"
        ax.annotate(label, (row["total_tokens"] / 1000.0, row["quality_proxy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        out_rows.append({
            "upstream_bundle": row["upstream_bundle"],
            "council_bundle": row["council_bundle"],
            "total_tokens_k": row["total_tokens"] / 1000.0,
            "quality_proxy": row["quality_proxy"],
            "llm_call_count": row["llm_call_count"],
        })
    ax.set_xlabel("Total tokens (k)")
    ax.set_ylabel("Quality proxy score")
    ax.set_title("Cost vs Quality Proxy")
    write_csv(OUTDIR / "model_cost_quality.csv", list(out_rows[0].keys()), out_rows)
    save_fig(fig, "model_cost_quality")


def radar_chart(rows: list[dict]) -> None:
    completed = [r for r in rows if r["completed"]]
    metrics = [
        ("runtime_s", "fast", "min"),
        ("total_tokens", "cheap", "min"),
        ("llm_call_count", "few calls", "min"),
        ("quality_proxy", "quality", "max"),
        ("final_reactor_volume_mL", "compact", "min"),
        ("final_flow_rate_mL_min", "throughput", "max"),
    ]
    normalized_rows = []
    values_by_metric = {k: [r[k] for r in completed] for k, _, _ in metrics}
    categories = [label for _, label, _ in metrics]
    angles = np.linspace(0, 2 * math.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.6, 8.2), subplot_kw={"polar": True}, constrained_layout=True)
    for row in completed:
        vals = []
        out = {"upstream_bundle": row["upstream_bundle"], "council_bundle": row["council_bundle"]}
        for key, label, direction in metrics:
            lo, hi = min(values_by_metric[key]), max(values_by_metric[key])
            if hi == lo:
                norm = 1.0
            else:
                frac = (row[key] - lo) / (hi - lo)
                norm = 1 - frac if direction == "min" else frac
            vals.append(norm)
            out[key] = row[key]
            out[f"{key}_normalized"] = norm
        vals += vals[:1]
        normalized_rows.append(out)
        ax.plot(angles, vals, linewidth=1.7, label=f"{DISPLAY[row['upstream_bundle']]}/{DISPLAY[row['council_bundle']]}")
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title("Normalized Model-Pair Profile", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    write_csv(OUTDIR / "model_radar_normalized.csv", list(normalized_rows[0].keys()), normalized_rows)
    save_fig(fig, "model_radar_profile")


def main() -> None:
    rows = read_rows()
    enriched = enrich_rows(rows)
    write_csv(OUTDIR / "model_matrix_enriched.csv", list(enriched[0].keys()), enriched)
    final_regime_scatter(enriched)
    grouped_effects(enriched)
    stage_runtime_breakdown(enriched)
    cost_quality_scatter(enriched)
    radar_chart(enriched)
    manifest = {
        "source_csv": str(SOURCE),
        "row_count": len(enriched),
        "completed_count": sum(1 for r in enriched if r["completed"]),
        "generated_files": sorted(p.name for p in OUTDIR.glob("model_*")) + sorted(p.name for p in OUTDIR.glob("cost_*")),
    }
    with (OUTDIR / "model_extra_visualization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
