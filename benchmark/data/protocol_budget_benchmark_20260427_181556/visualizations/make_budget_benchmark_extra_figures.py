from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cross_budget_repaired_completed_repeats_raw.csv"
OUTDIR = ROOT
BUDGETS = [1, 6, 12, 24]
COLORS = {
    1: "#355070",
    6: "#6d597a",
    12: "#b56576",
    24: "#e56b6f",
}


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


def run_dir_from_summary(summary_path: str) -> Path:
    return Path(summary_path).resolve().parent


def classify_family(row: dict) -> str:
    d = row["final_tubing_ID_mm"]
    q = row["final_flow_rate_mL_min"]
    tau = row["final_tau_min"]
    if d is None or q is None or tau is None:
        return "unknown"
    if d <= 0.55 and q >= 8.0:
        return "small-ID & high flowrate"
    if d <= 0.55 and q < 8.0:
        return "small-ID & mid flowrate"
    if d >= 1.5:
        return "large-ID only"
    if tau >= 20.0:
        return "long tau"
    return "mid-ID balanced"


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["budget"] = int(item["candidate_budget"])
        item["repeat_index"] = int(item["repeat_index"])
        for key in (
            "runtime_s",
            "llm_call_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "final_tau_min",
            "final_flow_rate_mL_min",
            "final_tubing_ID_mm",
            "final_BPR_bar",
            "final_reactor_volume_mL",
        ):
            item[key] = as_float(item.get(key, ""))

        run_dir = run_dir_from_summary(item["summary_path"])
        stage1 = load_json(run_dir / "snapshots" / "stage1_designer_result.json")
        stage2 = load_json(run_dir / "snapshots" / "stage2_initial_scoring.json")
        stage35 = load_json(run_dir / "snapshots" / "stage3_5_refinement_summary.json")
        stage4 = load_json(run_dir / "snapshots" / "stage4_chief_selection.json")

        stage1_survivors = len(stage1.get("survivors", []))
        stage2_blocked = len(stage2.get("blocked_by_scoring", []))
        changed_count = int(stage35.get("changed_count", 0) or 0)
        final_candidate_count = int(stage35.get("final_candidate_count", 0) or 0)
        disqualify_count = len(stage4.get("disqualify_ids", []))
        descendant_total = 0
        for change in stage35.get("candidate_changes", []):
            descendant_total += int(change.get("descendant_count", 0) or 0)

        item["stage1_survivor_count"] = stage1_survivors
        item["stage2_blocked_count"] = stage2_blocked
        item["stage2_post_block_count"] = max(stage1_survivors - stage2_blocked, 0)
        item["revision_changed_count"] = changed_count
        item["revision_final_candidate_count"] = final_candidate_count
        item["revision_descendant_total"] = descendant_total
        item["stage4_disqualify_count"] = disqualify_count
        item["selection_survivor_count"] = max(final_candidate_count - disqualify_count, 0)

        item["pathology_bpr_zero"] = 1 if (item["final_BPR_bar"] is not None and item["final_BPR_bar"] <= 0.1) else 0
        item["pathology_large_volume"] = 1 if (item["final_reactor_volume_mL"] is not None and item["final_reactor_volume_mL"] > 50.0) else 0
        item["pathology_tiny_d"] = 1 if (item["final_tubing_ID_mm"] is not None and item["final_tubing_ID_mm"] < 0.4) else 0
        item["pathology_any"] = 1 if (item["pathology_bpr_zero"] or item["pathology_large_volume"] or item["pathology_tiny_d"]) else 0
        item["design_family"] = classify_family(item)
        enriched.append(item)
    return enriched


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", dpi=220)
    fig.savefig(OUTDIR / f"{stem}.svg")
    plt.close(fig)


def parallel_coordinates(rows: list[dict]) -> None:
    vars_ = [
        ("final_tau_min", "tau"),
        ("final_flow_rate_mL_min", "Q"),
        ("final_tubing_ID_mm", "d"),
        ("final_BPR_bar", "BPR"),
        ("final_reactor_volume_mL", "V_R"),
    ]
    mins = {k: min(r[k] for r in rows) for k, _ in vars_}
    maxs = {k: max(r[k] for r in rows) for k, _ in vars_}
    out_rows = []
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    x = np.arange(len(vars_))
    for row in rows:
        y = []
        for key, _ in vars_:
            lo, hi = mins[key], maxs[key]
            val = row[key]
            norm = 0.5 if hi == lo else (val - lo) / (hi - lo)
            y.append(norm)
            out_rows.append({
                "budget": row["budget"],
                "repeat_index": row["repeat_index"],
                "metric": key,
                "value": val,
                "normalized_value": norm,
            })
        ax.plot(x, y, marker="o", linewidth=1.6, alpha=0.75, color=COLORS[row["budget"]])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in vars_])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["min", "mid", "max"])
    ax.set_title("Parallel Coordinates of Final Design Variables")
    write_csv(OUTDIR / "budget_parallel_coordinates.csv", ["budget", "repeat_index", "metric", "value", "normalized_value"], out_rows)
    save_fig(fig, "budget_parallel_coordinates")


def scatter_matrix(rows: list[dict]) -> None:
    vars_ = [
        ("final_tau_min", "tau"),
        ("final_flow_rate_mL_min", "Q"),
        ("final_tubing_ID_mm", "d"),
        ("final_reactor_volume_mL", "V_R"),
    ]
    n = len(vars_)
    fig, axes = plt.subplots(n, n, figsize=(11, 11), constrained_layout=True)
    for i, (ykey, ylabel) in enumerate(vars_):
        for j, (xkey, xlabel) in enumerate(vars_):
            ax = axes[i, j]
            if i == j:
                series = [r[xkey] for r in rows]
                ax.hist(series, bins=6, color="#adb5bd", edgecolor="white")
            else:
                for budget in BUDGETS:
                    subset = [r for r in rows if r["budget"] == budget]
                    ax.scatter(
                        [r[xkey] for r in subset],
                        [r[ykey] for r in subset],
                        s=28,
                        alpha=0.8,
                        color=COLORS[budget],
                    )
            if i == n - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticklabels([])
    fig.suptitle("Scatter Matrix of Final Design Variables", y=1.02)
    write_csv(
        OUTDIR / "budget_scatter_matrix_source.csv",
        ["budget", "repeat_index"] + [k for k, _ in vars_],
        [
            {"budget": r["budget"], "repeat_index": r["repeat_index"], **{k: r[k] for k, _ in vars_}}
            for r in rows
        ],
    )
    save_fig(fig, "budget_scatter_matrix")


def boxplots(rows: list[dict]) -> None:
    design_vars = [
        ("final_tau_min", "tau (min)"),
        ("final_flow_rate_mL_min", "Q (mL/min)"),
        ("final_tubing_ID_mm", "d (mm)"),
        ("final_reactor_volume_mL", "V_R (mL)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, label) in zip(axes, design_vars):
        data = [[r[key] for r in rows if r["budget"] == budget] for budget in BUDGETS]
        bp = ax.boxplot(data, patch_artist=True, labels=[str(b) for b in BUDGETS])
        for patch, budget in zip(bp["boxes"], BUDGETS):
            patch.set_facecolor(COLORS[budget])
            patch.set_alpha(0.7)
        ax.set_title(label)
        ax.set_xlabel("Budget")
    write_csv(
        OUTDIR / "budget_boxplot_design_source.csv",
        ["budget", "repeat_index"] + [k for k, _ in design_vars],
        [
            {"budget": r["budget"], "repeat_index": r["repeat_index"], **{k: r[k] for k, _ in design_vars}}
            for r in rows
        ],
    )
    save_fig(fig, "budget_boxplot_design")

    exec_vars = [
        ("runtime_s", "runtime (s)"),
        ("llm_call_count", "LLM calls"),
        ("total_tokens", "tokens"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, (key, label) in zip(axes, exec_vars):
        data = [[r[key] for r in rows if r["budget"] == budget] for budget in BUDGETS]
        bp = ax.boxplot(data, patch_artist=True, labels=[str(b) for b in BUDGETS])
        for patch, budget in zip(bp["boxes"], BUDGETS):
            patch.set_facecolor(COLORS[budget])
            patch.set_alpha(0.7)
        ax.set_title(label)
        ax.set_xlabel("Budget")
    write_csv(
        OUTDIR / "budget_boxplot_execution_source.csv",
        ["budget", "repeat_index"] + [k for k, _ in exec_vars],
        [
            {"budget": r["budget"], "repeat_index": r["repeat_index"], **{k: r[k] for k, _ in exec_vars}}
            for r in rows
        ],
    )
    save_fig(fig, "budget_boxplot_execution")


def pathology_plot(rows: list[dict]) -> None:
    metrics = [
        ("pathology_bpr_zero", "BPR = 0"),
        ("pathology_large_volume", "V_R > 50 mL"),
        ("pathology_tiny_d", "d < 0.4 mm"),
        ("pathology_any", "any pathology"),
    ]
    out_rows = []
    fig, ax = plt.subplots(figsize=(9.2, 5), constrained_layout=True)
    x = np.arange(len(BUDGETS))
    width = 0.18
    for idx, (key, label) in enumerate(metrics):
        rates = []
        for budget in BUDGETS:
            subset = [r for r in rows if r["budget"] == budget]
            rate = sum(r[key] for r in subset) / len(subset)
            rates.append(rate)
            out_rows.append({"budget": budget, "metric": key, "label": label, "rate": rate})
        ax.bar(x + (idx - 1.5) * width, rates, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Budget")
    ax.set_title("Pathology Rates by Budget")
    ax.legend()
    write_csv(OUTDIR / "budget_pathology_rates.csv", ["budget", "metric", "label", "rate"], out_rows)
    save_fig(fig, "budget_pathology_rates")


def revision_activity(rows: list[dict]) -> None:
    metrics = [
        ("revision_changed_count", "changed candidates"),
        ("revision_descendant_total", "descendants"),
        ("revision_final_candidate_count", "revised pool"),
        ("stage4_disqualify_count", "disqualified"),
    ]
    out_rows = []
    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    x = np.arange(len(BUDGETS))
    width = 0.18
    for idx, (key, label) in enumerate(metrics):
        means = []
        for budget in BUDGETS:
            subset = [r for r in rows if r["budget"] == budget]
            mean = sum(r[key] for r in subset) / len(subset)
            means.append(mean)
            out_rows.append({"budget": budget, "metric": key, "label": label, "mean": mean})
        ax.bar(x + (idx - 1.5) * width, means, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Budget")
    ax.set_ylabel("Mean count")
    ax.set_title("Revision Activity by Budget")
    ax.legend()
    write_csv(OUTDIR / "budget_revision_activity.csv", ["budget", "metric", "label", "mean"], out_rows)
    save_fig(fig, "budget_revision_activity")


def stage_flow(rows: list[dict]) -> None:
    stages = [
        ("stage1_survivor_count", "stage1"),
        ("stage2_post_block_count", "post-stage2"),
        ("revision_final_candidate_count", "post-revision"),
        ("selection_survivor_count", "post-selection"),
    ]
    out_rows = []
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    x = np.arange(len(stages))
    for budget in BUDGETS:
        subset = [r for r in rows if r["budget"] == budget]
        means = [sum(r[key] for r in subset) / len(subset) for key, _ in stages]
        ax.plot(x, means, marker="o", linewidth=2.0, color=COLORS[budget], label=f"budget {budget}")
        for (key, label), mean in zip(stages, means):
            out_rows.append({"budget": budget, "stage": key, "label": label, "mean_count": mean})
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stages])
    ax.set_ylabel("Mean candidate count")
    ax.set_title("Candidate Pool Flow Through Council Stages")
    ax.legend()
    write_csv(OUTDIR / "budget_stage_flow.csv", ["budget", "stage", "label", "mean_count"], out_rows)
    save_fig(fig, "budget_stage_flow")


def design_family_plot(rows: list[dict]) -> None:
    families = []
    for row in rows:
        if row["design_family"] not in families:
            families.append(row["design_family"])
    family_legend_labels = {
        "small-ID & high flowrate": "small-ID & high flowrate (d <= 0.55 mm, Q >= 8.0 mL/min)",
        "small-ID & mid flowrate": "small-ID & mid flowrate (d <= 0.55 mm, Q < 8.0 mL/min)",
        "large-ID only": "large-ID only (d >= 1.5 mm)",
        "long tau": "long tau (tau >= 20.0 min; not in prior bins)",
        "mid-ID balanced": "mid-ID balanced (0.55 < d < 1.5 mm, tau < 20.0 min)",
        "unknown": "unknown",
    }
    count_rows = []
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    bottom = np.zeros(len(BUDGETS))
    x = np.arange(len(BUDGETS))
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6a4c93"]
    for idx, family in enumerate(families):
        vals = []
        for budget in BUDGETS:
            cnt = sum(1 for r in rows if r["budget"] == budget and r["design_family"] == family)
            vals.append(cnt)
            count_rows.append({"budget": budget, "design_family": family, "count": cnt})
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=palette[idx % len(palette)],
            label=family_legend_labels.get(family, family),
        )
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_ylabel("Run count")
    ax.set_xlabel("Budget")
    ax.set_title("Final Design Family Composition by Budget")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    write_csv(
        OUTDIR / "budget_design_family_counts.csv",
        ["budget", "design_family", "design_family_definition", "count"],
        [
            {
                "budget": row["budget"],
                "design_family": row["design_family"],
                "design_family_definition": family_legend_labels.get(row["design_family"], row["design_family"]),
                "count": row["count"],
            }
            for row in count_rows
        ],
    )
    save_fig(fig, "budget_design_family_counts")


def main() -> None:
    rows = enrich_rows(read_rows())
    write_csv(OUTDIR / "budget_enriched_runs.csv", list(rows[0].keys()), rows)
    parallel_coordinates(rows)
    scatter_matrix(rows)
    boxplots(rows)
    pathology_plot(rows)
    revision_activity(rows)
    stage_flow(rows)
    design_family_plot(rows)
    manifest = {
        "source_csv": str(SOURCE),
        "row_count": len(rows),
        "budgets": BUDGETS,
        "generated_files": sorted(p.name for p in OUTDIR.glob("budget_*")),
    }
    with (OUTDIR / "budget_extra_visualization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
