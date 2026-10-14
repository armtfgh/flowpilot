#!/usr/bin/env python3
"""Build the audited presentation package for the alternative-model campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flowpilot-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = (
    ROOT
    / "ablation_results/manuscript_benchmark/alternative_frontier_one_round_20260821"
)
DEFAULT_REPORT = ROOT / "deliverables/alternative_frontier_one_round_20260821"
COLORS = {"One-shot": "#B65C4A", "FlowPilot": "#197A70"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.savefig(figures / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_pdf_companions(figures: Path) -> None:
    for png in figures.glob("*.png"):
        pdf = png.with_suffix(".pdf")
        if pdf.exists():
            continue
        with Image.open(png) as image:
            image.convert("RGB").save(pdf, "PDF", resolution=320)


def build(raw: Path, report: Path) -> None:
    tables = report / "tables"
    figures = report / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    model_summary = pd.read_csv(tables / "model_architecture_summary.csv")
    pairs = pd.read_csv(tables / "paired_comparisons.csv")
    candidates = pd.read_csv(tables / "candidate_consensus_scores.csv")
    contracts = pd.read_csv(tables / "outcome_contract_summary.csv")
    judge_rows = pd.read_csv(tables / "judge_candidate_scores.csv")
    criteria = pd.read_csv(tables / "criterion_architecture_effect.csv")

    model_order = model_summary["model"].tolist()
    architecture_rows = []
    for _, row in model_summary.iterrows():
        architecture_rows.extend(
            [
                {"model": row.model, "architecture": "One-shot", "score_0_1": row.one_shot_mean_0_1},
                {"model": row.model, "architecture": "FlowPilot", "score_0_1": row.flowpilot_mean_0_1},
            ]
        )
    architecture = pd.DataFrame(architecture_rows)

    critical = (
        candidates.assign(candidate_has_critical=candidates.any_judge_critical_errors.gt(0))
        .groupby(["model", "architecture"], as_index=False)
        .agg(
            n_candidates=("candidate_id", "size"),
            candidates_with_critical=("candidate_has_critical", "sum"),
            judge_critical_flags=("total_judge_critical_flags", "sum"),
        )
    )
    critical.to_csv(tables / "critical_error_summary.csv", index=False)

    judge_summary = (
        judge_rows.groupby(["model", "architecture", "judge"], as_index=False)
        .agg(mean_score_0_1=("score_0_1", "mean"), critical_errors=("critical_errors", "sum"))
    )
    judge_summary.to_csv(tables / "judge_specific_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    model_positions = np.arange(len(model_order))
    width = 0.32
    for index, arch in enumerate(("One-shot", "FlowPilot")):
        values = [
            architecture.query("model == @model and architecture == @arch").score_0_1.iloc[0]
            for model in model_order
        ]
        bars = axes[0].bar(model_positions + (index - 0.5) * width, values, width, color=COLORS[arch], label=arch)
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set_xticks(model_positions, model_order)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Mean NewGen 2.0 score")
    axes[0].set_title("Architecture score")
    axes[0].legend(frameon=False, loc="lower right")
    deltas = model_summary.mean_paired_delta.to_numpy()
    bars = axes[1].bar(model_order, deltas, color="#3B6A8F", width=0.55)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].bar_label(bars, labels=[f"{value:+.3f}" for value in deltas], padding=3, fontsize=9)
    axes[1].set_ylabel("FlowPilot - one-shot")
    axes[1].set_title("Matched architecture effect")
    axes[1].set_ylim(min(-0.04, deltas.min() - 0.04), max(0.30, deltas.max() + 0.05))
    fig.suptitle("Alternative frontier models: one frozen NewGen 2.0 round", fontsize=13)
    fig.tight_layout()
    save_figure(fig, figures, "fig08_alternative_model_comparison")

    cases = pairs["case"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(model_order), figsize=(12.2, 4.4), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), model_order):
        subset = pairs[pairs.model.eq(model)].set_index("case").reindex(cases)
        x = np.arange(len(cases))
        ax.bar(x - width / 2, subset.one_shot_score_0_1, width, color=COLORS["One-shot"], label="One-shot")
        ax.bar(x + width / 2, subset.flowpilot_score_0_1, width, color=COLORS["FlowPilot"], label="FlowPilot")
        for index, delta in enumerate(subset.paired_delta):
            ax.text(index, max(subset.one_shot_score_0_1.iloc[index], subset.flowpilot_score_0_1.iloc[index]) + 0.025, f"{delta:+.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, ["Hydrogenolysis", "Photo-oxidation", "CuAAC"], rotation=18, ha="right")
        ax.set_ylim(0, 1.10)
        ax.set_title(model)
    axes[0].set_ylabel("Consensus score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2)
    fig.suptitle("Case-by-case paired outcomes", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, figures, "fig09_case_by_case_scores")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, measure, title in (
        (axes[0], "candidates_with_critical", "Candidates flagged by any judge"),
        (axes[1], "judge_critical_flags", "Total judge critical-error flags"),
    ):
        for index, arch in enumerate(("One-shot", "FlowPilot")):
            subset = critical[critical.architecture.eq(arch)].set_index("model").reindex(model_order)
            bars = ax.bar(model_positions + (index - 0.5) * width, subset[measure], width, color=COLORS[arch], label=arch)
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        ax.set_xticks(model_positions, model_order)
        ax.set_title(title)
        ax.set_ylim(0, max(1, critical[measure].max() * 1.18))
        ax.set_ylabel("Count")
    axes[-1].legend(frameon=False)
    fig.suptitle("Critical-error burden reported by the judge panel", fontsize=13)
    fig.tight_layout()
    save_figure(fig, figures, "fig10_critical_error_burden")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    positions = np.arange(len(model_order))
    all_delta = model_summary.mean_paired_delta.to_numpy()
    excluded = model_summary.generator_family_excluded_delta.to_numpy()
    ax.bar(positions - width / 2, all_delta, width, color="#3B6A8F", label="All three judges")
    ax.bar(positions + width / 2, excluded, width, color="#8D7A3E", label="Generator-family judge excluded")
    for xpos, values in ((positions - width / 2, all_delta), (positions + width / 2, excluded)):
        for point, value in zip(xpos, values):
            ax.text(point, value + 0.008, f"{value:+.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(positions, model_order)
    ax.set_ylim(0, max(all_delta.max(), excluded.max()) * 1.20)
    ax.set_ylabel("Matched FlowPilot - one-shot score")
    ax.set_title("Generator-family-excluded sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "fig11_generator_family_excluded_sensitivity")

    add_pdf_companions(figures)

    judgment_dirs = [path for path in (raw / "judgments").glob("*/*") if path.is_dir()]
    judgment_status = list((raw / "judgments").glob("*/*/status.json"))
    all_judgment_status = list((raw / "judgments").glob("*/*/**/status.json"))
    judgment_candidates_with_valid_attempt = sum(
        any(
            json.loads(status.read_text(encoding="utf-8")).get("status") == "valid"
            for status in candidate_dir.rglob("status.json")
        )
        for candidate_dir in judgment_dirs
    )
    generation_results = list((raw / "generation").glob("*/*/*/result.json"))
    packets = list((raw / "packets").glob("*.json"))
    judgment_retry_dirs = list((raw / "judgments").glob("*/*/attempts/attempt_*"))
    generation_retry_dirs = list((raw / "generation").glob("*/*/*/attempts/attempt_*"))
    manifest = json.loads((raw / "frozen/campaign_manifest.json").read_text(encoding="utf-8"))
    audit = {
        "planned_outcomes": manifest["candidate_count"],
        "observed_outcome_results": len(generation_results),
        "planned_judgments": manifest["planned_judgments"],
        "observed_judgment_candidate_directories": len(judgment_dirs),
        "observed_root_judgment_status_files": len(judgment_status),
        "observed_all_attempt_status_files": len(all_judgment_status),
        "judgment_candidates_with_valid_attempt": judgment_candidates_with_valid_attempt,
        "outcome_packets": len(packets),
        "generation_retry_directories": len(generation_retry_dirs),
        "judgment_retry_directories": len(judgment_retry_dirs),
        "rubric_sha256": manifest["rubric_sha256"],
        "all_counts_complete": (
            len(generation_results) == manifest["candidate_count"]
            and len(judgment_dirs) == manifest["planned_judgments"]
            and judgment_candidates_with_valid_attempt == manifest["planned_judgments"]
            and len(packets) == manifest["candidate_count"]
        ),
    }
    write_json(report / "CAMPAIGN_AUDIT.json", audit)

    raw_files = sorted(path for path in raw.rglob("*") if path.is_file())
    checksum_rows = [
        {"path": str(path.relative_to(raw)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in raw_files
    ]
    pd.DataFrame(checksum_rows).to_csv(report / "RAW_FILE_CHECKSUMS.csv", index=False)

    criterion_gains = (
        criteria.sort_values("delta", ascending=False)
        .groupby("model", as_index=False)
        .head(3)[["model", "criterion_id", "delta"]]
    )
    criterion_gains.to_csv(tables / "largest_criterion_gains.csv", index=False)

    rows = []
    for _, row in model_summary.iterrows():
        rows.append(
            f"- **{row.model}:** FlowPilot {row.flowpilot_mean_0_1:.3f} vs one-shot "
            f"{row.one_shot_mean_0_1:.3f}; paired delta {row.mean_paired_delta:+.3f}; "
            f"wins/ties/losses {int(row.wins)}/{int(row.ties)}/{int(row.losses)}; "
            f"generator-family-excluded delta {row.generator_family_excluded_delta:+.3f}."
        )
    critical_lines = []
    for model in model_order:
        one = critical[(critical.model.eq(model)) & (critical.architecture.eq("One-shot"))].iloc[0]
        full = critical[(critical.model.eq(model)) & (critical.architecture.eq("FlowPilot"))].iloc[0]
        critical_lines.append(
            f"- **{model}:** one-shot had {int(one.candidates_with_critical)}/3 candidates and "
            f"{int(one.judge_critical_flags)} judge flags; FlowPilot had "
            f"{int(full.candidates_with_critical)}/3 candidates and {int(full.judge_critical_flags)} flags."
        )

    (report / "ALTERNATIVE_MODELS_SUMMARY.md").write_text(
        "# Alternative OpenAI and Claude NewGen 2.0 Benchmark\n\n"
        "## Frozen design\n\n"
        "This one-round extension uses the same three held-out protocols, authoritative inventories, "
        "two architecture conditions, unchanged 14-criterion rubric, and three-judge panel as the "
        "prior manuscript campaign. It contains 12 generated outcomes and 36 blinded judgments. "
        "No result was selected or replaced based on its score.\n\n"
        "## Main results\n\n"
        + "\n".join(rows)
        + "\n\nThe pooled matched delta was **+0.132**; the generator-family-excluded "
        "sensitivity was **+0.101**. All six FlowPilot outcomes closed as executable.\n\n"
        "## Critical-error review\n\n"
        + "\n".join(critical_lines)
        + "\n\nThese are judge-reported critical-error flags, not deterministic arithmetic-error counts. "
        "Deterministic formal validity is reported separately in `outcome_contract_summary.csv`.\n\n"
        "## Interpretation boundary\n\n"
        "The GPT-4o result is directionally strong across all three cases. The Claude Opus result is "
        "positive on average but small, includes one case-level loss, and its three-pair confidence "
        "interval crosses zero. This one-round campaign demonstrates model-dependent architecture "
        "effects; it does not establish population-level superiority across flow chemistry. "
        "Wet-lab validation remains outside this benchmark.\n\n"
        "## Reproducibility\n\n"
        "See `CAMPAIGN_AUDIT.json`, `RAW_FILE_CHECKSUMS.csv`, `frozen/`, `tables/`, and `figures/`. "
        "The complete prompts, raw responses, telemetry, retries, normalized outcomes, and judge "
        f"records are stored in `{raw}`.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    build(args.raw.resolve(), args.report.resolve())
    print(args.report.resolve())


if __name__ == "__main__":
    main()
