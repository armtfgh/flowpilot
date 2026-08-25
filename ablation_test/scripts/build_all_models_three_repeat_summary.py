#!/usr/bin/env python3
"""Build the complete NewGen 2.0 three-repeat cross-model package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "deliverables/newgen_2_0_all_models_three_repeat_20260824"
MODEL_ORDER = [
    "Qwen3.6-27B",
    "Qwen3.8-27B",
    "GPT-4o",
    "GPT-5.4",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
    "Claude Opus 5",
]
SOURCES = {
    "Qwen3.6-27B": ROOT / "deliverables/manuscript_three_model_three_repeat_postfix_20260820",
    "Claude Sonnet 4.6": ROOT / "deliverables/manuscript_three_model_three_repeat_postfix_20260820",
    "GPT-5.4": ROOT / "deliverables/openai_confirmatory_stationary_fix_20260820",
    "Qwen3.8-27B": ROOT / "deliverables/qwen38_three_repeat_20260824",
    "GPT-4o": ROOT / "deliverables/alternative_frontier_three_repeat_20260824",
    "Claude Opus 4.6": ROOT / "deliverables/alternative_frontier_three_repeat_20260824",
    "Claude Opus 5": ROOT / "deliverables/opus5_three_repeat_20260824",
}
ARCHITECTURES = ["One-shot", "FlowPilot"]
REPEATS = ["repeat_01", "repeat_02", "repeat_03"]
CASES = ["Hydrogenolysis", "Photochemical oxidation", "CuAAC"]
COLORS = {"One-shot": "#C4473A", "FlowPilot": "#16857A"}


def _read_table(model: str, name: str) -> pd.DataFrame:
    path = SOURCES[model] / "tables" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required source table for {model}: {path}")
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame = frame[frame["model"] == model].copy()
    return frame


def _validate_complete(scores: pd.DataFrame, models: list[str]) -> None:
    expected = pd.MultiIndex.from_product(
        [models, ARCHITECTURES, CASES, REPEATS],
        names=["model", "architecture", "case", "repeat_id"],
    )
    actual = pd.MultiIndex.from_frame(
        scores[["model", "architecture", "case", "repeat_id"]]
    )
    duplicates = actual[actual.duplicated()].tolist()
    missing = expected.difference(actual).tolist()
    extras = actual.difference(expected).tolist()
    if duplicates or missing or extras or len(scores) != len(expected):
        raise RuntimeError(
            "Three-repeat completeness gate failed: "
            f"duplicates={duplicates[:5]}, missing={missing[:5]}, extras={extras[:5]}, "
            f"rows={len(scores)}/{len(expected)}"
        )


def _completion_audit(
    scores: pd.DataFrame,
    criteria: pd.DataFrame,
    contracts: pd.DataFrame,
    models: list[str],
) -> dict:
    expected_candidates = len(models) * len(ARCHITECTURES) * len(CASES) * len(REPEATS)
    expected_judgments = expected_candidates * 3
    expected_criterion_rows = expected_judgments * 14
    judgment_keys = criteria[["candidate_id", "judge"]].drop_duplicates()
    criterion_keys = criteria[["candidate_id", "judge", "criterion_id"]]
    audit = {
        "models": len(models),
        "architectures_per_model": len(ARCHITECTURES),
        "cases_per_architecture": len(CASES),
        "generation_repeats_per_cell": len(REPEATS),
        "expected_candidates": expected_candidates,
        "observed_candidates": len(scores),
        "expected_judgments": expected_judgments,
        "observed_judgments": len(judgment_keys),
        "expected_criterion_rows": expected_criterion_rows,
        "observed_criterion_rows": len(criteria),
        "duplicate_candidate_ids": int(scores["candidate_id"].duplicated().sum()),
        "duplicate_criterion_rows": int(criterion_keys.duplicated().sum()),
        "contract_summary_rows": len(contracts),
        "expected_contract_summary_rows": len(models) * len(ARCHITECTURES),
        "sample_sd_basis": "three repeat-level campaign means; ddof=1",
    }
    audit["complete"] = (
        audit["observed_candidates"] == expected_candidates
        and audit["observed_judgments"] == expected_judgments
        and audit["observed_criterion_rows"] == expected_criterion_rows
        and audit["duplicate_candidate_ids"] == 0
        and audit["duplicate_criterion_rows"] == 0
        and audit["contract_summary_rows"] == audit["expected_contract_summary_rows"]
    )
    if not audit["complete"]:
        raise RuntimeError(f"Cross-model completion audit failed: {audit}")
    return audit


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    path = output / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_tables(scores: pd.DataFrame, criteria: pd.DataFrame, contracts: pd.DataFrame, output: Path) -> dict:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    scores.to_csv(tables / "all_candidate_consensus_scores.csv", index=False)

    repeat_means = (
        scores.groupby(["model", "architecture", "repeat_id"], as_index=False)
        ["mean_score_0_1"].mean()
        .rename(columns={"mean_score_0_1": "campaign_mean_score"})
    )
    repeat_means.to_csv(tables / "campaign_repeat_means.csv", index=False)
    architecture = (
        repeat_means.groupby(["model", "architecture"])["campaign_mean_score"]
        .agg(mean="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    architecture["sem"] = architecture["sample_sd"] / np.sqrt(architecture["n_repeats"])
    architecture.to_csv(tables / "model_architecture_mean_sd.csv", index=False)

    wide = scores.pivot(
        index=["model", "case", "repeat_id"],
        columns="architecture",
        values="mean_score_0_1",
    ).reset_index()
    wide["paired_delta"] = wide["FlowPilot"] - wide["One-shot"]
    wide.to_csv(tables / "case_repeat_paired_deltas.csv", index=False)
    delta_repeat = (
        wide.groupby(["model", "repeat_id"], as_index=False)["paired_delta"].mean()
        .rename(columns={"paired_delta": "campaign_mean_paired_delta"})
    )
    delta_repeat.to_csv(tables / "paired_delta_repeat_means.csv", index=False)
    delta_summary = (
        delta_repeat.groupby("model")["campaign_mean_paired_delta"]
        .agg(mean_delta="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    delta_summary["sem"] = delta_summary["sample_sd"] / np.sqrt(delta_summary["n_repeats"])
    delta_summary["ci95_half_width_t_df2"] = 4.3026527 * delta_summary["sem"]
    delta_summary.to_csv(tables / "paired_delta_mean_sd.csv", index=False)

    case_summary = (
        scores.groupby(["model", "architecture", "case"])["mean_score_0_1"]
        .agg(mean="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    case_summary.to_csv(tables / "case_architecture_mean_sd.csv", index=False)

    applicable = criteria[criteria["applicability"] == "APPLICABLE"].copy()
    criterion_summary = (
        applicable.groupby(["model", "architecture", "criterion_id"])["score_0_4"]
        .mean().div(4).rename("mean_score_0_1").reset_index()
    )
    criterion_summary.to_csv(tables / "criterion_model_architecture_scores.csv", index=False)
    criteria.to_csv(tables / "all_criterion_judgments.csv", index=False)
    contracts.to_csv(tables / "outcome_contract_summary.csv", index=False)
    return {
        "repeat_means": repeat_means,
        "architecture": architecture,
        "wide": wide,
        "delta_summary": delta_summary,
        "case_summary": case_summary,
        "criterion_summary": criterion_summary,
    }


def _figures(data: dict, contracts: pd.DataFrame, output: Path, models: list[str]) -> None:
    architecture = data["architecture"]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset, arch in [(-0.18, "One-shot"), (0.18, "FlowPilot")]:
        rows = architecture[architecture["architecture"] == arch].set_index("model").loc[models]
        ax.errorbar(x + offset, rows["mean"], yerr=rows["sample_sd"], fmt="o", ms=8,
                    capsize=4, lw=1.8, color=COLORS[arch], label=arch)
    ax.set_xticks(x, models, rotation=25, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("Consensus score (mean ± sample SD)")
    ax.set_title("Matched three-repeat NewGen 2.0 comparison")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=2)
    _save(fig, output, "fig01_model_architecture_scores_with_sd")

    delta = data["delta_summary"].set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#16857A" if value >= 0 else "#C4473A" for value in delta["mean_delta"]]
    ax.barh(models, delta["mean_delta"], xerr=delta["sample_sd"], color=colors,
            alpha=.9, capsize=4)
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("FlowPilot − one-shot (mean ± sample SD)")
    ax.set_title("Paired architecture effect across three repeats")
    ax.grid(axis="x", alpha=.25)
    _save(fig, output, "fig02_paired_architecture_effect_with_sd")

    case = data["case_summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, case_name in zip(axes, CASES):
        rows = case[(case["case"] == case_name) & (case["architecture"] == "FlowPilot")]
        rows = rows.set_index("model").loc[models]
        ax.errorbar(range(len(models)), rows["mean"], yerr=rows["sample_sd"], fmt="o",
                    ms=7, capsize=3, color=COLORS["FlowPilot"])
        ax.set_title(case_name); ax.set_xticks(range(len(models)), models, rotation=65, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("FlowPilot score (mean ± sample SD)"); axes[0].set_ylim(0.75, 1.02)
    fig.suptitle("Case-by-case FlowPilot repeatability")
    _save(fig, output, "fig03_flowpilot_case_scores_with_sd")

    repeat = data["repeat_means"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        for model in models:
            rows = repeat[(repeat["model"] == model) & (repeat["architecture"] == arch)]
            rows = rows.set_index("repeat_id").loc[REPEATS]
            ax.plot([1, 2, 3], rows["campaign_mean_score"], marker="o", label=model)
        ax.set_title(arch); ax.set_xticks([1, 2, 3]); ax.set_xlabel("Generation repeat")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Campaign mean score"); axes[0].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Repeat-level campaign stability")
    _save(fig, output, "fig04_repeat_level_stability")

    criterion = data["criterion_summary"].pivot_table(
        index=["model", "criterion_id"], columns="architecture", values="mean_score_0_1"
    ).reset_index()
    criterion["delta"] = criterion["FlowPilot"] - criterion["One-shot"]
    heat = criterion.pivot(index="model", columns="criterion_id", values="delta").loc[models]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    image = ax.imshow(heat, aspect="auto", cmap="RdBu", vmin=-.5, vmax=.5)
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Criterion-level FlowPilot − one-shot effect")
    fig.colorbar(image, ax=ax, label="Normalized criterion-score difference")
    _save(fig, output, "fig05_criterion_effect_heatmap")

    contract = contracts.set_index(["model", "architecture"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        rows = contract.xs(arch, level="architecture").loc[models]
        ax.barh(models, rows["schema_valid"] / rows["n_outcomes"], color=COLORS[arch])
        ax.set_xlim(0, 1.02); ax.set_title(arch); ax.set_xlabel("Schema-valid fraction")
        ax.grid(axis="x", alpha=.25)
    fig.suptitle("Delivered-outcome schema validity")
    _save(fig, output, "fig06_schema_validity")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.1]},
        layout="constrained",
    )
    y = np.arange(len(models))
    for offset, arch in [(-0.12, "One-shot"), (0.12, "FlowPilot")]:
        rows = architecture[architecture["architecture"] == arch].set_index("model").loc[models]
        ax1.errorbar(
            rows["mean"],
            y + offset,
            xerr=rows["sample_sd"],
            fmt="o",
            ms=7,
            capsize=4,
            color=COLORS[arch],
            label=arch,
        )
    ax1.set_yticks(y, models)
    ax1.invert_yaxis()
    ax1.set_xlim(0.6, 1.0)
    ax1.set_xlabel("Consensus score (mean ± sample SD)")
    ax1.set_title("a  Outcome quality")
    ax1.grid(axis="x", alpha=.25)
    ax1.legend(frameon=False, ncol=2)
    delta_plot = data["delta_summary"].set_index("model").loc[models]
    bars = ax2.barh(
        y,
        delta_plot["mean_delta"],
        xerr=delta_plot["sample_sd"],
        color="#16857A",
        capsize=4,
    )
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("FlowPilot − one-shot")
    ax2.set_title("b  Matched architecture effect")
    ax2.grid(axis="x", alpha=.25)
    ax2.set_xlim(-0.03, 0.27)
    ax2.tick_params(axis="y", labelleft=False)
    for bar, value, sd in zip(bars, delta_plot["mean_delta"], delta_plot["sample_sd"]):
        ax2.text(
            value + sd + .005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            fontsize=9,
        )
    fig.suptitle("NewGen 2.0 matched three-repeat benchmark")
    _save(fig, output, "fig07_publication_summary")


def _write_report(output: Path, data: dict, models: list[str], excluded: list[str]) -> None:
    architecture = data["architecture"]
    delta = data["delta_summary"]
    lines = [
        "# NewGen 2.0: All-model three-repeat summary",
        "",
        f"This package compares the same three protocols, two architectures, three generation repeats, "
        f"and three blinded judge families for {len(models)} included models. No incomplete model is admitted.",
        "",
        "## Statistical unit",
        "",
        "For each model and architecture, each repeat first averages the three case scores. The reported "
        "mean and sample SD are then calculated across those three repeat-level campaign means (n=3, "
        "ddof=1). Judge disagreement is retained separately in the candidate table and is not substituted "
        "for generation-repeat variability. Error bars show sample SD, not confidence intervals.",
        "",
        "## Summary",
        "",
        "| Model | One-shot mean ± SD | FlowPilot mean ± SD | Paired delta mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        one = architecture[(architecture.model == model) & (architecture.architecture == "One-shot")].iloc[0]
        flow = architecture[(architecture.model == model) & (architecture.architecture == "FlowPilot")].iloc[0]
        drow = delta[delta.model == model].iloc[0]
        lines.append(
            f"| {model} | {one['mean']:.3f} ± {one['sample_sd']:.3f} | "
            f"{flow['mean']:.3f} ± {flow['sample_sd']:.3f} | "
            f"{drow['mean_delta']:+.3f} ± {drow['sample_sd']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These are outcome-quality and repeatability measurements for the frozen cases and inventory. "
        "They support matched architecture comparisons but do not establish universal superiority over "
        "all chemistry domains. With n=3, uncertainty remains substantial; the paired case-repeat table "
        "must accompany aggregate claims.",
    ])
    if excluded:
        lines.extend([
            "",
            "## Declared scope exclusion",
            "",
            "Excluded by user request from this publication package: " + ", ".join(excluded) + ". "
            "The complete archive remains available separately; no excluded result was used in these aggregates.",
        ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_publication_docs(output: Path, excluded: list[str]) -> None:
    methods = """# Benchmark methods

The benchmark used three frozen flow-chemistry cases: hydrogenolysis, photochemical oxidation, and CuAAC. Each generator model produced a general one-shot outcome and a FlowPilot outcome for each case in three independent generation repeats. The same protocol, objective, authoritative inventory, architecture budget, and repeat identifiers were used within each matched model comparison.

Every delivered outcome was blinded to generator identity and evaluated against the same 14-criterion NewGen 2.0 rubric by three judge families: Qwen, OpenAI, and Claude. Applicable criterion scores ranged from 0 to 4 and were normalized to 0-1 before averaging. NOT_APPLICABLE criteria were excluded from the denominator. Deterministic verification supplied numerical, inventory, topology, and schema checks to every judge.

For each model and architecture, case scores were first averaged within each generation repeat. The reported campaign mean and sample standard deviation were then calculated across the three repeat-level means (n=3; ddof=1). Paired architecture effects were calculated as FlowPilot minus one-shot for the same model, case, and repeat, then averaged by repeat. Judge disagreement is reported separately and is not treated as generation-repeat variability. Error bars represent sample standard deviation, not confidence intervals.

All candidate, criterion-level, repeat-level, and provenance records are included in this folder. The completeness audit must pass before figures are generated.
"""
    (output / "METHODS.md").write_text(methods, encoding="utf-8")
    captions = """# Figure captions

**Figure 1. Model and architecture outcome scores.** Consensus NewGen 2.0 scores for one-shot and FlowPilot outcomes. Points show the mean across three repeat-level campaign means; error bars show sample SD (n=3).

**Figure 2. Matched architecture effect.** Mean paired difference between FlowPilot and one-shot for each generator model. Positive values favor FlowPilot. Error bars show sample SD across three repeat-level paired means.

**Figure 3. FlowPilot case-level repeatability.** FlowPilot scores for each frozen chemistry case and model. Points and error bars show mean and sample SD across three generation repeats.

**Figure 4. Repeat-level stability.** Campaign means for each of the three generation repeats, separated by architecture. Lines connect repeats for visualization and do not imply temporal trends.

**Figure 5. Criterion-level architecture effects.** Difference between normalized FlowPilot and one-shot scores for each universal outcome criterion. Blue favors FlowPilot; red favors one-shot. UO-09 is absent because it was not applicable to the frozen single-stage cases.

**Figure 6. Delivered-outcome schema validity.** Fraction of the nine generated outcomes per model and architecture that satisfied the common delivered-outcome schema.

**Figure 7. Publication summary.** Panel a compares one-shot and FlowPilot consensus scores; panel b shows their matched difference. All error bars are sample SD across three generation repeats.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    exclusion = {
        "excluded_models": excluded,
        "reason": "Excluded from this publication package at the user's explicit request.",
        "timing": "Post-campaign packaging decision; generation and raw results were not deleted.",
        "full_archive": str(DEFAULT_OUTPUT),
    }
    (output / "EXCLUSION_RECORD.json").write_text(json.dumps(exclusion, indent=2) + "\n", encoding="utf-8")


def _manifest(output: Path) -> None:
    rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.sha256"):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}")
    (output / "ARTIFACT_MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-model", action="append", default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.exclude_model) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown excluded model(s): {unknown}")
    models = [model for model in MODEL_ORDER if model not in set(args.exclude_model)]
    if not models:
        raise ValueError("At least one model must remain in the publication package")

    score_frames, criterion_frames, contract_frames = [], [], []
    provenance = {}
    for model in models:
        score_frames.append(_read_table(model, "candidate_scores_with_repeats.csv"))
        criterion_frames.append(_read_table(model, "criterion_judgments.csv"))
        contract_frames.append(_read_table(model, "outcome_contract_summary.csv"))
        provenance[model] = str(SOURCES[model])
    scores = pd.concat(score_frames, ignore_index=True)
    criteria = pd.concat(criterion_frames, ignore_index=True)
    contracts = pd.concat(contract_frames, ignore_index=True)
    _validate_complete(scores, models)
    audit = _completion_audit(scores, criteria, contracts, models)
    data = _write_tables(scores, criteria, contracts, output)
    _figures(data, contracts, output, models)
    _write_report(output, data, models, args.exclude_model)
    _write_publication_docs(output, args.exclude_model)
    (output / "source_provenance.json").write_text(
        json.dumps({"sources": provenance, "models": models, "repeats": REPEATS}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "completion_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    _manifest(output)
    print(output)


if __name__ == "__main__":
    main()
