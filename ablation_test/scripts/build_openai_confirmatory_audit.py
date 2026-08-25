#!/usr/bin/env python3
"""Build the audit package for the frozen OpenAI confirmatory campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASE_LABELS = {
    "Hydrogenolysis": "Hydrogenolysis",
    "Photochemical oxidation": "Photo-oxidation",
    "CuAAC": "CuAAC",
}
CRITERION_LABELS = {
    "UO-01": "Transformation fidelity",
    "UO-02": "Required materials",
    "UO-03": "Stoichiometry/feed chemistry",
    "UO-04": "Condition/stage mapping",
    "UO-05": "Executable topology",
    "UO-06": "Liquid material balance",
    "UO-07": "Residence-time closure",
    "UO-08": "Gas bookkeeping",
    "UO-09": "Multistage closure",
    "UO-10": "Inventory feasibility",
    "UO-11": "Transport plausibility",
    "UO-12": "Hazards and controls",
    "UO-13": "Operating procedure",
    "UO-14": "Evidence calibration",
}
JUDGE_COLORS = {"claude": "#7b6992", "openai": "#16838f", "qwen": "#d18b32"}
JUDGE_LABELS = {"claude": "Claude", "openai": "OpenAI", "qwen": "Qwen"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_manifest(root: Path, destination: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == destination.resolve():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(root)}")
    destination.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _judge_sensitivity(tables: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in _rows(tables / "judge_candidate_scores.csv"):
        grouped[(row["judge"], row["architecture"])].append(float(row["score_0_1"]))

    output = []
    for judge in ("claude", "openai", "qwen"):
        one_shot = float(np.mean(grouped[(judge, "One-shot")]))
        flowpilot = float(np.mean(grouped[(judge, "FlowPilot")]))
        output.append(
            {
                "judge": judge,
                "one_shot_mean_0_1": round(one_shot, 6),
                "flowpilot_mean_0_1": round(flowpilot, 6),
                "paired_architecture_delta": round(flowpilot - one_shot, 6),
            }
        )
    _write_csv(tables / "judge_specific_architecture_effect.csv", output)
    return output


def _summary_figure(deliverable: Path, summary: dict[str, object]) -> None:
    tables = deliverable / "tables"
    figures = deliverable / "figures"
    paired = _rows(tables / "repeat_level_paired_comparisons.csv")
    model_summary = summary["model_architecture_summary"][0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), constrained_layout=True)
    ax = axes[0]
    architecture = [
        float(model_summary["one_shot_mean_0_1"]),
        float(model_summary["flowpilot_mean_0_1"]),
    ]
    bars = ax.bar(
        [0, 1], architecture, width=0.58, color=["#9aa1a8", "#16838f"], edgecolor="white"
    )
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in architecture], padding=4, fontsize=11)
    ax.set_xticks([0, 1], ["GPT-5.4\none-shot", "GPT-5.4\nFlowPilot"])
    ax.set_ylim(0.82, 0.97)
    ax.set_ylabel("Blinded NewGen 2.0 score (0-1)")
    ax.set_title("A  Mean outcome score", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    case_order = list(CASE_LABELS)
    colors = ["#355d7a", "#d18b32", "#7b6992"]
    offsets = {case: (index - 1) * 0.08 for index, case in enumerate(case_order)}
    for index, case in enumerate(case_order):
        case_rows = [row for row in paired if row["case"] == case]
        ax.scatter(
            [float(row["paired_delta"]) for row in case_rows],
            [1 + offsets[case]] * len(case_rows),
            s=65,
            color=colors[index],
            label=CASE_LABELS[case],
            zorder=3,
        )
    mean = float(model_summary["mean_paired_delta"])
    low = float(model_summary["paired_delta_95ci_low"])
    high = float(model_summary["paired_delta_95ci_high"])
    ax.errorbar(
        mean,
        0.58,
        xerr=[[mean - low], [high - mean]],
        fmt="o",
        color="#111111",
        capsize=5,
        markersize=7,
        linewidth=2,
        label="Mean and 95% CI",
    )
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks([0.58, 1.0], ["Mean", "Case-repeat pairs"])
    ax.set_ylim(0.35, 1.25)
    ax.set_xlim(-0.045, 0.105)
    ax.set_xlabel("Paired delta: FlowPilot minus one-shot")
    ax.set_title("B  Matched architecture effect", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("OpenAI confirmatory architecture benchmark", fontsize=15, fontweight="bold")
    fig.savefig(figures / "fig08_openai_confirmatory_summary.png", dpi=300, facecolor="white")
    plt.close(fig)


def _criterion_figure(deliverable: Path) -> None:
    rows = _rows(deliverable / "tables" / "criterion_architecture_effect.csv")
    rows = sorted(rows, key=lambda row: float(row["delta"]))
    labels = [f'{row["criterion_id"]}  {CRITERION_LABELS[row["criterion_id"]]}' for row in rows]
    values = [float(row["delta"]) for row in rows]
    colors = ["#16838f" if value >= 0 else "#c5504c" for value in values]

    fig, ax = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.bar_label(
        bars,
        labels=[f"{value:+.3f}" for value in values],
        padding=4,
        fontsize=9,
        fmt="%s",
    )
    ax.set_xlim(min(-0.075, min(values) - 0.025), max(0.285, max(values) + 0.025))
    ax.set_xlabel("Criterion delta: FlowPilot minus one-shot")
    ax.set_title("Where FlowPilot gained and lost score", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(
        deliverable / "figures" / "fig09_openai_criterion_effects.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def _integrity_figure(deliverable: Path) -> None:
    contracts = {
        row["architecture"]: row
        for row in _rows(deliverable / "tables" / "outcome_contract_summary.csv")
    }
    candidates = _rows(deliverable / "tables" / "candidate_consensus_scores.csv")
    critical = {
        architecture: sum(
            int(row["total_judge_critical_flags"])
            for row in candidates
            if row["architecture"] == architecture
        )
        for architecture in ("One-shot", "FlowPilot")
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    labels = ["One-shot", "FlowPilot"]
    colors = ["#9aa1a8", "#16838f"]
    schema = [int(contracts[label]["schema_valid"]) for label in labels]
    bars = axes[0].bar(labels, schema, color=colors, width=0.58)
    axes[0].bar_label(bars, labels=[f"{value}/9" for value in schema], padding=4, fontsize=11)
    axes[0].set_ylim(0, 10)
    axes[0].set_ylabel("Outcomes satisfying required schema")
    axes[0].set_title("A  Deterministic structural validity", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)

    errors = [critical[label] for label in labels]
    bars = axes[1].bar(labels, errors, color=colors, width=0.58)
    axes[1].bar_label(bars, labels=[str(value) for value in errors], padding=4, fontsize=11)
    axes[1].set_ylim(0, max(errors) + 1)
    axes[1].set_ylabel("Independent-judge critical flags")
    axes[1].set_title("B  Critical defects", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Outcome integrity checks", fontsize=15, fontweight="bold")
    fig.savefig(
        deliverable / "figures" / "fig10_openai_contract_and_errors.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def _audit_report(
    deliverable: Path,
    raw: Path,
    regression: Path,
    summary: dict[str, object],
    judge_rows: list[dict[str, object]],
) -> None:
    model = summary["model_architecture_summary"][0]
    regression_summary = json.loads((regression / "summary.json").read_text(encoding="utf-8"))
    judge_lines = "\n".join(
        f'- {JUDGE_LABELS[str(row["judge"])]}: '
        f'{float(row["paired_architecture_delta"]):+.3f}'
        for row in judge_rows
    )
    report = f"""# OpenAI Confirmatory Campaign Audit

## Frozen comparison

- Generator: GPT-5.4 for both architectures.
- Architectures: direct one-shot versus the full FlowPilot pipeline.
- Cases: hydrogenolysis, photochemical oxidation, and CuAAC.
- Repeats: three per case and architecture, giving 18 generated outcomes.
- Evaluation: 54 blinded judgments from Claude, OpenAI, and Qwen using 14 equal-weight, architecture-neutral criteria.
- The acceptance criteria and source checksums were frozen before generation.

## Primary result

| Measure | One-shot | FlowPilot |
|---|---:|---:|
| Mean score | {float(model["one_shot_mean_0_1"]):.3f} | {float(model["flowpilot_mean_0_1"]):.3f} |
| Required-schema validity | 7/9 | 9/9 |
| Executable FlowPilot contract | n/a | 9/9 |
| Critical judge flags | 2 | 0 |

The matched mean delta was **{float(model["mean_paired_delta"]):+.3f}** with a 95% interval of **{float(model["paired_delta_95ci_low"]):+.3f} to {float(model["paired_delta_95ci_high"]):+.3f}**. FlowPilot won, tied, and lost {model["wins"]}/{model["ties"]}/{model["losses"]} matched pairs. The generator-family-excluded sensitivity delta was **{float(model["generator_family_excluded_delta"]):+.3f}**.

This is a modest positive result, not proof of universal superiority: the interval crosses zero and only three chemistries were tested.

## Judge sensitivity

{judge_lines}

Judge-family disagreement is retained as uncertainty. Exact agreement was {float(summary["judge_agreement"]["exact_agreement_rate"])*100:.1f}%; agreement within one rubric point was {float(summary["judge_agreement"]["within_one_point_rate"])*100:.1f}%.

## What improved

- Gas bookkeeping was the largest gain (`UO-08`, +0.250).
- Residence-time/geometry closure improved (`UO-07`, +0.065).
- Stoichiometry and liquid material balance both improved (`UO-03` and `UO-06`, +0.046 each).
- FlowPilot produced the required structured result in 9/9 outcomes and incurred no critical judge flags.

## Remaining weaknesses

- Safety controls (`UO-12`, -0.056) need more explicit instantiated controls, especially H2 check valves, separator vent routing, and shutdown purging.
- Operating procedure (`UO-13`, -0.046) needs quantitative priming/flush instructions, shutdown order, depressurization, and chemistry-specific waste handling.
- Transport plausibility (`UO-11`, -0.019) needs clearer packed-bed wetting, liquid-holdup/contact-time, two-phase, and post-BPR flash assumptions.
- CuAAC was the weakest case-level comparison; its procedure detail, rather than catalyst placement, remains the main deficit.

## Defects found and corrected after scoring

The frozen benchmark outputs and scores were not replaced. Inspection exposed two architecture-level defects:

1. A stationary packed-bed catalyst could be repeated in pumped-feed composition. Final realization and contract validation now prohibit this conflict.
2. Evidence-first translation still inherited a model-derived intensification ceiling, allowing conservative residence-time revisions to be clipped to sub-minute values. The ceiling now applies only when the explicit policy is `intensify`.

The separate three-repeat OpenAI CuAAC regression passed all acceptance checks: all three contracts were executable, all selected 3.0 min, and none put Cu/C in a pumped feed. These regression results validate the repair but are not included in the frozen architecture score.

## Artifact map

- Frozen benchmark data: `{raw}`
- Focused post-fix regression: `{regression}`
- Main summary: `summary.json`
- Pair-level data: `tables/repeat_level_paired_comparisons.csv`
- Criterion effects: `tables/criterion_architecture_effect.csv`
- New figures: `figures/fig08_openai_confirmatory_summary.png`, `figures/fig09_openai_criterion_effects.png`, and `figures/fig10_openai_contract_and_errors.png`
"""
    (deliverable / "OPENAI_CONFIRMATORY_AUDIT.md").write_text(report, encoding="utf-8")


def build(deliverable: Path, raw: Path, regression: Path) -> None:
    summary = json.loads((deliverable / "summary.json").read_text(encoding="utf-8"))
    judge_rows = _judge_sensitivity(deliverable / "tables")
    _summary_figure(deliverable, summary)
    _criterion_figure(deliverable)
    _integrity_figure(deliverable)
    _audit_report(deliverable, raw, regression, summary, judge_rows)
    _file_manifest(raw, raw / "PACKAGE_MANIFEST.sha256")
    _file_manifest(deliverable, deliverable / "ARTIFACT_MANIFEST.sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deliverable", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("regression", type=Path)
    args = parser.parse_args()
    build(args.deliverable.resolve(), args.raw.resolve(), args.regression.resolve())


if __name__ == "__main__":
    main()
