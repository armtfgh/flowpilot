#!/usr/bin/env python3
"""Aggregate NewGen 2.0 judgments and create publication/presentation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.newgen_2_outcome import read_json, validate_response, write_json

DEFAULT_INPUT = ROOT / "ablation_results/newgen_benchmark/newgen_2_0_20260818"
DEFAULT_OUTPUT = ROOT / "deliverables/flowpilot_newgen_2_0_20260818"
JUDGE_FAMILIES = {"qwen": "qwen", "openai": "openai", "claude": "anthropic"}


def resolved_response(root: Path, judge: str, cid: str, rubric: dict, packet: dict) -> dict[str, Any] | None:
    call = root / "judgments" / judge / cid
    pairs = [(call / "status.json", call / "parsed_response.json")]
    pairs += [(path, path.with_name("parsed_response.json")) for path in (call / "attempts").glob("attempt_*/status.json")]
    valid = []
    for status, parsed in pairs:
        if status.is_file() and parsed.is_file() and read_json(status).get("status") == "valid":
            response = read_json(parsed)
            if not validate_response(response, rubric, packet):
                valid.append((status.stat().st_mtime_ns, response))
    return max(valid, default=(0, None), key=lambda item: item[0])[1]


def mean(values):
    return statistics.fmean(values) if values else math.nan


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(root: Path, output: Path) -> dict[str, Any]:
    rubric = read_json(root / "frozen/outcome_rubric.json")
    campaign = read_json(root / "frozen/campaign_manifest.json") if (root / "frozen/campaign_manifest.json").is_file() else {}
    selected_judges = campaign.get("selected_judges") or list(JUDGE_FAMILIES)
    key = read_json(root / "frozen/candidate_key_confidential.json")["candidates"]
    key_by_id = {row["candidate_id"]: row for row in key}
    criterion_rows, judgment_rows = [], []
    for judge in selected_judges:
        for candidate in key:
            cid = candidate["candidate_id"]
            packet = read_json(root / "packets" / f"{cid}.json")
            response = resolved_response(root, judge, cid, rubric, packet)
            if response is None:
                continue
            applicable = [row for row in response["criterion_scores"] if row["applicability"] == "APPLICABLE"]
            score = mean([row["score"] for row in applicable]) / 4
            judgment_rows.append({
                "candidate_id": cid, "judge": judge, "judge_family": JUDGE_FAMILIES[judge],
                "model": candidate["generator_model"], "generator_family": candidate["generator_family"],
                "architecture": candidate["architecture"], "case": candidate["case"],
                "score_0_1": round(score, 6), "applicable_criteria": len(applicable),
                "critical_errors": sum(bool(row["critical_error"]) for row in applicable),
                "overall_executability": response["overall_executability"],
            })
            for row in response["criterion_scores"]:
                criterion_rows.append({
                    "candidate_id": cid, "judge": judge, "judge_family": JUDGE_FAMILIES[judge],
                    "model": candidate["generator_model"], "generator_family": candidate["generator_family"],
                    "architecture": candidate["architecture"], "case": candidate["case"],
                    "criterion_id": row["criterion_id"], "applicability": row["applicability"],
                    "score_0_4": row["score"], "critical_error": row["critical_error"],
                    "severity": row["severity"], "confidence": row["confidence"],
                    "rationale": row["rationale"], "evidence_paths": " | ".join(row["evidence_paths"]),
                    "observed_values": " | ".join(row["observed_values"]),
                    "expected_or_correct": row["expected_or_correct"], "required_correction": row["required_correction"],
                })
    expected_judgments = len(key) * len(selected_judges)
    if len(judgment_rows) != expected_judgments:
        raise RuntimeError(f"Campaign incomplete: {len(judgment_rows)}/{expected_judgments} valid judgments")

    candidate_rows = []
    for cid in key_by_id:
        rows = [row for row in judgment_rows if row["candidate_id"] == cid]
        meta = key_by_id[cid]
        all_scores = [row["score_0_1"] for row in rows]
        excluded = [row["score_0_1"] for row in rows if row["judge_family"] != meta["generator_family"]]
        crit_by_judge = [row["critical_errors"] for row in rows]
        candidate_rows.append({
            "candidate_id": cid, "model": meta["generator_model"], "generator_family": meta["generator_family"],
            "architecture": meta["architecture"], "case": meta["case"],
            "mean_score_0_1": round(mean(all_scores), 6), "median_score_0_1": round(statistics.median(all_scores), 6),
            "judge_sd": round(sd(all_scores), 6),
            "generator_family_excluded_score_0_1": round(mean(excluded), 6),
            "any_judge_critical_errors": sum(value > 0 for value in crit_by_judge),
            "total_judge_critical_flags": sum(crit_by_judge),
        })

    architecture_rows = []
    groups = defaultdict(list)
    for row in candidate_rows:
        groups[(row["model"], row["architecture"])].append(row)
    for (model, architecture), rows in sorted(groups.items()):
        values = [row["mean_score_0_1"] for row in rows]
        sensitivity = [row["generator_family_excluded_score_0_1"] for row in rows]
        architecture_rows.append({
            "model": model, "architecture": architecture, "n_cases": len(rows),
            "mean_score_0_1": round(mean(values), 6), "median_score_0_1": round(statistics.median(values), 6),
            "between_case_sd": round(sd(values), 6),
            "generator_family_excluded_mean_0_1": round(mean(sensitivity), 6),
            "critical_flags": sum(row["total_judge_critical_flags"] for row in rows),
        })

    paired_rows = []
    for model in sorted({row["model"] for row in candidate_rows}):
        for case in sorted({row["case"] for row in candidate_rows if row["model"] == model}):
            cells = {row["architecture"]: row for row in candidate_rows if row["model"] == model and row["case"] == case}
            if {"FlowPilot", "One-shot"} <= set(cells):
                paired_rows.append({
                    "model": model, "case": case,
                    "one_shot_score_0_1": cells["One-shot"]["mean_score_0_1"],
                    "flowpilot_score_0_1": cells["FlowPilot"]["mean_score_0_1"],
                    "paired_delta": round(cells["FlowPilot"]["mean_score_0_1"] - cells["One-shot"]["mean_score_0_1"], 6),
                    "one_shot_excluded_score": cells["One-shot"]["generator_family_excluded_score_0_1"],
                    "flowpilot_excluded_score": cells["FlowPilot"]["generator_family_excluded_score_0_1"],
                    "excluded_paired_delta": round(cells["FlowPilot"]["generator_family_excluded_score_0_1"] - cells["One-shot"]["generator_family_excluded_score_0_1"], 6),
                })

    # Agreement is measured on the same candidate-criterion units.
    units = defaultdict(dict)
    for row in criterion_rows:
        if row["applicability"] == "APPLICABLE":
            units[(row["candidate_id"], row["criterion_id"])][row["judge"]] = row["score_0_4"]
    exact, within_one, abs_diffs, comparable = 0, 0, [], 0
    for scores in units.values():
        if len(scores) != len(selected_judges):
            continue
        values = list(scores.values()); comparable += 1
        exact += len(set(values)) == 1
        within_one += max(values) - min(values) <= 1
        abs_diffs.extend(
            abs(values[i] - values[j])
            for i in range(len(values)) for j in range(i + 1, len(values))
        )
    agreement = {
        "candidate_criterion_units": comparable,
        "exact_agreement_rate": round(exact / comparable, 6),
        "within_one_point_rate": round(within_one / comparable, 6),
        "mean_pairwise_absolute_difference_0_4": round(mean(abs_diffs), 6),
    }

    tables = output / "tables"
    write_csv(tables / "criterion_judgments.csv", criterion_rows)
    write_csv(tables / "judge_candidate_scores.csv", judgment_rows)
    write_csv(tables / "candidate_consensus_scores.csv", candidate_rows)
    write_csv(tables / "architecture_summary.csv", architecture_rows)
    write_csv(tables / "paired_comparisons.csv", paired_rows)
    write_json(tables / "judge_agreement.json", agreement)
    summary = {
        "schema_version": "flowpilot_newgen_2_0_summary_v1.0",
        "candidate_count": len(candidate_rows), "judge_count": len(selected_judges),
        "selected_judges": selected_judges, "judgment_count": len(judgment_rows),
        "criterion_judgment_count": len(criterion_rows), "architecture_summary": architecture_rows,
        "paired_comparisons": paired_rows, "judge_agreement": agreement,
        "mean_paired_delta": round(mean([row["paired_delta"] for row in paired_rows]), 6),
        "mean_generator_excluded_paired_delta": round(mean([row["excluded_paired_delta"] for row in paired_rows]), 6),
    }
    write_json(output / "summary.json", summary)
    return summary


def figures(output: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    tables, figdir = output / "tables", output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(tables / "candidate_consensus_scores.csv")
    criteria = pd.read_csv(tables / "criterion_judgments.csv")
    paired = pd.read_csv(tables / "paired_comparisons.csv")
    colors = {"One-shot": "#9AA0A6", "FlowPilot": "#167D8D"}

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    summary = candidates.groupby(["model", "architecture"])["mean_score_0_1"].mean().unstack()
    summary[["One-shot", "FlowPilot"]].plot.bar(ax=ax, color=[colors["One-shot"], colors["FlowPilot"]], width=.72)
    ax.set_ylim(0, 1); ax.set_ylabel("Consensus outcome score (0-1)"); ax.set_xlabel("")
    ax.set_title("NewGen 2.0: identical rubric across architectures"); ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(figdir / "fig01_architecture_score.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    model_colors = {"GPT-4o": "#4C78A8", "Qwen3.6-27B": "#E45756"}
    case_markers = {"CuAAC": "o", "Hydrogenolysis": "s", "Two-stage amidation": "^"}
    for _, row in paired.reset_index(drop=True).iterrows():
        ax.plot(
            [0, 1], [row.one_shot_score_0_1, row.flowpilot_score_0_1],
            marker=case_markers.get(row["case"], "o"),
            color=model_colors.get(row["model"], "#167D8D"), alpha=.85,
            label=f"{row['model']} | {row['case']}",
        )
    ax.set_xlim(-.15, 1.15); ax.set_ylim(0, 1); ax.set_xticks([0,1], ["One-shot", "FlowPilot"])
    ax.set_ylabel("Consensus outcome score (0-1)"); ax.set_title("Matched candidate pairs")
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(figdir / "fig02_paired_candidates.png", dpi=300); plt.close(fig)

    applicable = criteria[criteria.applicability == "APPLICABLE"].copy()
    heat = applicable.groupby(["architecture", "criterion_id"])["score_0_4"].mean().unstack().reindex(["One-shot", "FlowPilot"])
    fig, ax = plt.subplots(figsize=(11.5, 3.2)); image = ax.imshow(heat.values / 4, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right"); ax.set_yticks(range(2), heat.index)
    for y in range(2):
        for x in range(len(heat.columns)):
            ax.text(x, y, f"{heat.iloc[y,x]/4:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Mean normalized criterion score"); ax.set_title("Universal criterion profile")
    fig.tight_layout(); fig.savefig(figdir / "fig03_criterion_heatmap.png", dpi=300); plt.close(fig)

    judge_scores = pd.read_csv(tables / "judge_candidate_scores.csv")
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    judges = list(judge_scores.judge.drop_duplicates())
    labels = {"qwen": "Qwen", "openai": "OpenAI", "claude": "Claude"}
    data = [judge_scores[judge_scores.judge == judge].score_0_1.values for judge in judges]
    ax.boxplot(data, tick_labels=[labels.get(judge, judge) for judge in judges], patch_artist=True, boxprops={"facecolor":"#D7E8EA"})
    ax.set_ylim(0,1); ax.set_ylabel("Candidate score (0-1)"); ax.set_title("Judge severity and dispersion")
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(figdir / "fig04_judge_dispersion.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sensitivity = candidates.groupby("architecture")[["mean_score_0_1", "generator_family_excluded_score_0_1"]].mean().reindex(["One-shot", "FlowPilot"])
    sensitivity.plot.bar(ax=ax, color=["#2E5964", "#D28E2D"])
    ax.set_ylim(0,1); ax.set_ylabel("Mean score (0-1)"); ax.set_xlabel(""); ax.set_title("Generator-family exclusion sensitivity")
    panel_size = judge_scores.judge.nunique()
    full_label = "Both judges" if panel_size == 2 else f"All {panel_size} judges"
    sensitivity_label = "Cross-family judge only" if panel_size == 2 else "Exclude same-family judge"
    ax.legend([full_label, sensitivity_label], frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figdir / "fig05_family_bias_sensitivity.png", dpi=300); plt.close(fig)


def documentation(root: Path, output: Path, summary: dict[str, Any]) -> None:
    rows = summary["architecture_summary"]
    import csv
    judge_rows = list(csv.DictReader((output / "tables/judge_candidate_scores.csv").open(encoding="utf-8")))
    judge_means = {
        judge: mean([float(row["score_0_1"]) for row in judge_rows if row["judge"] == judge])
        for judge in summary["selected_judges"]
    }
    judge_mean_text = "; ".join(f"{judge} {value:.3f}" for judge, value in judge_means.items())
    result_lines = [f"- {r['model']} / {r['architecture']}: {r['mean_score_0_1']:.3f} (family-excluded {r['generator_family_excluded_mean_0_1']:.3f})" for r in rows]
    case_count = len({row["case"] for row in summary["paired_comparisons"]})
    candidate_count = summary["candidate_count"]
    agreement_label = (
        "high" if summary["judge_agreement"]["within_one_point_rate"] >= 0.85
        else "moderate" if summary["judge_agreement"]["within_one_point_rate"] >= 0.70
        else "low"
    )
    text = f"""# NewGen 2.0 Benchmark

## Purpose

NewGen 2.0 compares final batch-to-flow design outcomes using one frozen, universal rubric. It does not award points for councils, validators, agent count, logs, or architecture-specific fields.

## Design

- {candidate_count} matched candidates: {case_count} held-out chemistries x 2 generator families x 2 architectures.
- Independent judge panel: {', '.join(summary['selected_judges'])} ({summary['judge_count']} judges).
- Generator and architecture labels are withheld. Perfect architecture blinding is not claimed because output style may reveal provenance.
- Every judge sees the same protocol, objective, inventory, held-out source facts, normalized final outcome, deterministic verification sheet, and rubric.
- The reference flow procedure is evidence, not the only acceptable solution.

## Score

Each applicable criterion receives an anchored integer from 0 to 4. All criteria have equal weight. For candidate *c* and criterion *k*:

`criterion_consensus(c,k) = mean(score from every selected judge)`

`candidate_score(c) = mean(all applicable criterion_consensus) / 4`

There are no hand-tuned 0.1/0.2 weights and no architecture bonus. Only gas bookkeeping (UO-08) and multistage closure (UO-09) may be not applicable. Missing required information is scored 0 or 1.

## Criteria

UO-01 transformation fidelity; UO-02 required materials; UO-03 stoichiometry/feed chemistry; UO-04 condition/stage mapping; UO-05 executable topology; UO-06 liquid material balance; UO-07 residence-time/geometry closure; UO-08 gas bookkeeping; UO-09 multistage closure; UO-10 inventory feasibility; UO-11 transport plausibility; UO-12 hazards/controls; UO-13 operating procedure/work-up; UO-14 evidence/uncertainty calibration.

## Results

{chr(10).join(result_lines)}

Mean matched FlowPilot-minus-one-shot delta: **{summary['mean_paired_delta']:+.3f}**. Generator-family-excluded sensitivity delta: **{summary['mean_generator_excluded_paired_delta']:+.3f}**.

Judge agreement: exact {summary['judge_agreement']['exact_agreement_rate']:.1%}; within one point {summary['judge_agreement']['within_one_point_rate']:.1%}; mean pairwise absolute difference {summary['judge_agreement']['mean_pairwise_absolute_difference_0_4']:.3f}/4.

Judge mean candidate scores: {judge_mean_text}. This calibration difference is why both consensus and judge-specific tables are retained.

## Interpretation Limits

This is a small matched benchmark, not a universal performance claim. LLM judges can share biases and are not substitutes for wet-lab validation. Critical-error flags are reported separately and never used to manipulate or cap the numerical score. With two judges, the generator-family-excluded result is a single cross-family judgment, not an independent consensus; it is a directional sensitivity check only. Inter-judge agreement is {agreement_label} by the predeclared within-one-point diagnostic. The result should be described as preliminary until repeated generation and stronger independent evaluation are added.
"""
    (output / "NEWGEN_2_0_REPORT.md").write_text(text, encoding="utf-8")
    (output / "SCORING_METHOD.md").write_text(
        "# Scoring Method\n\nThe complete frozen machine-readable rubric is included in `frozen/outcome_rubric.json`. "
        "Scores are integer anchored ratings (0-4), averaged across judges and applicable criteria, then divided by four. "
        "No criterion-specific weights are used.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=DEFAULT_INPUT); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); root, output = args.input.resolve(), args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(root, output); figures(output); documentation(root, output, summary)
    frozen = output / "frozen"; frozen.mkdir(exist_ok=True)
    for name in ("outcome_rubric.json", "campaign_manifest.json", "packet_metrics.json"):
        source = root / "frozen" / name
        if source.is_file():
            (frozen / name).write_bytes(source.read_bytes())
    protocol = ROOT / "ablation_test/benchmarks/NEWGEN_2_0_PROTOCOL.md"
    if protocol.is_file():
        (output / "NEWGEN_2_0_PROTOCOL.md").write_bytes(protocol.read_bytes())
    print(output)


if __name__ == "__main__":
    main()
