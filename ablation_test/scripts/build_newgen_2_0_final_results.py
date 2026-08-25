#!/usr/bin/env python3
"""Build the final NewGen 2.0 composite results package.

Claude and Qwen records come from the frozen three-model campaign. GPT records
come from the separately frozen OpenAI confirmatory campaign. The package keeps
this provenance explicit and never substitutes individual favorable outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("Qwen3.6-27B", "Claude Sonnet 4.6", "GPT-5.4")
ARCHITECTURES = ("One-shot", "FlowPilot")
SOURCE_MODELS = {
    "base": {"Qwen3.6-27B", "Claude Sonnet 4.6"},
    "gpt": {"GPT-5.4"},
}
COLORS = {"One-shot": "#9aa1a8", "FlowPilot": "#16838f"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(path: Path, models: set[str]) -> list[dict[str, str]]:
    return [row for row in read_csv(path) if row.get("model") in models]


def model_sort(row: dict[str, Any]) -> tuple[int, str, str, str]:
    model = str(row.get("model", ""))
    architecture = str(row.get("architecture", ""))
    return (
        MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
        architecture,
        str(row.get("case", "")),
        str(row.get("repeat_id", "")),
    )


def agreement_from_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["applicability"] == "APPLICABLE":
            units[(row["candidate_id"], row["criterion_id"])][row["judge"]] = float(
                row["score_0_4"]
            )

    exact = 0
    within_one = 0
    comparable = 0
    absolute_differences: list[float] = []
    for scores in units.values():
        if len(scores) != 3:
            continue
        values = list(scores.values())
        comparable += 1
        exact += len(set(values)) == 1
        within_one += max(values) - min(values) <= 1
        absolute_differences.extend(
            abs(values[left] - values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )
    return {
        "candidate_criterion_units": comparable,
        "exact_agreement_rate": round(exact / comparable, 6),
        "within_one_point_rate": round(within_one / comparable, 6),
        "mean_pairwise_absolute_difference_0_4": round(
            statistics.fmean(absolute_differences), 6
        ),
    }


def package_manifest(root: Path) -> None:
    destination = root / "PACKAGE_MANIFEST.sha256"
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == destination.resolve():
            continue
        entries.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        )
    destination.write_text("\n".join(entries) + "\n", encoding="utf-8")


def copy_raw_records(
    output: Path,
    source_roots: dict[str, Path],
    candidate_rows: list[dict[str, str]],
) -> dict[str, str]:
    raw = output / "raw"
    candidate_sources: dict[str, str] = {}
    run_paths: dict[str, str] = {}

    for row in candidate_rows:
        source_key = "gpt" if row["model"] == "GPT-5.4" else "base"
        source_root = source_roots[source_key]
        source_run = Path(row["run_directory"])
        relative_run = source_run.relative_to(source_root / "generation")
        target_run = raw / "generation" / relative_run
        shutil.copytree(source_run, target_run, dirs_exist_ok=True)
        run_paths[row["candidate_id"]] = str(target_run.relative_to(output))
        candidate_sources[row["candidate_id"]] = source_key

        packet = source_root / "packets" / f'{row["candidate_id"]}.json'
        target_packet = raw / "packets" / packet.name
        target_packet.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packet, target_packet)
        for judge in ("claude", "openai", "qwen"):
            source_judgment = source_root / "judgments" / judge / row["candidate_id"]
            target_judgment = raw / "judgments" / judge / row["candidate_id"]
            shutil.copytree(source_judgment, target_judgment, dirs_exist_ok=True)

    return run_paths


def merge_tables(
    output: Path,
    base_deliverable: Path,
    gpt_deliverable: Path,
    source_roots: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    tables = output / "tables"
    common_csv = sorted(
        {path.name for path in (base_deliverable / "tables").glob("*.csv")}
        & {path.name for path in (gpt_deliverable / "tables").glob("*.csv")}
    )

    merged_by_table: dict[str, list[dict[str, str]]] = {}
    for name in common_csv:
        rows = selected_rows(base_deliverable / "tables" / name, SOURCE_MODELS["base"])
        rows.extend(selected_rows(gpt_deliverable / "tables" / name, SOURCE_MODELS["gpt"]))
        merged_by_table[name] = sorted(rows, key=model_sort)

    candidate_rows = merged_by_table["candidate_scores_with_repeats.csv"]
    if len(candidate_rows) != 54 or len({row["candidate_id"] for row in candidate_rows}) != 54:
        raise RuntimeError("Expected 54 unique selected candidate records")

    run_paths = copy_raw_records(output, source_roots, candidate_rows)
    for row in candidate_rows:
        row["run_directory"] = run_paths[row["candidate_id"]]

    for name, rows in merged_by_table.items():
        write_csv(tables / name, rows)

    criterion_rows = merged_by_table["criterion_judgments.csv"]
    judge_rows = merged_by_table["judge_candidate_scores.csv"]
    agreement = agreement_from_rows(criterion_rows)
    write_json(tables / "judge_agreement.json", agreement)

    model_rows = [
        {
            "model": row["model"],
            "n_matched_pairs": int(row["n_matched_pairs"]),
            "one_shot_mean_0_1": float(row["one_shot_mean_0_1"]),
            "flowpilot_mean_0_1": float(row["flowpilot_mean_0_1"]),
            "mean_paired_delta": float(row["mean_paired_delta"]),
            "paired_delta_sd": float(row["paired_delta_sd"]),
            "paired_delta_95ci_low": float(row["paired_delta_95ci_low"]),
            "paired_delta_95ci_high": float(row["paired_delta_95ci_high"]),
            "wins": int(row["wins"]),
            "ties": int(row["ties"]),
            "losses": int(row["losses"]),
            "generator_family_excluded_delta": float(
                row["generator_family_excluded_delta"]
            ),
        }
        for row in merged_by_table["model_architecture_summary.csv"]
    ]
    paired_rows = merged_by_table["repeat_level_paired_comparisons.csv"]
    contract_rows = [
        {
            "model": row["model"],
            "architecture": row["architecture"],
            **{
                field: int(row[field])
                for field in (
                    "n_outcomes",
                    "schema_valid",
                    "executable",
                    "blocked",
                    "assessable_one_shot",
                    "missing",
                )
            },
        }
        for row in merged_by_table["outcome_contract_summary.csv"]
    ]
    summary = {
        "schema_version": "flowpilot_newgen_2_0_final_composite_v1.0",
        "package_status": "final_composite",
        "candidate_count": len(candidate_rows),
        "judgment_count": len(judge_rows),
        "criterion_judgment_count": len(criterion_rows),
        "case_count": len({row["case"] for row in candidate_rows}),
        "repeat_count": len({row["repeat_id"] for row in candidate_rows}),
        "generator_count": len(MODEL_ORDER),
        "judge_count": 3,
        "model_architecture_summary": model_rows,
        "overall_mean_paired_delta": round(
            statistics.fmean(float(row["paired_delta"]) for row in paired_rows), 6
        ),
        "overall_generator_family_excluded_delta": round(
            statistics.fmean(float(row["excluded_paired_delta"]) for row in paired_rows),
            6,
        ),
        "judge_agreement": agreement,
        "outcome_contract_summary": contract_rows,
        "provenance_policy": {
            "Qwen3.6-27B": "manuscript_three_model_three_repeat_postfix_20260820",
            "Claude Sonnet 4.6": "manuscript_three_model_three_repeat_postfix_20260820",
            "GPT-5.4": "openai_confirmatory_stationary_fix_20260820",
        },
    }
    write_json(output / "summary.json", summary)
    return summary, candidate_rows


def copy_frozen_provenance(
    output: Path, source_roots: dict[str, Path], candidate_rows: list[dict[str, str]]
) -> None:
    frozen = output / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    rubric_hashes = {
        hashlib.sha256((root / "frozen" / "outcome_rubric.json").read_bytes()).hexdigest()
        for root in source_roots.values()
    }
    if len(rubric_hashes) != 1:
        raise RuntimeError("Source campaigns do not use the same frozen rubric")
    shutil.copy2(source_roots["base"] / "frozen" / "outcome_rubric.json", frozen)

    selected_ids = {row["candidate_id"] for row in candidate_rows}
    candidates = []
    for source_key, root in source_roots.items():
        source_candidates = read_json(root / "frozen" / "candidate_key_confidential.json")[
            "candidates"
        ]
        for candidate in source_candidates:
            if (
                candidate["candidate_id"] not in selected_ids
                or candidate["generator_model"] not in SOURCE_MODELS[source_key]
            ):
                continue
            item = dict(candidate)
            item["source_campaign"] = root.name
            item["source_run_directory"] = item["run_directory"]
            source_run = Path(item["run_directory"])
            relative_run = source_run.relative_to(root / "generation")
            item["run_directory"] = str(Path("raw/generation") / relative_run)
            candidates.append(item)
    if len(candidates) != 54:
        raise RuntimeError("Candidate provenance did not close to 54 records")
    write_json(frozen / "candidate_key_confidential.json", {"candidates": candidates})

    provenance = output / "provenance" / "source_campaigns"
    for source_key, root in source_roots.items():
        target = provenance / source_key
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "campaign_manifest.json",
            "case_manifest.json",
            "execution_plan.json",
            "PREREGISTRATION.md",
            "source_code_checksums.json",
        ):
            source = root / "frozen" / name
            if source.exists():
                shutil.copy2(source, target / name)


def create_figures(output: Path, summary: dict[str, Any]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    model_rows = {row["model"]: row for row in summary["model_architecture_summary"]}
    x = np.arange(len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    one = [float(model_rows[model]["one_shot_mean_0_1"]) for model in MODEL_ORDER]
    flow = [float(model_rows[model]["flowpilot_mean_0_1"]) for model in MODEL_ORDER]
    bars_a = ax.bar(x - 0.19, one, 0.38, color=COLORS["One-shot"], label="One-shot")
    bars_b = ax.bar(x + 0.19, flow, 0.38, color=COLORS["FlowPilot"], label="FlowPilot")
    ax.bar_label(bars_a, labels=[f"{value:.3f}" for value in one], padding=3, fontsize=9)
    ax.bar_label(bars_b, labels=[f"{value:.3f}" for value in flow], padding=3, fontsize=9)
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylim(0.70, 0.98)
    ax.set_ylabel("Blinded NewGen 2.0 score (0-1)")
    ax.set_title("Final NewGen 2.0 architecture comparison", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "fig01_final_architecture_scores.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    means = [float(model_rows[model]["mean_paired_delta"]) for model in MODEL_ORDER]
    lows = [float(model_rows[model]["paired_delta_95ci_low"]) for model in MODEL_ORDER]
    highs = [float(model_rows[model]["paired_delta_95ci_high"]) for model in MODEL_ORDER]
    errors = np.array([[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]])
    ax.errorbar(x, means, yerr=errors, fmt="o", color="#16838f", capsize=6, linewidth=2)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylabel("Paired delta: FlowPilot minus one-shot")
    ax.set_title("Matched architecture effect with 95% intervals", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    for index, value in enumerate(means):
        ax.annotate(f"{value:+.3f}", (index, value), xytext=(8, 8), textcoords="offset points")
    fig.savefig(figures / "fig02_final_paired_effects.png", dpi=300, facecolor="white")
    plt.close(fig)

    contracts = read_csv(output / "tables" / "outcome_contract_summary.csv")
    contract_map = {(row["model"], row["architecture"]): row for row in contracts}
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    one_schema = [int(contract_map[(model, "One-shot")]["schema_valid"]) for model in MODEL_ORDER]
    flow_schema = [int(contract_map[(model, "FlowPilot")]["schema_valid"]) for model in MODEL_ORDER]
    bars_a = ax.bar(x - 0.19, one_schema, 0.38, color=COLORS["One-shot"], label="One-shot")
    bars_b = ax.bar(x + 0.19, flow_schema, 0.38, color=COLORS["FlowPilot"], label="FlowPilot")
    ax.bar_label(bars_a, labels=[f"{value}/9" for value in one_schema], padding=3)
    ax.bar_label(bars_b, labels=[f"{value}/9" for value in flow_schema], padding=3)
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Outcomes satisfying required schema")
    ax.set_title("Deterministic structural validity", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "fig03_final_schema_validity.png", dpi=300, facecolor="white")
    plt.close(fig)

    criterion_rows = read_csv(output / "tables" / "criterion_architecture_effect.csv")
    criteria = sorted({row["criterion_id"] for row in criterion_rows})
    values = np.array(
        [
            [
                next(
                    (float(row["delta"]) for row in criterion_rows if row["model"] == model and row["criterion_id"] == criterion),
                    math.nan,
                )
                for criterion in criteria
            ]
            for model in MODEL_ORDER
        ]
    )
    fig, ax = plt.subplots(figsize=(12.5, 4.7), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu", vmin=-0.25, vmax=0.25, aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            if not math.isnan(values[row_index, column_index]):
                ax.text(column_index, row_index, f"{values[row_index, column_index]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(criteria)), criteria, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Criterion-level architecture effect", fontweight="bold")
    fig.colorbar(image, ax=ax, label="FlowPilot minus one-shot")
    fig.savefig(figures / "fig04_final_criterion_effects.png", dpi=300, facecolor="white")
    plt.close(fig)


def create_report(output: Path, summary: dict[str, Any]) -> None:
    rows = {row["model"]: row for row in summary["model_architecture_summary"]}
    lines = []
    for model in MODEL_ORDER:
        row = rows[model]
        lines.append(
            f'| {model} | {float(row["one_shot_mean_0_1"]):.3f} | '
            f'{float(row["flowpilot_mean_0_1"]):.3f} | '
            f'{float(row["mean_paired_delta"]):+.3f} | '
            f'{float(row["paired_delta_95ci_low"]):+.3f} to '
            f'{float(row["paired_delta_95ci_high"]):+.3f} | '
            f'{row["wins"]}/{row["ties"]}/{row["losses"]} |'
        )
    report = f"""# NewGen 2.0 Final Results

## Composition rule

This is a provenance-preserving composite package, not a claim that all three model families were generated in one simultaneous campaign.

- Qwen3.6-27B and Claude Sonnet 4.6 are retained unchanged from `manuscript_three_model_three_repeat_postfix_20260820`.
- GPT-5.4 is replaced in full by `openai_confirmatory_stationary_fix_20260820`.
- No individual outcome was selected according to its score.
- Both source campaigns used the identical frozen NewGen 2.0 rubric.

## Final results

| Generator | One-shot | FlowPilot | Paired delta | 95% interval | W/T/L |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Across all 27 matched case-repeat pairs, the mean architecture effect is **{float(summary["overall_mean_paired_delta"]):+.3f}**. The generator-family-excluded sensitivity effect is **{float(summary["overall_generator_family_excluded_delta"]):+.3f}**.

## Deterministic closure

- FlowPilot produced 27/27 schema-valid outcomes and 27/27 executable final contracts.
- One-shot schema validity was 9/9 for Qwen, 2/9 for Claude, and 7/9 for corrected-campaign GPT.
- The final package contains 54 generated outcomes and 162 selected final judgments. Every failed or retried raw attempt associated with those outcomes is also retained.

## Interpretation

Qwen shows the largest and most consistent architecture gain. Claude also shows a positive interval in this three-case sample. Corrected-campaign GPT shows a modest positive mean, but its interval crosses zero. These data support architecture benefit and stronger deterministic closure; they do not establish universal superiority across flow chemistry.

## Files

- `summary.json`: final numerical summary and source policy.
- `tables/`: merged candidate, criterion, pair, contract, operational, and repeatability data.
- `figures/`: presentation-ready final comparisons.
- `raw/`: selected raw generations, blinded packets, judge prompts/responses, retries, and telemetry.
- `frozen/`: common rubric and merged candidate key.
- `provenance/source_campaigns/`: original frozen campaign manifests and preregistrations.
- `supplementary_regression/`: post-score CuAAC residence-time policy regression; excluded from benchmark scores.
- `PACKAGE_MANIFEST.sha256`: checksum for every packaged artifact.
"""
    (output / "FINAL_RESULTS_REPORT.md").write_text(report, encoding="utf-8")


def build(
    output: Path,
    base_raw: Path,
    base_deliverable: Path,
    gpt_raw: Path,
    gpt_deliverable: Path,
    regression: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    source_roots = {"base": base_raw, "gpt": gpt_raw}
    summary, candidates = merge_tables(
        output, base_deliverable, gpt_deliverable, source_roots
    )
    copy_frozen_provenance(output, source_roots, candidates)
    shutil.copytree(
        regression,
        output / "supplementary_regression" / regression.name,
        dirs_exist_ok=True,
    )
    create_figures(output, summary)
    create_report(output, summary)
    package_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("base_raw", type=Path)
    parser.add_argument("base_deliverable", type=Path)
    parser.add_argument("gpt_raw", type=Path)
    parser.add_argument("gpt_deliverable", type=Path)
    parser.add_argument("regression", type=Path)
    args = parser.parse_args()
    build(
        args.output.resolve(),
        args.base_raw.resolve(),
        args.base_deliverable.resolve(),
        args.gpt_raw.resolve(),
        args.gpt_deliverable.resolve(),
        args.regression.resolve(),
    )


if __name__ == "__main__":
    main()
