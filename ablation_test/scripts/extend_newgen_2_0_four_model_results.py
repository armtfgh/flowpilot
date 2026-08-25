#!/usr/bin/env python3
"""Add the completed Qwen3.8 exploratory campaign to the final NewGen package."""

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


MODELS = ("Qwen3.6-27B", "Qwen3.8-27B", "Claude Sonnet 4.6", "GPT-5.4")
ARCHITECTURES = ("One-shot", "FlowPilot")
CASES = ("Hydrogenolysis", "Photochemical oxidation", "CuAAC")
MODEL_COLORS = {
    "Qwen3.6-27B": "#4C78A8",
    "Qwen3.8-27B": "#59A14F",
    "Claude Sonnet 4.6": "#B279A2",
    "GPT-5.4": "#E59F44",
}
ARCH_COLORS = {"One-shot": "#8D969E", "FlowPilot": "#16838F"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def sd(values) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def merge_source_tables(package: Path, qwen38_deliverable: Path) -> dict[str, list[dict[str, str]]]:
    table_names = (
        "candidate_scores_with_repeats.csv",
        "criterion_judgments.csv",
        "judge_candidate_scores.csv",
        "repeat_level_paired_comparisons.csv",
        "model_architecture_summary.csv",
        "outcome_contract_summary.csv",
        "operational_summary.csv",
    )
    merged = {}
    target = package / "tables/four_model_extension"
    for name in table_names:
        rows = read_csv(package / "tables" / name)
        qrows = [
            row for row in read_csv(qwen38_deliverable / "tables" / name)
            if row.get("model") == "Qwen3.8-27B"
        ]
        rows.extend(qrows)
        merged[name] = rows
        write_csv(target / name, rows)
    return merged


def copy_qwen38_records(package: Path, qwen38_root: Path, candidate_rows: list[dict[str, str]]) -> None:
    for row in candidate_rows:
        if row["model"] != "Qwen3.8-27B":
            continue
        source_run = Path(row["run_directory"])
        relative = source_run.relative_to(qwen38_root / "generation")
        target_run = package / "raw/generation" / relative
        shutil.copytree(source_run, target_run, dirs_exist_ok=True)
        row["run_directory"] = str(Path("raw/generation") / relative)
        packet = qwen38_root / "packets" / f'{row["candidate_id"]}.json'
        target_packet = package / "raw/packets" / packet.name
        target_packet.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packet, target_packet)
        for judge in ("qwen", "openai", "claude"):
            shutil.copytree(
                qwen38_root / "judgments" / judge / row["candidate_id"],
                package / "raw/judgments" / judge / row["candidate_id"],
                dirs_exist_ok=True,
            )

    provenance = package / "provenance/source_campaigns/qwen38_exploratory"
    provenance.mkdir(parents=True, exist_ok=True)
    for source in (qwen38_root / "frozen").iterdir():
        if source.is_file():
            shutil.copy2(source, provenance / source.name)

    original = read_json(package / "frozen/candidate_key_confidential.json")["candidates"]
    qwen = read_json(qwen38_root / "frozen/candidate_key_confidential.json")["candidates"]
    for item in qwen:
        item = dict(item)
        source_run = Path(item["run_directory"])
        relative = source_run.relative_to(qwen38_root / "generation")
        item["source_campaign"] = qwen38_root.name
        item["source_run_directory"] = item["run_directory"]
        item["run_directory"] = str(Path("raw/generation") / relative)
        original.append(item)
    if len(original) != 60 or len({row["candidate_id"] for row in original}) != 60:
        raise RuntimeError("Four-model candidate provenance must contain 60 unique outcomes")
    write_json(package / "frozen/candidate_key_four_models_confidential.json", {"candidates": original})


def criterion_metadata(package: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    rubric = read_json(package / "frozen/outcome_rubric.json")
    rows = {row["criterion_id"]: row for row in rubric["criteria"]}
    return list(rows), rows


def derived_tables(package: Path, merged: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    target = package / "tables/four_model_extension"
    candidates = merged["candidate_scores_with_repeats.csv"]
    criteria = merged["criterion_judgments.csv"]
    paired = merged["repeat_level_paired_comparisons.csv"]
    contracts = merged["outcome_contract_summary.csv"]
    criterion_ids, metadata = criterion_metadata(package)

    case_rows = []
    for model in MODELS:
        for architecture in ARCHITECTURES:
            for case in CASES:
                values = [
                    float(row["mean_score_0_1"]) for row in candidates
                    if row["model"] == model and row["architecture"] == architecture and row["case"] == case
                ]
                case_rows.append({
                    "model": model, "architecture": architecture, "case": case,
                    "n_outcomes": len(values), "mean_score_0_1": round(mean(values), 6),
                    "sd": round(sd(values), 6),
                })
    write_csv(target / "case_architecture_summary.csv", case_rows)

    applicable = [row for row in criteria if row["applicability"] == "APPLICABLE" and row["score_0_4"] != ""]
    criterion_rows = []
    for model in MODELS:
        for architecture in ARCHITECTURES:
            for criterion_id in criterion_ids:
                selected = [
                    row for row in applicable
                    if row["model"] == model and row["architecture"] == architecture and row["criterion_id"] == criterion_id
                ]
                values = [float(row["score_0_4"]) / 4 for row in selected]
                criterion_rows.append({
                    "model": model, "architecture": architecture,
                    "criterion_id": criterion_id, "domain": metadata[criterion_id]["domain"],
                    "criterion_name": metadata[criterion_id]["name"],
                    "n_judge_ratings": len(values), "mean_score_0_1": round(mean(values), 6),
                    "critical_judge_flags": sum(truthy(row["critical_error"]) for row in selected),
                    "low_score_judge_events": sum(float(row["score_0_4"]) <= 1 for row in selected),
                })
    write_csv(target / "criterion_summary.csv", criterion_rows)

    units: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    candidate_info = {row["candidate_id"]: row for row in candidates}
    for row in applicable:
        units[(row["candidate_id"], row["criterion_id"])].append(row)
    unit_rows = []
    for (candidate_id, criterion_id), rows in units.items():
        info = candidate_info[candidate_id]
        values = [float(row["score_0_4"]) / 4 for row in rows]
        unit_rows.append({
            "candidate_id": candidate_id, "model": info["model"],
            "architecture": info["architecture"], "case": info["case"],
            "criterion_id": criterion_id, "criterion_name": metadata[criterion_id]["name"],
            "mean_score_0_1": round(mean(values), 6),
            "consensus_failure": mean(values) <= 0.25,
            "any_critical_judge_flag": any(truthy(row["critical_error"]) for row in rows),
        })
    write_csv(target / "criterion_error_units.csv", unit_rows)

    error_rows = []
    error_criterion_rows = []
    for model in MODELS:
        for architecture in ARCHITECTURES:
            outcome_ids = {
                row["candidate_id"] for row in candidates
                if row["model"] == model and row["architecture"] == architecture
            }
            selected = [row for row in unit_rows if row["candidate_id"] in outcome_ids]
            failure_count = sum(bool(row["consensus_failure"]) for row in selected)
            critical_count = sum(bool(row["any_critical_judge_flag"]) for row in selected)
            affected = {row["candidate_id"] for row in selected if row["any_critical_judge_flag"]}
            contract = next(row for row in contracts if row["model"] == model and row["architecture"] == architecture)
            n = len(outcome_ids)
            error_rows.append({
                "model": model, "architecture": architecture, "n_outcomes": n,
                "applicable_candidate_criteria": len(selected),
                "consensus_failure_units": failure_count,
                "consensus_failures_per_outcome": round(failure_count / n, 6),
                "critical_flagged_units": critical_count,
                "critical_flagged_outcomes": len(affected),
                "critical_flagged_outcome_rate": round(len(affected) / n, 6),
                "schema_invalid_outcomes": int(contract["n_outcomes"]) - int(contract["schema_valid"]),
            })
            for criterion_id in criterion_ids:
                by_criterion = [row for row in selected if row["criterion_id"] == criterion_id]
                assessed = len(by_criterion)
                error_criterion_rows.append({
                    "model": model, "architecture": architecture,
                    "criterion_id": criterion_id, "criterion_name": metadata[criterion_id]["name"],
                    "outcomes_assessed": assessed,
                    "consensus_failure_count": sum(bool(row["consensus_failure"]) for row in by_criterion),
                    "consensus_failure_rate": round(sum(bool(row["consensus_failure"]) for row in by_criterion) / assessed, 6) if assessed else "",
                    "critical_flagged_count": sum(bool(row["any_critical_judge_flag"]) for row in by_criterion),
                })
    write_csv(target / "error_summary.csv", error_rows)
    write_csv(target / "error_by_criterion.csv", error_criterion_rows)

    domain_rows = []
    domains = list(dict.fromkeys(metadata[cid]["domain"] for cid in criterion_ids))
    for model in MODELS:
        for architecture in ARCHITECTURES:
            for domain in domains:
                values = [
                    float(row["score_0_4"]) / 4 for row in applicable
                    if row["model"] == model and row["architecture"] == architecture
                    and metadata[row["criterion_id"]]["domain"] == domain
                ]
                domain_rows.append({
                    "model": model, "architecture": architecture, "domain": domain,
                    "n_judge_ratings": len(values), "mean_score_0_1": round(mean(values), 6),
                })
    write_csv(target / "domain_summary.csv", domain_rows)

    pair_case_rows = []
    for model in MODELS:
        for case in CASES:
            rows = [row for row in paired if row["model"] == model and row["case"] == case]
            values = [float(row["paired_delta"]) for row in rows]
            pair_case_rows.append({
                "model": model, "case": case, "n_pairs": len(values),
                "mean_paired_delta": round(mean(values), 6), "sd": round(sd(values), 6),
            })
    write_csv(target / "case_paired_effects.csv", pair_case_rows)
    return {
        "case_rows": case_rows, "criterion_rows": criterion_rows,
        "error_rows": error_rows, "error_criterion_rows": error_criterion_rows,
        "domain_rows": domain_rows, "pair_case_rows": pair_case_rows,
        "model_rows": merged["model_architecture_summary.csv"],
        "contracts": contracts, "candidate_rows": candidates,
        "judge_rows": merged["judge_candidate_scores.csv"],
        "criterion_ids": criterion_ids, "metadata": metadata,
    }


def heatmap(ax, matrix, xlabels, ylabels, title, *, vmin=0, vmax=1, cmap="RdYlGn", fmt=".2f"):
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title, fontweight="bold")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if not np.isnan(matrix[y, x]):
                ax.text(x, y, format(matrix[y, x], fmt), ha="center", va="center", fontsize=7)
    return image


def save(fig, figures: Path, name: str) -> None:
    fig.savefig(figures / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def figures(package: Path, data: dict[str, Any]) -> None:
    out = package / "figures/four_model_extension"
    out.mkdir(parents=True, exist_ok=True)
    cases = data["case_rows"]
    x = np.arange(len(CASES))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, model in enumerate(MODELS):
        values = [next(row["mean_score_0_1"] for row in cases if row["model"] == model and row["architecture"] == "FlowPilot" and row["case"] == case) for case in CASES]
        ax.bar(x + (index - 1.5) * width, values, width, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(x, CASES); ax.set_ylim(0, 1); ax.set_ylabel("NewGen 2.0 score")
    ax.set_title("FlowPilot score by protocol and generator model", fontweight="bold")
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig01_flowpilot_case_scores")

    model_rows = {row["model"]: row for row in data["model_rows"]}
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    one = [float(model_rows[m]["one_shot_mean_0_1"]) for m in MODELS]
    flow = [float(model_rows[m]["flowpilot_mean_0_1"]) for m in MODELS]
    b1 = ax.bar(np.arange(4)-.19, one, .38, color=ARCH_COLORS["One-shot"], label="One-shot")
    b2 = ax.bar(np.arange(4)+.19, flow, .38, color=ARCH_COLORS["FlowPilot"], label="FlowPilot")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8); ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(range(4), MODELS); ax.set_ylim(0, 1); ax.set_ylabel("NewGen 2.0 score")
    ax.set_title("Architecture benchmark across four generator models", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig02_four_model_architecture_scores")

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    means = np.array([float(model_rows[m]["mean_paired_delta"]) for m in MODELS])
    lows = np.array([float(model_rows[m]["paired_delta_95ci_low"]) for m in MODELS])
    highs = np.array([float(model_rows[m]["paired_delta_95ci_high"]) for m in MODELS])
    ax.errorbar(range(4), means, yerr=np.vstack((means-lows, highs-means)), fmt="o", color="#16838F", capsize=6, linewidth=2)
    ax.axhline(0, color="#333333", linewidth=1); ax.set_xticks(range(4), MODELS)
    ax.set_ylabel("FlowPilot minus one-shot"); ax.set_title("Matched architecture effect", fontweight="bold")
    ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig03_paired_architecture_effects")

    criterion_ids = data["criterion_ids"]
    labels = [f"{cid}\n{data['metadata'][cid]['name']}" for cid in criterion_ids]
    lookup = {(r["model"], r["architecture"], r["criterion_id"]): float(r["mean_score_0_1"]) for r in data["criterion_rows"]}
    matrix = np.array([[lookup.get((m, "FlowPilot", cid), np.nan) for cid in criterion_ids] for m in MODELS])
    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    im = heatmap(ax, matrix, labels, MODELS, "FlowPilot criterion scores")
    fig.colorbar(im, ax=ax, label="Mean criterion score")
    save(fig, out, "fig04_flowpilot_criterion_scores")

    delta = np.array([[lookup.get((m, "FlowPilot", cid), np.nan)-lookup.get((m, "One-shot", cid), np.nan) for cid in criterion_ids] for m in MODELS])
    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    im = heatmap(ax, delta, labels, MODELS, "Criterion-level architecture effect", vmin=-.5, vmax=.5, cmap="RdYlGn", fmt="+.2f")
    fig.colorbar(im, ax=ax, label="FlowPilot minus one-shot")
    save(fig, out, "fig05_criterion_architecture_effects")

    errors = {(r["model"], r["architecture"]): r for r in data["error_rows"]}
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, architecture in enumerate(ARCHITECTURES):
        values = [float(errors[(m, architecture)]["consensus_failures_per_outcome"]) for m in MODELS]
        ax.bar(np.arange(4)+(index-.5)*.38, values, .38, color=ARCH_COLORS[architecture], label=architecture)
    ax.set_xticks(range(4), MODELS); ax.set_ylabel("Consensus criterion failures per outcome")
    ax.set_title("Normalized substantive error burden", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig06_consensus_errors_per_outcome")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, architecture in enumerate(ARCHITECTURES):
        values = [100*float(errors[(m, architecture)]["critical_flagged_outcome_rate"]) for m in MODELS]
        ax.bar(np.arange(4)+(index-.5)*.38, values, .38, color=ARCH_COLORS[architecture], label=architecture)
    ax.set_xticks(range(4), MODELS); ax.set_ylim(0, 100); ax.set_ylabel("Outcomes with any critical judge flag (%)")
    ax.set_title("Critical-error incidence", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig07_critical_error_incidence")

    error_lookup = {(r["model"], r["architecture"], r["criterion_id"]): (float(r["consensus_failure_rate"]) if r["consensus_failure_rate"] != "" else np.nan) for r in data["error_criterion_rows"]}
    rows = [f"{model} | {arch}" for model in MODELS for arch in ARCHITECTURES]
    matrix = np.array([[error_lookup.get((model, arch, cid), np.nan) for cid in criterion_ids] for model in MODELS for arch in ARCHITECTURES])
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    im = heatmap(ax, matrix, criterion_ids, rows, "Consensus failure rate by criterion", cmap="Reds")
    fig.colorbar(im, ax=ax, label="Fraction of assessed outcomes")
    save(fig, out, "fig08_error_rate_by_criterion")

    pair_lookup = {(r["model"], r["case"]): float(r["mean_paired_delta"]) for r in data["pair_case_rows"]}
    matrix = np.array([[pair_lookup[(m, c)] for c in CASES] for m in MODELS])
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    im = heatmap(ax, matrix, CASES, MODELS, "Case-level FlowPilot gain", vmin=-.1, vmax=.35, cmap="RdYlGn", fmt="+.3f")
    fig.colorbar(im, ax=ax, label="FlowPilot minus one-shot")
    save(fig, out, "fig09_case_level_architecture_gain")

    contracts = {(r["model"], r["architecture"]): r for r in data["contracts"]}
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, architecture in enumerate(ARCHITECTURES):
        values = [100*int(contracts[(m, architecture)]["schema_valid"])/int(contracts[(m, architecture)]["n_outcomes"]) for m in MODELS]
        ax.bar(np.arange(4)+(index-.5)*.38, values, .38, color=ARCH_COLORS[architecture], label=architecture)
    ax.set_xticks(range(4), MODELS); ax.set_ylim(0, 105); ax.set_ylabel("Schema-valid outcomes (%)")
    ax.set_title("Deterministic structural validity", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig10_schema_validity")

    judge_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in data["judge_rows"]:
        judge_groups[(row["model"], row["architecture"], row["judge"])].append(float(row["score_0_1"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, constrained_layout=True)
    for ax, architecture in zip(axes, ARCHITECTURES):
        for judge, marker in (("qwen", "o"), ("openai", "s"), ("claude", "^")):
            values = [mean(judge_groups[(m, architecture, judge)]) for m in MODELS]
            ax.plot(range(4), values, marker=marker, linewidth=1.7, label=judge.title())
        ax.set_xticks(range(4), MODELS, rotation=20, ha="right"); ax.set_ylim(0, 1)
        ax.set_title(architecture, fontweight="bold"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean judge score"); axes[1].legend(frameon=False)
    fig.suptitle("Judge-specific model scores", fontweight="bold")
    save(fig, out, "fig11_judge_specific_scores")

    domains = list(dict.fromkeys(row["domain"] for row in data["domain_rows"]))
    domain_lookup = {(r["model"], r["architecture"], r["domain"]): float(r["mean_score_0_1"]) for r in data["domain_rows"]}
    matrix = np.array([[domain_lookup[(m, "FlowPilot", d)] for d in domains] for m in MODELS])
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    im = heatmap(ax, matrix, domains, MODELS, "FlowPilot scores by evaluation domain")
    fig.colorbar(im, ax=ax, label="Mean domain score")
    save(fig, out, "fig12_flowpilot_domain_scores")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, architecture in enumerate(ARCHITECTURES):
        values = [float(errors[(m, architecture)]["critical_flagged_units"]) / float(errors[(m, architecture)]["n_outcomes"]) for m in MODELS]
        bars = ax.bar(np.arange(4)+(index-.5)*.38, values, .38, color=ARCH_COLORS[architecture], label=architecture)
        totals = [int(errors[(m, architecture)]["critical_flagged_units"]) for m in MODELS]
        ax.bar_label(bars, labels=[f"{total} total" for total in totals], padding=3, fontsize=8)
    ax.set_xticks(range(4), MODELS); ax.set_ylabel("Critical-flagged criterion units per outcome")
    ax.set_title("Normalized critical-error count", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig13_critical_error_units_per_outcome")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, architecture in enumerate(ARCHITECTURES):
        values = [int(errors[(m, architecture)]["schema_invalid_outcomes"]) for m in MODELS]
        bars = ax.bar(np.arange(4)+(index-.5)*.38, values, .38, color=ARCH_COLORS[architecture], label=architecture)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(range(4), MODELS); ax.set_ylabel("Schema-invalid outcomes")
    ax.set_title("Absolute deterministic format errors", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig14_schema_invalid_counts")


def documentation(package: Path, data: dict[str, Any]) -> dict[str, Any]:
    model_rows = {row["model"]: row for row in data["model_rows"]}
    summary_rows = []
    for model in MODELS:
        row = model_rows[model]
        summary_rows.append({
            "model": model,
            "n_matched_pairs": int(row["n_matched_pairs"]),
            "one_shot_mean_0_1": float(row["one_shot_mean_0_1"]),
            "flowpilot_mean_0_1": float(row["flowpilot_mean_0_1"]),
            "mean_paired_delta": float(row["mean_paired_delta"]),
            "paired_delta_95ci_low": float(row["paired_delta_95ci_low"]),
            "paired_delta_95ci_high": float(row["paired_delta_95ci_high"]),
            "wins": int(row["wins"]), "ties": int(row["ties"]), "losses": int(row["losses"]),
        })
    summary = {
        "schema_version": "flowpilot_newgen_2_0_four_model_extension_v1.0",
        "status": "three-model confirmatory plus one-repeat Qwen3.8 exploratory extension",
        "candidate_count": len(data["candidate_rows"]),
        "judgment_count": len(data["judge_rows"]),
        "models": summary_rows,
        "repeat_policy": {
            "Qwen3.6-27B": 3, "Claude Sonnet 4.6": 3, "GPT-5.4": 3, "Qwen3.8-27B": 1,
        },
        "error_definition": "A consensus failure is an applicable candidate-criterion unit with mean judge score <=1/4. Critical incidence is reported separately as any judge critical flag.",
    }
    write_json(package / "summary_four_models.json", summary)
    candidate_ids = [row["candidate_id"] for row in data["candidate_rows"]]
    judge_counts: dict[str, int] = defaultdict(int)
    for row in data["judge_rows"]:
        judge_counts[row["candidate_id"]] += 1
    criterion_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in read_csv(package / "tables/four_model_extension/criterion_judgments.csv"):
        criterion_counts[(row["candidate_id"], row["judge"])] += 1
    validation = {
        "candidate_count_is_60": len(candidate_ids) == 60,
        "candidate_ids_unique": len(set(candidate_ids)) == 60,
        "three_judges_per_candidate": all(judge_counts[candidate_id] == 3 for candidate_id in candidate_ids),
        "fourteen_criteria_per_candidate_judge": all(value == 14 for value in criterion_counts.values()),
        "criterion_unit_count": len(criterion_counts),
        "raw_result_present_for_every_candidate": all((package / row["run_directory"] / "result.json").is_file() for row in data["candidate_rows"]),
        "qwen38_outcome_count": sum(row["model"] == "Qwen3.8-27B" for row in data["candidate_rows"]),
        "rubric_sha256": hashlib.sha256((package / "frozen/outcome_rubric.json").read_bytes()).hexdigest(),
    }
    validation["all_checks_pass"] = all(
        value is True for key, value in validation.items()
        if key not in {"criterion_unit_count", "qwen38_outcome_count", "rubric_sha256"}
    ) and validation["qwen38_outcome_count"] == 6
    write_json(package / "FOUR_MODEL_VALIDATION.json", validation)
    table = "\n".join(
        f"| {r['model']} | {r['n_matched_pairs']} | {r['one_shot_mean_0_1']:.3f} | {r['flowpilot_mean_0_1']:.3f} | {r['mean_paired_delta']:+.3f} | {r['paired_delta_95ci_low']:+.3f} to {r['paired_delta_95ci_high']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} |"
        for r in summary_rows
    )
    (package / "FOUR_MODEL_EXTENSION_REPORT.md").write_text(
        "# NewGen 2.0 Four-Model Extension\n\n"
        "This extension adds the completed Qwen3.8-27B campaign to the existing final-results folder. "
        "Qwen3.8 used the same three protocols, inventories, frozen 14-criterion rubric, and three-judge panel. "
        "It has one repeat per case, whereas the other model rows have three; it is therefore exploratory and "
        "must not be interpreted as equally replicated.\n\n"
        "| Generator | Pairs | One-shot | FlowPilot | Delta | 95% interval | W/T/L |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" + table + "\n\n"
        "## Figure set\n\n"
        "1. `fig01_flowpilot_case_scores`: only FlowPilot, resolved by protocol and model.\n"
        "2. `fig02_four_model_architecture_scores`: one-shot versus FlowPilot for all four models.\n"
        "3. `fig03_paired_architecture_effects`: matched deltas and 95% intervals.\n"
        "4. `fig04_flowpilot_criterion_scores`: FlowPilot's 14 universal criterion scores.\n"
        "5. `fig05_criterion_architecture_effects`: criterion-level FlowPilot minus one-shot effects.\n"
        "6. `fig06_consensus_errors_per_outcome`: strict consensus failures normalized by outcomes.\n"
        "7. `fig07_critical_error_incidence`: outcomes receiving any critical judge flag.\n"
        "8. `fig08_error_rate_by_criterion`: criterion-localized consensus failures.\n"
        "9. `fig09_case_level_architecture_gain`: matched effect by protocol.\n"
        "10. `fig10_schema_validity`: deterministic structural-validity rate.\n"
        "11. `fig11_judge_specific_scores`: Qwen, OpenAI, and Claude score profiles.\n"
        "12. `fig12_flowpilot_domain_scores`: chemistry, process, engineering, inventory, safety, operations, and evidence.\n"
        "13. `fig13_critical_error_units_per_outcome`: normalized critical-flag count with absolute totals.\n"
        "14. `fig14_schema_invalid_counts`: absolute deterministic format errors.\n\n"
        "Every figure is supplied as 300 dpi PNG and vector PDF.\n\n"
        "## Main observations\n\n"
        "- Qwen3.8 scored 0.742 in one-shot and 0.924 in FlowPilot (paired delta +0.182; 3/3 wins). "
        "Its 95% interval crosses zero because only three pairs are available.\n"
        "- Qwen3.8 FlowPilot scores were 0.878 for hydrogenolysis, 0.942 for photochemical oxidation, "
        "and 0.951 for CuAAC.\n"
        "- Qwen3.8 one-shot had two strict consensus-failure units and one of three outcomes with a "
        "critical judge flag; Qwen3.8 FlowPilot had neither in this exploratory round.\n"
        "- Physical and transport plausibility (UO-11) was the lowest Qwen3.8 FlowPilot criterion "
        "at 0.806, so model portability does not eliminate engineering uncertainty.\n\n"
        "## Error interpretation\n\n"
        "A consensus criterion failure is an applicable candidate-by-criterion unit whose mean score across "
        "the three judges is at most 1 on the 0-4 rubric. This avoids counting three judge ratings as three "
        "independent design errors. Critical-flag incidence and schema invalidity are reported separately. "
        "Raw counts should not be compared without normalization because Qwen3.8 has fewer outcomes.\n",
        encoding="utf-8",
    )
    (package / "tables/four_model_extension/ERROR_DEFINITIONS.md").write_text(
        "# Error Definitions\n\n"
        "- `consensus_failure_unit`: applicable candidate-criterion mean <= 1/4 across available judges.\n"
        "- `critical_flagged_unit`: at least one judge marked that candidate-criterion critical.\n"
        "- `critical_flagged_outcome`: at least one critical-flagged unit in an outcome.\n"
        "- `schema_invalid_outcome`: deterministic output-schema validation failed.\n"
        "- Primary cross-model error comparison: consensus failures per outcome, which normalizes unequal repeats.\n",
        encoding="utf-8",
    )
    return summary


def manifest(package: Path) -> None:
    destination = package / "PACKAGE_MANIFEST.sha256"
    rows = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.resolve() != destination.resolve():
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(package)}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--qwen38-root", type=Path, required=True)
    parser.add_argument("--qwen38-deliverable", type=Path, required=True)
    args = parser.parse_args()
    package, qroot, qdeliver = (args.package.resolve(), args.qwen38_root.resolve(), args.qwen38_deliverable.resolve())
    rubric_hashes = {
        hashlib.sha256((package / "frozen/outcome_rubric.json").read_bytes()).hexdigest(),
        hashlib.sha256((qroot / "frozen/outcome_rubric.json").read_bytes()).hexdigest(),
    }
    if len(rubric_hashes) != 1:
        raise RuntimeError("Rubric hashes do not match")
    merged = merge_source_tables(package, qdeliver)
    copy_qwen38_records(package, qroot, merged["candidate_scores_with_repeats.csv"])
    write_csv(
        package / "tables/four_model_extension/candidate_scores_with_repeats.csv",
        merged["candidate_scores_with_repeats.csv"],
    )
    data = derived_tables(package, merged)
    figures(package, data)
    summary = documentation(package, data)
    manifest(package)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
