#!/usr/bin/env python3
"""Aggregate NewGen 2.0 module-attribution outcomes and render figures."""

from __future__ import annotations

import csv
import hashlib
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

from ablation_test.scripts.build_newgen_2_0_report import aggregate
from ablation_test.scripts.audit_qwen38_module_execution import audit as audit_execution


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return statistics.fmean(values) if values else math.nan


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def deterministic_rows(root: Path) -> list[dict[str, Any]]:
    key = read_json(root / "frozen/candidate_key_confidential.json")["candidates"]
    rows = []
    for item in key:
        run = Path(item["run_directory"])
        metrics = read_json(run / "metrics.json")
        result = read_json(run / "result.json")
        run_summary = read_json(run / "run_summary.json")
        tokens = run_summary.get("token_totals") or {}
        final = result.get("final_design") or {}
        execution = item.get("execution_config")
        if execution is None and (run / "snapshots/council_execution_config.json").is_file():
            execution = read_json(run / "snapshots/council_execution_config.json")
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "case": item["case"],
                "condition": item["architecture"],
                "condition_id": item["condition_id"],
                "repeat_id": item["repeat_id"],
                "candidate_budget": item.get("candidate_budget"),
                "runtime_s": run_summary.get("runtime_s"),
                "llm_call_count": run_summary.get("llm_call_count"),
                "prompt_tokens": tokens.get("prompt_tokens", tokens.get("input_tokens", 0)),
                "completion_tokens": tokens.get("completion_tokens", tokens.get("output_tokens", 0)),
                "deterministic_score": metrics.get("deterministic_composite_score"),
                "schema_valid": bool(metrics.get("schema_valid")),
                "deployment_gate_count": int(metrics.get("deployment_gate_count_v2") or 0),
                "executable": final.get("status") == "executable",
                "blocked": final.get("status") == "blocked",
                "residence_time_min": (result.get("proposal") or {}).get("residence_time_min"),
                "flow_rate_mL_min": (result.get("proposal") or {}).get("flow_rate_mL_min"),
                "reactor_volume_mL": (result.get("proposal") or {}).get("reactor_volume_mL"),
                "execution_config": json.dumps(execution, sort_keys=True) if execution else "",
            }
        )
    return rows


def build_report(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    execution_audit = audit_execution(root, output)
    summary = aggregate(root, output)
    consensus = csv_rows(output / "tables/candidate_consensus_scores.csv")
    criterion = csv_rows(output / "tables/criterion_judgments.csv")
    deterministic = deterministic_rows(root)
    write_csv(output / "tables/deterministic_outcomes.csv", deterministic)
    det_by_id = {row["candidate_id"]: row for row in deterministic}
    consensus_by_id = {row["candidate_id"]: row for row in consensus}

    outcome_rows = []
    for cid, det in det_by_id.items():
        judged = consensus_by_id[cid]
        outcome_rows.append(
            {
                **det,
                "consensus_score_0_1": float(judged["mean_score_0_1"]),
                "family_excluded_score_0_1": float(
                    judged["generator_family_excluded_score_0_1"]
                ),
                "critical_flags": int(judged["total_judge_critical_flags"]),
            }
        )
    write_csv(output / "tables/module_outcomes.csv", outcome_rows)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcome_rows:
        groups[row["condition"]].append(row)
    condition_rows = []
    for condition, rows in groups.items():
        scores = [row["consensus_score_0_1"] for row in rows]
        deterministic_scores = [
            float(row["deterministic_score"])
            for row in rows
            if isinstance(row["deterministic_score"], (int, float))
        ]
        condition_rows.append(
            {
                "condition": condition,
                "n_outcomes": len(rows),
                "mean_score_0_1": round(mean(scores), 6),
                "score_sd": round(sd(scores), 6),
                "mean_deterministic_score": round(mean(deterministic_scores), 6),
                "executable_rate": round(sum(row["executable"] for row in rows) / len(rows), 6),
                "schema_valid_rate": round(sum(row["schema_valid"] for row in rows) / len(rows), 6),
                "mean_gate_count": round(mean([row["deployment_gate_count"] for row in rows]), 6),
                "critical_flags": sum(row["critical_flags"] for row in rows),
                "mean_runtime_s": round(mean([float(row["runtime_s"] or 0) for row in rows]), 3),
                "mean_llm_calls": round(mean([float(row["llm_call_count"] or 0) for row in rows]), 3),
                "mean_total_tokens": round(mean([
                    float(row["prompt_tokens"] or 0) + float(row["completion_tokens"] or 0)
                    for row in rows
                ]), 3),
            }
        )
    condition_rows.sort(key=lambda row: row["mean_score_0_1"], reverse=True)
    write_csv(output / "tables/module_summary.csv", condition_rows)

    by_cell = {
        (row["case"], row["repeat_id"], row["condition"]): row
        for row in outcome_rows
    }
    paired = []
    for row in outcome_rows:
        if row["condition"] == "Full FlowPilot":
            continue
        baseline = by_cell.get((row["case"], row["repeat_id"], "Full FlowPilot"))
        if baseline:
            paired.append(
                {
                    "case": row["case"],
                    "repeat_id": row["repeat_id"],
                    "condition": row["condition"],
                    "condition_score": row["consensus_score_0_1"],
                    "full_score": baseline["consensus_score_0_1"],
                    "condition_minus_full": round(
                        row["consensus_score_0_1"] - baseline["consensus_score_0_1"], 6
                    ),
                }
            )
    write_csv(output / "tables/paired_module_effects.csv", paired)
    effect_rows = []
    for condition in sorted({row["condition"] for row in paired}):
        values = [row["condition_minus_full"] for row in paired if row["condition"] == condition]
        effect_rows.append(
            {
                "condition": condition,
                "n_pairs": len(values),
                "mean_condition_minus_full": round(mean(values), 6),
                "sd": round(sd(values), 6),
                "wins": sum(value > 0 for value in values),
                "ties": sum(value == 0 for value in values),
                "losses": sum(value < 0 for value in values),
            }
        )
    write_csv(output / "tables/module_effects.csv", effect_rows)

    applicable = [row for row in criterion if row["applicability"] == "APPLICABLE"]
    criterion_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in applicable:
        criterion_groups[(row["architecture"], row["criterion_id"])].append(
            float(row["score_0_4"]) / 4.0
        )
    criterion_effects = []
    criterion_ids = sorted({key[1] for key in criterion_groups})
    for condition in groups:
        if condition == "Full FlowPilot":
            continue
        for criterion_id in criterion_ids:
            condition_values = criterion_groups.get((condition, criterion_id), [])
            full_values = criterion_groups.get(("Full FlowPilot", criterion_id), [])
            if condition_values and full_values:
                criterion_effects.append(
                    {
                        "condition": condition,
                        "criterion_id": criterion_id,
                        "condition_mean": round(mean(condition_values), 6),
                        "full_mean": round(mean(full_values), 6),
                        "condition_minus_full": round(
                            mean(condition_values) - mean(full_values), 6
                        ),
                    }
                )
    write_csv(output / "tables/criterion_module_effects.csv", criterion_effects)
    _figures(output, condition_rows, effect_rows, criterion_effects, outcome_rows)

    module_summary = {
        "schema_version": "flowpilot_newgen_2_module_attribution_v1.0",
        "candidate_count": len(outcome_rows),
        "condition_count": len(condition_rows),
        "case_count": len({row["case"] for row in outcome_rows}),
        "repeat_count": len({row["repeat_id"] for row in outcome_rows}),
        "judgment_count": summary["judgment_count"],
        "conditions": condition_rows,
        "effects_vs_full": effect_rows,
        "judge_agreement": summary["judge_agreement"],
        "execution_audit": {
            key: execution_audit[key]
            for key in ("run_count", "passed", "failed", "all_passed", "executable_count")
        },
    }
    write_json(output / "module_summary.json", module_summary)
    _documentation(root, output, module_summary)
    _manifest(output)
    _manifest(root, "CAMPAIGN_ARTIFACT_MANIFEST.sha256")
    return module_summary


def _figures(output, summaries, effects, criterion_effects, outcomes):
    import matplotlib.pyplot as plt
    import numpy as np

    figdir = output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(figdir / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
        fig.savefig(figdir / f"{name}.pdf", facecolor="white", bbox_inches="tight")
        plt.close(fig)

    ordered = sorted(summaries, key=lambda row: row["mean_score_0_1"])
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#16838f" if row["condition"] == "Full FlowPilot" else "#8d969e" for row in ordered]
    ax.barh([row["condition"] for row in ordered], [row["mean_score_0_1"] for row in ordered], color=colors)
    ax.set_xlim(0, 1); ax.set_xlabel("NewGen 2.0 consensus score")
    ax.set_title("Qwen3.8 module-attribution outcomes")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    save(fig, "fig01_module_scores")

    ordered_effects = sorted(effects, key=lambda row: row["mean_condition_minus_full"])
    fig, ax = plt.subplots(figsize=(10, 7))
    values = [row["mean_condition_minus_full"] for row in ordered_effects]
    ax.barh([row["condition"] for row in ordered_effects], values, color=["#c45b5b" if value < 0 else "#4d9b72" for value in values])
    ax.axvline(0, color="black", linewidth=1); ax.set_xlabel("Condition minus full FlowPilot")
    ax.set_title("Matched architecture effects")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    save(fig, "fig02_paired_module_effects")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(summaries)); width = .38
    ax.bar(x-width/2, [row["mean_deterministic_score"] for row in summaries], width, label="Deterministic score", color="#4c78a8")
    closure = [row["executable_rate"] for row in summaries]
    closure_labels = ["N/A" if row["condition"] in {"One-shot", "No council"} else f"{row['executable_rate']:.0%}" for row in summaries]
    bars = ax.bar(x+width/2, closure, width, label="FlowPilot final-contract closure", color="#e59f44")
    for bar, label in zip(bars, closure_labels):
        if label == "N/A":
            ax.text(bar.get_x()+bar.get_width()/2, .03, label, ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x, [row["condition"] for row in summaries], rotation=55, ha="right")
    ax.set_ylim(0, 1); ax.set_title("Deterministic integrity and contract closure"); ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    save(fig, "fig03_deterministic_closure")

    fig, (ax, key_ax) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.1, 1]})
    numbered = sorted(summaries, key=lambda row: (row["mean_llm_calls"], row["condition"]))
    for number, row in enumerate(numbered, start=1):
        color = "#16838f" if row["condition"] == "Full FlowPilot" else "#8d969e"
        ax.scatter(row["mean_llm_calls"], row["mean_score_0_1"], color=color, s=80)
        ax.text(row["mean_llm_calls"], row["mean_score_0_1"], str(number), color="white", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Mean LLM calls per outcome"); ax.set_ylabel("NewGen 2.0 consensus score")
    ax.set_ylim(0, 1); ax.set_title("Quality versus council call budget")
    ax.grid(alpha=.2)
    key_ax.axis("off")
    key_ax.text(0, 1, "Condition key", va="top", fontweight="bold")
    for number, row in enumerate(numbered, start=1):
        key_ax.text(0, 1-number*.058, f"{number:>2}  {row['condition']}", va="top", fontsize=8)
    fig.tight_layout(); save(fig, "fig05_quality_vs_calls")

    if criterion_effects:
        conditions = sorted({row["condition"] for row in criterion_effects})
        criteria = sorted({row["criterion_id"] for row in criterion_effects})
        lookup = {(row["condition"], row["criterion_id"]): row["condition_minus_full"] for row in criterion_effects}
        matrix = np.array([[lookup.get((condition, criterion), np.nan) for criterion in criteria] for condition in conditions])
        fig, ax = plt.subplots(figsize=(12, max(5, .45*len(conditions))))
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-.5, vmax=.5, aspect="auto")
        ax.set_xticks(range(len(criteria)), criteria, rotation=45, ha="right")
        ax.set_yticks(range(len(conditions)), conditions)
        fig.colorbar(image, ax=ax, label="Condition minus full criterion score")
        ax.set_title("Criterion-level module effects"); fig.tight_layout()
        save(fig, "fig04_criterion_effects")

    conditions = [row["condition"] for row in summaries]
    cases = sorted({row["case"] for row in outcomes})
    outcome_lookup = {(row["condition"], row["case"]): row["consensus_score_0_1"] for row in outcomes}
    matrix = np.array([[outcome_lookup.get((condition, case), np.nan) for case in cases] for condition in conditions])
    fig, ax = plt.subplots(figsize=(9, max(6, .42*len(conditions))))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right")
    ax.set_yticks(range(len(conditions)), conditions)
    for y in range(matrix.shape[0]):
        for x_index in range(matrix.shape[1]):
            ax.text(x_index, y, f"{matrix[y, x_index]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Consensus score")
    ax.set_title("Condition performance by protocol"); fig.tight_layout()
    save(fig, "fig06_case_condition_scores")

    fig, ax = plt.subplots(figsize=(10, 7))
    critical_order = sorted(summaries, key=lambda row: row["critical_flags"])
    ax.barh([row["condition"] for row in critical_order], [row["critical_flags"] for row in critical_order], color=["#c45b5b" if row["critical_flags"] else "#7ca58b" for row in critical_order])
    ax.set_xlabel("Total critical judge flags across three outcomes")
    ax.set_title("Critical-error burden by condition")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    save(fig, "fig07_critical_flags")

    full = next(row for row in summaries if row["condition"] == "Full FlowPilot")
    effect_lookup = {row["condition"]: row["mean_condition_minus_full"] for row in effects}
    efficiency_rows = [row for row in summaries if row["condition"] not in {"Full FlowPilot", "One-shot", "No council"}]
    fig, (ax, key_ax) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.1, 1]})
    for number, row in enumerate(efficiency_rows, start=1):
        calls_saved = full["mean_llm_calls"] - row["mean_llm_calls"]
        effect = effect_lookup[row["condition"]]
        ax.scatter(calls_saved, effect, color="#4c78a8", s=80)
        ax.text(calls_saved, effect, str(number), color="white", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.axhline(0, color="#333333", linewidth=1); ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("LLM calls saved versus full FlowPilot"); ax.set_ylabel("Condition minus full score")
    ax.set_title("Quality-cost effect of module removal"); ax.grid(alpha=.2)
    key_ax.axis("off"); key_ax.text(0, 1, "Condition key", va="top", fontweight="bold")
    for number, row in enumerate(efficiency_rows, start=1):
        key_ax.text(0, 1-number*.07, f"{number:>2}  {row['condition']}", va="top", fontsize=8)
    fig.tight_layout(); save(fig, "fig08_module_efficiency_effect")

    delta_lookup = {}
    for condition in conditions:
        for case in cases:
            value = outcome_lookup.get((condition, case), np.nan)
            baseline = outcome_lookup.get(("Full FlowPilot", case), np.nan)
            delta_lookup[(condition, case)] = value - baseline
    delta_matrix = np.array([[delta_lookup[(condition, case)] for case in cases] for condition in conditions if condition != "Full FlowPilot"])
    delta_conditions = [condition for condition in conditions if condition != "Full FlowPilot"]
    fig, ax = plt.subplots(figsize=(9, max(6, .42*len(delta_conditions))))
    image = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-.5, vmax=.5, aspect="auto")
    ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right")
    ax.set_yticks(range(len(delta_conditions)), delta_conditions)
    for y in range(delta_matrix.shape[0]):
        for x_index in range(delta_matrix.shape[1]):
            ax.text(x_index, y, f"{delta_matrix[y, x_index]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Condition minus full score")
    ax.set_title("Case-level module effects versus full FlowPilot"); fig.tight_layout()
    save(fig, "fig09_case_level_effects")


def _manifest(output, filename="ARTIFACT_MANIFEST.sha256"):
    destination = output / filename
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.resolve() != destination.resolve():
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _documentation(root, output, summary):
    rows = summary["conditions"]
    table = "\n".join(
        f"| {row['condition']} | {row['mean_score_0_1']:.3f} | {row['mean_deterministic_score']:.3f} | {row['executable_rate']:.1%} | {row['mean_llm_calls']:.1f} | {row['critical_flags']} |"
        for row in rows
    )
    text = f"""# NewGen 2.0 Qwen3.8 Module Attribution

## Frozen design

- Fixed generator: Qwen3.8-27B.
- Universal 14-criterion NewGen 2.0 rubric.
- Same protocol, objective, strict case inventory, source exclusion, temperature, and seed schedule across matched conditions.
- Three independent judge families: Qwen, OpenAI, and Claude.
- Disabled specialists receive no call and no default-score penalty; active council weights are renormalized.
- Deterministic inventory and safety gates remain active in executable conditions.

## Results

| Condition | Consensus | Deterministic | Executable | LLM calls | Critical flags |
|---|---:|---:|---:|---:|---:|
{table}

The execution-contract audit passed {summary['execution_audit']['passed']}/{summary['execution_audit']['run_count']} saved runs. This three-case screen reports descriptive paired effects and does not make inferential or universal-superiority claims. It does not claim complete architecture blinding because output structure can reveal provenance. Raw prompts, responses, retries, stage snapshots, and model metadata remain in `{root}`.
"""
    (output / "MODULE_ATTRIBUTION_REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_report(args.root.resolve(), args.output.resolve())
