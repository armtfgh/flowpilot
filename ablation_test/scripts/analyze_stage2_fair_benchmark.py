from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "qwen27b_one_shot": "Qwen 27B\none-shot",
    "qwen27b_full_flowpilot": "Qwen 27B\n+ FlowPilot",
    "claude_sonnet5_one_shot": "Claude Sonnet 5\none-shot",
    "gpt56_terra_one_shot": "GPT-5.6 Terra\none-shot",
}
COLORS = {
    "qwen27b_one_shot": "#4C78A8",
    "qwen27b_full_flowpilot": "#2A9D8F",
    "claude_sonnet5_one_shot": "#8E6C8A",
    "gpt56_terra_one_shot": "#E76F51",
}


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - margin, center + margin


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def load_cells(manifest: Path) -> list[dict[str, Any]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    completed = []
    for row in rows:
        if row["status"] != "completed":
            continue
        row["disposition_correct_bool"] = _bool(row["disposition_correct"])
        row["engineering_pass_bool"] = _bool(row["critical_engineering_pass"])
        row["joint_success"] = bool(
            row["disposition_correct_bool"] and row["engineering_pass_bool"]
        )
        row["violations"] = int(row["critical_violation_count"] or 0)
        row["runtime"] = float(row["runtime_s"])
        row["calls"] = int(float(row["llm_call_count"]))
        row["tokens"] = int(float(row["total_tokens"]))
        completed.append(row)
    return completed


def condition_summary(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[row["condition_id"]].append(row)

    output = []
    for condition, rows in sorted(grouped.items()):
        joint = sum(row["joint_success"] for row in rows)
        disposition = sum(row["disposition_correct_bool"] for row in rows)
        engineering = sum(row["engineering_pass_bool"] for row in rows)
        low, high = _wilson(joint, len(rows))
        feasible = [row for row in rows if row["scenario_kind"] == "feasible"]
        infeasible = [row for row in rows if row["scenario_kind"] == "infeasible"]
        output.append(
            {
                "condition_id": condition,
                "completed_cells": len(rows),
                "joint_success_n": joint,
                "joint_success_rate": joint / len(rows),
                "joint_success_wilson95_low": low,
                "joint_success_wilson95_high": high,
                "disposition_accuracy": disposition / len(rows),
                "critical_engineering_pass_rate": engineering / len(rows),
                "feasible_design_success_rate": (
                    sum(row["joint_success"] for row in feasible) / len(feasible)
                ),
                "infeasible_block_success_rate": (
                    sum(row["joint_success"] for row in infeasible) / len(infeasible)
                ),
                "critical_violation_count": sum(row["violations"] for row in rows),
                "mean_runtime_s": mean(row["runtime"] for row in rows),
                "median_runtime_s": median(row["runtime"] for row in rows),
                "mean_llm_calls": mean(row["calls"] for row in rows),
                "mean_total_tokens": mean(row["tokens"] for row in rows),
                "total_tokens": sum(row["tokens"] for row in rows),
            }
        )
    return output


def scenario_summary(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[(row["condition_id"], row["scenario_id"])].append(row)

    output = []
    for (condition, scenario), rows in sorted(grouped.items()):
        disposition_counts = Counter(row["reported_disposition"] for row in rows)
        output.append(
            {
                "condition_id": condition,
                "scenario_id": scenario,
                "scenario_kind": rows[0]["scenario_kind"],
                "expected_disposition": rows[0]["expected_disposition"],
                "repeat_count": len(rows),
                "joint_success_rate": mean(row["joint_success"] for row in rows),
                "all_repeats_joint_success": all(
                    row["joint_success"] for row in rows
                ),
                "reported_disposition_agreement": len(disposition_counts) == 1,
                "reported_dispositions": "|".join(
                    f"{key}:{value}" for key, value in sorted(disposition_counts.items())
                ),
                "critical_violation_count": sum(row["violations"] for row in rows),
            }
        )
    return output


def pairwise_summary(
    scenarios: list[dict[str, Any]],
    treatment: str = "qwen27b_full_flowpilot",
    bootstrap_samples: int = 20_000,
    seed: int = 20260730,
) -> list[dict[str, Any]]:
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scenarios:
        by_condition[row["condition_id"]][row["scenario_id"]] = row
    if treatment not in by_condition:
        return []

    rng = np.random.default_rng(seed)
    output = []
    treatment_rows = by_condition[treatment]
    for comparator, comparator_rows in sorted(by_condition.items()):
        if comparator == treatment:
            continue
        common = sorted(set(treatment_rows) & set(comparator_rows))
        differences = np.array(
            [
                treatment_rows[scenario]["joint_success_rate"]
                - comparator_rows[scenario]["joint_success_rate"]
                for scenario in common
            ],
            dtype=float,
        )
        boot = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            boot[index] = rng.choice(differences, size=len(differences), replace=True).mean()
        output.append(
            {
                "treatment": treatment,
                "comparator": comparator,
                "scenario_clusters": len(common),
                "mean_joint_success_delta": differences.mean(),
                "cluster_bootstrap95_low": float(np.quantile(boot, 0.025)),
                "cluster_bootstrap95_high": float(np.quantile(boot, 0.975)),
                "scenario_clusters_treatment_better": int(sum(differences > 0)),
                "scenario_clusters_tied": int(sum(differences == 0)),
                "scenario_clusters_treatment_worse": int(sum(differences < 0)),
            }
        )
    return output


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_figures(
    summaries: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    output: Path,
) -> None:
    conditions = [row["condition_id"] for row in summaries]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    values = [100 * row["joint_success_rate"] for row in summaries]
    lower = [
        max(
            0.0,
            100
            * (
                row["joint_success_rate"]
                - row["joint_success_wilson95_low"]
            ),
        )
        for row in summaries
    ]
    upper = [
        max(
            0.0,
            100
            * (
                row["joint_success_wilson95_high"]
                - row["joint_success_rate"]
            ),
        )
        for row in summaries
    ]
    ax.bar(x, values, color=[COLORS[item] for item in conditions], width=0.66)
    ax.errorbar(x, values, yerr=[lower, upper], fmt="none", color="#222222", capsize=4)
    ax.set_xticks(x, [LABELS[item] for item in conditions])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Joint success (%)")
    ax.set_title("Correct disposition and critical engineering constraints")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, output, "01_joint_success")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.34
    feasible = [100 * row["feasible_design_success_rate"] for row in summaries]
    blocked = [100 * row["infeasible_block_success_rate"] for row in summaries]
    ax.bar(x - width / 2, feasible, width, color="#2A9D8F", label="Feasible: valid design")
    ax.bar(x + width / 2, blocked, width, color="#E76F51", label="Infeasible: correct block")
    ax.set_xticks(x, [LABELS[item] for item in conditions])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success (%)")
    ax.set_title("Performance on feasible and infeasible inventory scenarios")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, output, "02_feasible_vs_infeasible")

    if pairwise:
        fig, ax = plt.subplots(figsize=(8.2, 3.4))
        y = np.arange(len(pairwise))
        values = np.array([100 * row["mean_joint_success_delta"] for row in pairwise])
        low = np.maximum(
            0.0,
            values
            - np.array(
                [100 * row["cluster_bootstrap95_low"] for row in pairwise]
            ),
        )
        high = np.maximum(
            0.0,
            np.array(
                [100 * row["cluster_bootstrap95_high"] for row in pairwise]
            )
            - values,
        )
        ax.errorbar(values, y, xerr=[low, high], fmt="o", color="#264653", capsize=5)
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_yticks(
            y,
            [
                f"FlowPilot minus\n{LABELS[row['comparator']].replace(chr(10), ' ')}"
                for row in pairwise
            ],
        )
        ax.set_xlabel("Scenario-clustered joint-success difference (percentage points)")
        ax.set_title("Paired architecture effect; 95% scenario bootstrap interval")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        _save(fig, output, "03_paired_architecture_effect")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5))
    runtimes = [row["median_runtime_s"] for row in summaries]
    tokens = [row["mean_total_tokens"] for row in summaries]
    colors = [COLORS[item] for item in conditions]
    ax1.bar(x, runtimes, color=colors)
    ax1.set_xticks(x, [LABELS[item] for item in conditions])
    ax1.set_ylabel("Median runtime (s)")
    ax1.set_title("Runtime")
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax2.bar(x, tokens, color=colors)
    ax2.set_xticks(x, [LABELS[item] for item in conditions])
    ax2.set_yscale("log")
    ax2.set_ylabel("Mean tokens per cell (log scale)")
    ax2.set_title("Model-token workload")
    ax2.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    for axis in (ax1, ax2):
        axis.set_axisbelow(True)
    _save(fig, output, "04_compute_cost")

    repeat_rows = []
    for condition in conditions:
        rows = [row for row in scenarios if row["condition_id"] == condition]
        repeat_rows.append(
            {
                "condition": condition,
                "disposition_agreement": mean(
                    row["reported_disposition_agreement"] for row in rows
                ),
                "all_repeats_success": mean(
                    row["all_repeats_joint_success"] for row in rows
                ),
            }
        )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    agreement = [100 * row["disposition_agreement"] for row in repeat_rows]
    reliable = [100 * row["all_repeats_success"] for row in repeat_rows]
    ax.bar(x - width / 2, agreement, width, color="#F4A261", label="Disposition agreement")
    ax.bar(x + width / 2, reliable, width, color="#457B9D", label="All repeats successful")
    ax.set_xticks(x, [LABELS[item] for item in conditions])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Scenario clusters (%)")
    ax.set_title("Three-repeat outcome consistency")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, output, "05_repeatability")


def write_report(
    output: Path,
    summaries: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    by_id = {row["condition_id"]: row for row in summaries}
    full = by_id["qwen27b_full_flowpilot"]
    qwen = by_id["qwen27b_one_shot"]
    gpt = by_id["gpt56_terra_one_shot"]
    completed = int(status.get("status_counts", {}).get("completed", 0))
    planned = int(status.get("planned_cells", completed))
    complete = bool(status.get("complete", completed == planned))

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    baseline_best = max(qwen["joint_success_rate"], gpt["joint_success_rate"])
    delta = full["joint_success_rate"] - baseline_best
    if delta > 0:
        primary_conclusion = (
            "The observed joint-success rate is higher for Qwen 27B inside "
            f"FlowPilot than for either one-shot baseline by {100 * delta:.1f} "
            "percentage points versus the stronger baseline."
        )
    elif delta == 0:
        primary_conclusion = (
            "Qwen 27B inside FlowPilot ties the stronger one-shot baseline on "
            "observed joint success."
        )
    else:
        primary_conclusion = (
            "Qwen 27B inside FlowPilot does not exceed the stronger one-shot "
            f"baseline on observed joint success ({100 * delta:.1f} percentage points)."
        )

    lines = [
        "# Stage 2 Confirmatory Matched Benchmark Report",
        "",
        f"Study: `{status.get('study_id', 'unknown')}`",
        "",
        f"Status: **{'COMPLETE' if complete else 'PARTIAL'} - "
        f"{completed}/{planned} cells accounted**",
        "",
        "## Primary result",
        "",
        primary_conclusion,
        "",
        "Joint success is deliberately strict: a cell succeeds only when it reports",
        "the correct `SCREEN/BLOCK` disposition and satisfies every critical",
        "deterministic engineering constraint.",
        "",
        "| Condition | Joint success | Feasible design | Infeasible block | Violations | Median runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {LABELS[row['condition_id']].replace(chr(10), ' ')} "
            f"| {row['joint_success_n']}/{row['completed_cells']} ({pct(row['joint_success_rate'])}) "
            f"| {pct(row['feasible_design_success_rate'])} "
            f"| {pct(row['infeasible_block_success_rate'])} "
            f"| {row['critical_violation_count']} "
            f"| {row['median_runtime_s']:.1f} s |"
        )
    lines.extend(
        [
            "",
            f"On feasible scenarios, FlowPilot achieved "
            f"{pct(full['feasible_design_success_rate'])}; Qwen one-shot achieved "
            f"{pct(qwen['feasible_design_success_rate'])}; and GPT-5.6 Terra one-shot "
            f"achieved {pct(gpt['feasible_design_success_rate'])}. On deliberately",
            f"infeasible inventories, the corresponding correct-block rates were "
            f"{pct(full['infeasible_block_success_rate'])}, "
            f"{pct(qwen['infeasible_block_success_rate'])}, and "
            f"{pct(gpt['infeasible_block_success_rate'])}.",
            "",
            "## Paired scenario analysis",
            "",
            "The uncertainty intervals resample the ten scenario clusters, not the 30",
            "repeat cells, avoiding treatment of repeats as independent protocols.",
            "",
            "| Comparison | Mean difference | 95% cluster bootstrap | Better / tied / worse clusters |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            f"| FlowPilot - {LABELS[row['comparator']].replace(chr(10), ' ')} "
            f"| {100 * row['mean_joint_success_delta']:+.1f} pp "
            f"| [{100 * row['cluster_bootstrap95_low']:+.1f}, "
            f"{100 * row['cluster_bootstrap95_high']:+.1f}] pp "
            f"| {row['scenario_clusters_treatment_better']} / "
            f"{row['scenario_clusters_tied']} / "
            f"{row['scenario_clusters_treatment_worse']} |"
        )
    lines.extend(
        [
            "",
            "## Repeatability",
            "",
            "Disposition agreement measures reproducibility, while all-repeat joint",
            "success additionally requires correctness and engineering compliance in",
            "every repeat. Agreement alone is therefore not treated as evidence of quality.",
            "",
        ]
    )
    for condition in sorted(by_id):
        rows = [row for row in scenarios if row["condition_id"] == condition]
        agreement = mean(row["reported_disposition_agreement"] for row in rows)
        reliable = mean(row["all_repeats_joint_success"] for row in rows)
        lines.append(
            f"- {LABELS[condition].replace(chr(10), ' ')}: "
            f"{pct(agreement)} disposition agreement; "
            f"{pct(reliable)} all-repeat joint success."
        )
    lines.extend(
        [
            "",
            "## Compute burden",
            "",
            f"Full FlowPilot used a mean of {full['mean_llm_calls']:.1f} LLM calls and",
            f"{full['mean_total_tokens']:.0f} tokens per cell, versus one call and",
            f"{qwen['mean_total_tokens']:.0f} tokens for Qwen one-shot. Its median runtime",
            f"was {full['median_runtime_s']:.1f} seconds versus",
            f"{qwen['median_runtime_s']:.1f} seconds. Runtime and token use are reported",
            "as operational costs, not included in the scientific success endpoint.",
            "",
            "## Interpretation",
            "",
            "This benchmark isolates architecture value by using the same Qwen 27B model",
            "both as a one-shot baseline and inside FlowPilot. The GPT-5.6 Terra one-shot",
            "condition tests whether a larger general model can replace the specialized",
            "pipeline. The endpoint rewards valid executable designs on feasible cases",
            "and correct refusal on infeasible cases; it does not award points merely for",
            "producing more detailed text.",
            "",
            "## Figures",
            "",
            "![Joint success](figures/01_joint_success.png)",
            "",
            "![Feasible and infeasible split](figures/02_feasible_vs_infeasible.png)",
            "",
            "![Paired effects](figures/03_paired_architecture_effect.png)",
            "",
            "![Compute cost](figures/04_compute_cost.png)",
            "",
            "![Repeatability](figures/05_repeatability.png)",
            "",
            "## Integrity",
            "",
            f"- Run state: `{status.get('status')}`.",
            f"- Completed cells: `{completed}`.",
            f"- Remaining cells: `{status.get('remaining_cells', 0)}`.",
            "- Scenario-cluster bootstrap intervals treat protocols, not repeated calls,",
            "  as the independent sampling unit.",
            "- Raw prompts, responses, calculations, validations, manifests, and model",
            "  metadata remain in the run directory.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="stage2_matched_run_20260730",
    )
    parser.add_argument(
        "--output-name",
        default="stage2_partial_analysis_20260730",
    )
    args = parser.parse_args()

    study = args.study_dir.resolve()
    run = study / args.run_name
    output = study / args.output_name
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    cells = load_cells(run / "run_manifest.csv")
    summaries = condition_summary(cells)
    scenarios = scenario_summary(cells)
    pairwise = pairwise_summary(scenarios)
    status = json.loads((run / "stage2_status.json").read_text(encoding="utf-8"))

    _write_csv(tables / "condition_summary.csv", summaries)
    _write_csv(tables / "scenario_repeatability.csv", scenarios)
    _write_csv(tables / "pairwise_scenario_cluster.csv", pairwise)
    build_figures(summaries, scenarios, pairwise, figures)
    write_report(output, summaries, scenarios, pairwise, status)

    summary = {
        "schema_version": "flowpilot_stage2_confirmatory_analysis_v1.0",
        "study_id": status["study_id"],
        "run_status": status["status"],
        "completed_cells": status["status_counts"].get("completed", 0),
        "remaining_cells": status["remaining_cells"],
        "conditions_analyzed": [row["condition_id"] for row in summaries],
        "condition_summary": summaries,
        "pairwise_scenario_cluster": pairwise,
        "conclusion": (
            "See REPORT.md for the observed architecture comparison and its "
            "scenario-clustered uncertainty intervals."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
