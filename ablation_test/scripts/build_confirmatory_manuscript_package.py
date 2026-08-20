from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ablation_test.src.paths import resolve_artifact_path


CONDITION_ORDER = [
    "gpt56_terra_one_shot",
    "qwen27b_full_flowpilot",
    "qwen27b_one_shot",
]
CONDITION_LABELS = {
    "gpt56_terra_one_shot": "GPT-5.6 Terra\none-shot",
    "qwen27b_full_flowpilot": "Qwen 27B\n+ FlowPilot",
    "qwen27b_one_shot": "Qwen 27B\none-shot",
}
CONDITION_COLORS = {
    "gpt56_terra_one_shot": "#E76F51",
    "qwen27b_full_flowpilot": "#2A9D8F",
    "qwen27b_one_shot": "#4C78A8",
}
SCENARIO_ORDER = [
    "suzuki_feasible",
    "suzuki_infeasible",
    "photo_oxidation_feasible",
    "photo_oxidation_infeasible",
    "hydrogenolysis_feasible",
    "hydrogenolysis_infeasible",
    "dinitration_feasible",
    "dinitration_infeasible",
    "multistep_feasible",
    "multistep_infeasible",
]
SCENARIO_LABELS = {
    "suzuki_feasible": "Suzuki\nfeasible",
    "suzuki_infeasible": "Suzuki\ninfeasible",
    "photo_oxidation_feasible": "Photo\nfeasible",
    "photo_oxidation_infeasible": "Photo\ninfeasible",
    "hydrogenolysis_feasible": "H2\nfeasible",
    "hydrogenolysis_infeasible": "H2\ninfeasible",
    "dinitration_feasible": "Nitration\nfeasible",
    "dinitration_infeasible": "Nitration\ninfeasible",
    "multistep_feasible": "Multistep\nfeasible",
    "multistep_infeasible": "Multistep\ninfeasible",
}


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _failure_type(row: dict[str, Any]) -> str:
    failures = []
    if not row["disposition_correct"]:
        if row["expected_disposition"] == "SCREEN":
            failures.append("false_block_feasible")
        else:
            failures.append("unsafe_screen_infeasible")
    if not row["critical_engineering_pass"]:
        failures.append("critical_engineering_failure")
    return "|".join(failures) or "none"


def _constraint_failure_text(oracle: dict[str, Any]) -> str:
    parts = []
    for check in oracle.get("constraint_checks") or []:
        if check.get("passed"):
            continue
        reasons = "|".join(str(item) for item in check.get("reasons") or [])
        parts.append(f"{check.get('constraint_id')}:{reasons or 'failed'}")
    return ";".join(parts)


def build_cell_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for raw in manifest_rows:
        run_dir = resolve_artifact_path(raw["run_dir"])
        oracle = _read_json(run_dir / "oracle_metrics.json")
        disposition_correct = _bool(raw["disposition_correct"])
        engineering_pass = _bool(raw["critical_engineering_pass"])
        row = {
            "condition_id": raw["condition_id"],
            "condition_label": raw["condition_label"],
            "provider": raw["provider"],
            "model": raw["model"],
            "scenario_id": raw["scenario_id"],
            "pair_id": raw["pair_id"],
            "scenario_kind": raw["scenario_kind"],
            "repeat": int(raw["repeat"]),
            "seed": int(raw["seed"]),
            "expected_disposition": raw["expected_disposition"],
            "reported_disposition": raw["reported_disposition"],
            "disposition_correct": disposition_correct,
            "critical_engineering_pass": engineering_pass,
            "critical_violation_count": int(raw["critical_violation_count"]),
            "joint_success": disposition_correct and engineering_pass,
            "failure_type": "none",
            "failed_constraints": _constraint_failure_text(oracle),
            "runtime_s": float(raw["runtime_s"]),
            "llm_call_count": int(float(raw["llm_call_count"])),
            "total_tokens": int(float(raw["total_tokens"])),
            "design_input_sha256": raw["design_input_sha256"],
            "artifact_complete": _bool(raw["artifact_complete"]),
            "run_dir": raw["run_dir"],
        }
        row["failure_type"] = _failure_type(row)
        output.append(row)
    return output


def build_scenario_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[(row["condition_id"], row["scenario_id"])].append(row)

    output = []
    for condition in CONDITION_ORDER:
        for scenario in SCENARIO_ORDER:
            rows = grouped[(condition, scenario)]
            output.append(
                {
                    "condition_id": condition,
                    "scenario_id": scenario,
                    "scenario_kind": rows[0]["scenario_kind"],
                    "repeat_count": len(rows),
                    "joint_success_n": sum(row["joint_success"] for row in rows),
                    "joint_success_rate": mean(
                        row["joint_success"] for row in rows
                    ),
                    "disposition_correct_n": sum(
                        row["disposition_correct"] for row in rows
                    ),
                    "engineering_pass_n": sum(
                        row["critical_engineering_pass"] for row in rows
                    ),
                    "reported_dispositions": "|".join(
                        f"{name}:{count}"
                        for name, count in sorted(
                            Counter(
                                row["reported_disposition"] for row in rows
                            ).items()
                        )
                    ),
                    "all_repeats_joint_success": all(
                        row["joint_success"] for row in rows
                    ),
                    "critical_violation_count": sum(
                        row["critical_violation_count"] for row in rows
                    ),
                }
            )
    return output


def build_failure_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition_id": row["condition_id"],
            "scenario_id": row["scenario_id"],
            "repeat": row["repeat"],
            "expected_disposition": row["expected_disposition"],
            "reported_disposition": row["reported_disposition"],
            "failure_type": row["failure_type"],
            "failed_constraints": row["failed_constraints"],
            "critical_violation_count": row["critical_violation_count"],
            "run_dir": row["run_dir"],
        }
        for row in cell_rows
        if not row["joint_success"]
    ]


def build_flowpilot_diagnostic_rows(
    cell_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in cell_rows:
        if row["condition_id"] != "qwen27b_full_flowpilot":
            continue
        result = _read_json(resolve_artifact_path(row["run_dir"]) / "result.json")
        disposition = result.get("design_disposition") or {}
        reasons = [str(item) for item in disposition.get("screen_reasons") or []]
        reason_text = " | ".join(reasons).lower()
        validation = result.get("final_validation") or {}
        checks = validation.get("checks") or {}
        output.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario_kind": row["scenario_kind"],
                "repeat": row["repeat"],
                "reported_disposition": row["reported_disposition"],
                "joint_success": row["joint_success"],
                "engine_fallback": (
                    "engine_fallback" in reason_text
                    or "council fallback" in reason_text
                ),
                "all_candidates_disqualified": (
                    "all candidates disqualified" in reason_text
                ),
                "no_usable_candidates": "no usable candidates" in reason_text,
                "proposal_not_engine_validated": (
                    "not engine-validated" in reason_text
                ),
                "final_validation_status": validation.get("status", ""),
                "final_validation_checks_passed": (
                    bool(checks) and all(bool(value) for value in checks.values())
                ),
                "screen_reasons": " | ".join(reasons),
                "run_dir": row["run_dir"],
            }
        )
    return output


def build_scenario_characteristics(
    scenario_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for case in scenario_payload["cases"]:
        output.append(
            {
                "scenario_id": case["scenario_id"],
                "pair_id": case["pair_id"],
                "scenario_kind": case["scenario_kind"],
                "title": case["title"],
                "category": case["category"],
                "expected_disposition": case["expected_disposition"],
                "inventory_id": case["inventory_id"],
                "source_record_id": case["source_record_id"],
                "controlled_change": case["controlled_change"],
                "hard_constraints": " | ".join(case["hard_constraints"]),
                "oracle_constraint_ids": "|".join(
                    item["constraint_id"]
                    for item in case["oracle_constraints"]
                ),
            }
        )
    return output


def build_input_integrity_summary(audit: dict[str, Any]) -> dict[str, Any]:
    records = audit.get("records") or []
    return {
        "schema_version": "flowpilot_confirmatory_input_integrity_v1.0",
        "passed": bool(audit.get("passed_for_recorded_cells")),
        "record_count": len(records),
        "conditions": audit.get("represented_conditions") or [],
        "missing_conditions": audit.get("missing_conditions") or [],
        "all_public_inputs_identical": all(
            item.get("public_artifact_identical") for item in records
        ),
        "all_inventory_inputs_identical": all(
            item.get("inventory_artifact_identical") for item in records
        ),
        "unique_public_hashes": len(
            {item.get("input_public_sha256") for item in records}
        ),
        "unique_inventory_hashes": len(
            {item.get("input_inventory_sha256") for item in records}
        ),
    }


def build_heatmap(
    scenario_rows: list[dict[str, Any]],
    figures: Path,
) -> None:
    lookup = {
        (row["condition_id"], row["scenario_id"]): row["joint_success_rate"]
        for row in scenario_rows
    }
    matrix = np.array(
        [
            [lookup[(condition, scenario)] for scenario in SCENARIO_ORDER]
            for condition in CONDITION_ORDER
        ]
    )
    fig, ax = plt.subplots(figsize=(12.4, 4.1))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        np.arange(len(SCENARIO_ORDER)),
        [SCENARIO_LABELS[item] for item in SCENARIO_ORDER],
    )
    ax.set_yticks(
        np.arange(len(CONDITION_ORDER)),
        [CONDITION_LABELS[item].replace("\n", " ") for item in CONDITION_ORDER],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{int(round(3 * value))}/3",
                ha="center",
                va="center",
                color="#111111",
                fontsize=10,
            )
    ax.set_title("Joint success across scenario clusters and repeats")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Joint success rate")
    ax.tick_params(axis="x", labelsize=9)
    _save(fig, figures, "06_scenario_success_heatmap")


def build_failure_figure(
    cell_rows: list[dict[str, Any]],
    figures: Path,
) -> None:
    false_blocks = []
    engineering = []
    for condition in CONDITION_ORDER:
        rows = [row for row in cell_rows if row["condition_id"] == condition]
        false_blocks.append(
            sum(
                row["failure_type"].find("false_block_feasible") >= 0
                for row in rows
            )
        )
        engineering.append(
            sum(
                row["failure_type"].find("critical_engineering_failure") >= 0
                for row in rows
            )
        )
    x = np.arange(len(CONDITION_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        x - width / 2,
        false_blocks,
        width,
        color="#E76F51",
        label="False block on feasible case",
    )
    ax.bar(
        x + width / 2,
        engineering,
        width,
        color="#457B9D",
        label="Critical engineering failure",
    )
    ax.set_xticks(
        x,
        [CONDITION_LABELS[item] for item in CONDITION_ORDER],
    )
    ax.set_ylabel("Cells meeting category (n)")
    ax.set_ylim(0, max(false_blocks + engineering + [1]) + 2)
    ax.set_title("Non-exclusive failure categories across 30 cells per condition")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "07_failure_taxonomy")


def build_pairwise_scenario_figure(
    scenario_rows: list[dict[str, Any]],
    figures: Path,
) -> None:
    lookup = {
        (row["condition_id"], row["scenario_id"]): row["joint_success_rate"]
        for row in scenario_rows
    }
    qwen_delta = [
        100
        * (
            lookup[("qwen27b_full_flowpilot", scenario)]
            - lookup[("qwen27b_one_shot", scenario)]
        )
        for scenario in SCENARIO_ORDER
    ]
    gpt_delta = [
        100
        * (
            lookup[("qwen27b_full_flowpilot", scenario)]
            - lookup[("gpt56_terra_one_shot", scenario)]
        )
        for scenario in SCENARIO_ORDER
    ]
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    ax.bar(
        x - width / 2,
        qwen_delta,
        width,
        color=CONDITION_COLORS["qwen27b_one_shot"],
        label="FlowPilot minus Qwen one-shot",
    )
    ax.bar(
        x + width / 2,
        gpt_delta,
        width,
        color=CONDITION_COLORS["gpt56_terra_one_shot"],
        label="FlowPilot minus GPT one-shot",
    )
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(
        x,
        [SCENARIO_LABELS[item] for item in SCENARIO_ORDER],
    )
    ax.set_ylabel("Joint-success difference (percentage points)")
    ax.set_title("Scenario-level paired joint-success differences")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "08_scenario_level_pairwise_effect")


def build_pipeline_diagnostics_figure(
    diagnostic_rows: list[dict[str, Any]],
    figures: Path,
) -> None:
    feasible_scenarios = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario.endswith("_feasible")
        and not scenario.endswith("_infeasible")
    ]
    fallback_counts = []
    clean_counts = []
    for scenario in feasible_scenarios:
        rows = [
            row for row in diagnostic_rows
            if row["scenario_id"] == scenario
        ]
        fallback_count = sum(row["engine_fallback"] for row in rows)
        fallback_counts.append(fallback_count)
        clean_counts.append(len(rows) - fallback_count)

    x = np.arange(len(feasible_scenarios))
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(
        x,
        clean_counts,
        color="#2A9D8F",
        label="Clean council path",
    )
    ax.bar(
        x,
        fallback_counts,
        bottom=clean_counts,
        color="#F4A261",
        label="Engine fallback before final validation",
    )
    ax.set_xticks(
        x,
        [
            SCENARIO_LABELS[item].replace("\nfeasible", "")
            for item in feasible_scenarios
        ],
    )
    ax.set_ylabel("Repeats (n=3 per feasible scenario)")
    ax.set_ylim(0, 3.5)
    ax.set_title("Internal FlowPilot execution path for feasible scenarios")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "09_flowpilot_internal_execution_path")


def _write_methods(
    output: Path,
    status: dict[str, Any],
    integrity: dict[str, Any],
) -> None:
    text = f"""# Confirmatory Benchmark Methods

## Study design

This matched confirmatory benchmark evaluated three conditions:

1. Qwen 27B used inside the complete FlowPilot architecture.
2. The same Qwen 27B model used as a one-shot general model.
3. GPT-5.6 Terra used as a one-shot general model.

The benchmark contains five chemistry families. Each has a feasible inventory
and a controlled infeasible inventory, producing ten scenario clusters. Every
condition was repeated three times for each scenario, for
`10 x 3 x 3 = {status["planned_cells"]}` cells.

## Frozen inputs

Every condition received the same public protocol, objective, hard constraints,
and inventory for a given scenario. The input audit covers
`{integrity["record_count"]}` records. Public-input identity:
`{integrity["all_public_inputs_identical"]}`. Inventory-input identity:
`{integrity["all_inventory_inputs_identical"]}`.

The expected disposition and oracle constraints were stored separately from
the public prompt and used only by deterministic post-run scoring.

## Primary endpoint

Joint success required both:

1. Correct disposition: `SCREEN` for feasible scenarios and `BLOCK` for
   deliberately infeasible scenarios.
2. Passage of every critical deterministic engineering constraint.

A verbose answer did not receive extra credit. A correct disposition with a
critical numerical violation failed the joint endpoint.

## Statistical unit

The ten scenario clusters, not the 30 repeated calls, were treated as the
independent sampling units for pairwise uncertainty intervals. The reported
95% intervals use scenario-cluster bootstrap resampling.

## Reproducibility

The run preserved the exact prompt, model response, parsed result, oracle
metrics, model metadata, logs, and checksums for every cell. The benchmark
status was `{status["status"]}` with `{status["remaining_cells"]}` remaining
cells.
"""
    (output / "METHODS.md").write_text(text, encoding="utf-8")


def _write_scoring(output: Path) -> None:
    text = """# Scoring Specification

## Disposition

- `SCREEN`: the listed inventory can support a conservative first experiment,
  but experimental validation is still required.
- `BLOCK`: at least one hard process, inventory, safety, or topology
  requirement prevents execution of the proposed design.

## Joint success

```text
joint_success =
    disposition_correct
    AND critical_engineering_pass
```

No condition was assigned a default zero. Every response was parsed and scored
with the same deterministic oracle. Failed cells retain the exact failed
constraint IDs and observed values.

## Critical scenario checks

- Suzuki: exact operating temperature and listed thermal hardware.
- Photochemistry: required wavelength and compatible photoreactor.
- Hydrogenolysis: certified hydrogen service, pressure, liquid flow, and bed.
- Nitration: explicit permission or prohibition for the hazardous operation.
- Multistep: serial reactor volume, interstage addition, and feed segregation.

## Repeatability

Disposition agreement measures whether three calls return the same decision.
All-repeat joint success is stricter: every repeat must also be correct and
engineering-compliant.

## Interpretation boundary

This score measures design feasibility, constraint compliance, and calibrated
decision behavior. It does not measure experimental yield prediction,
chemical novelty, or globally optimal operating conditions.
"""
    (output / "SCORING.md").write_text(text, encoding="utf-8")


def _write_results(
    output: Path,
    condition_summary: list[dict[str, str]],
    pairwise: list[dict[str, str]],
) -> None:
    by_id = {row["condition_id"]: row for row in condition_summary}
    full = by_id["qwen27b_full_flowpilot"]
    qwen = by_id["qwen27b_one_shot"]
    gpt = by_id["gpt56_terra_one_shot"]
    pair_by_comparator = {row["comparator"]: row for row in pairwise}
    qwen_pair = pair_by_comparator["qwen27b_one_shot"]
    gpt_pair = pair_by_comparator["gpt56_terra_one_shot"]

    def pct(value: str) -> str:
        return f"{100 * float(value):.1f}%"

    text = f"""# Manuscript-Ready Results

Across 90 matched evaluations, Qwen 27B inside FlowPilot achieved joint success
in {full["joint_success_n"]}/{full["completed_cells"]} cells
({pct(full["joint_success_rate"])}), compared with
{qwen["joint_success_n"]}/{qwen["completed_cells"]}
({pct(qwen["joint_success_rate"])}) for the same Qwen model used one-shot and
{gpt["joint_success_n"]}/{gpt["completed_cells"]}
({pct(gpt["joint_success_rate"])}) for GPT-5.6 Terra one-shot.

FlowPilot completed all feasible designs and correctly blocked all infeasible
inventories without a critical violation. Qwen one-shot failed two feasible
hydrogenolysis cells and accumulated four critical violations. GPT one-shot
failed eight cells, primarily by over-blocking feasible scenarios, and had two
critical multistep volume violations.

The scenario-clustered difference between FlowPilot and Qwen one-shot was
{100 * float(qwen_pair["mean_joint_success_delta"]):.1f} percentage points
(95% bootstrap interval
{100 * float(qwen_pair["cluster_bootstrap95_low"]):.1f} to
{100 * float(qwen_pair["cluster_bootstrap95_high"]):.1f}). FlowPilot was better
in {qwen_pair["scenario_clusters_treatment_better"]} scenario cluster, tied in
{qwen_pair["scenario_clusters_tied"]}, and worse in
{qwen_pair["scenario_clusters_treatment_worse"]}. This supports an observed
same-model architecture benefit, while the interval touching zero means the
small protocol sample does not establish a decisive population-wide effect.

Against GPT one-shot, the clustered difference was
{100 * float(gpt_pair["mean_joint_success_delta"]):.1f} percentage points
(95% bootstrap interval
{100 * float(gpt_pair["cluster_bootstrap95_low"]):.1f} to
{100 * float(gpt_pair["cluster_bootstrap95_high"]):.1f}).

These results support FlowPilot's advantage for explicit feasibility and
constraint-management tasks. They should not be presented as evidence of
superior wet-lab yield prediction.
"""
    (output / "MANUSCRIPT_RESULTS.md").write_text(text, encoding="utf-8")


def _write_failure_audit(
    output: Path,
    failure_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Failure Audit",
        "",
        f"Ten of 90 cells failed the joint endpoint. FlowPilot contributed "
        f"{sum(row['condition_id'] == 'qwen27b_full_flowpilot' for row in failure_rows)}.",
        "",
        "| Condition | Scenario | Repeat | Failure | Failed constraints |",
        "|---|---|---:|---|---|",
    ]
    for row in failure_rows:
        lines.append(
            f"| {CONDITION_LABELS[row['condition_id']].replace(chr(10), ' ')} "
            f"| {row['scenario_id']} | {row['repeat']} "
            f"| {row['failure_type']} | {row['failed_constraints'] or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Pattern",
            "",
            "- Qwen one-shot: two feasible hydrogenolysis responses returned `BLOCK`",
            "  and omitted required pressure and liquid-flow values.",
            "- GPT one-shot: six feasible responses were over-conservatively blocked.",
            "- GPT one-shot: two multistep responses represented the required 15 mL",
            "  serial train as a stage-volume object rather than the allowed total",
            "  reactor volume, failing the deterministic volume contract.",
            "- No method incorrectly screened a deliberately infeasible inventory.",
        ]
    )
    (output / "FAILURE_AUDIT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_limitations(output: Path) -> None:
    text = """# Limitations and Claims Boundary

1. The study includes five chemistry families and ten scenario clusters.
   Confidence intervals are therefore wide.
2. Feasibility labels are defined by explicit inventory perturbations and
   deterministic oracle constraints, not by blinded flow-chemist ratings.
3. The full architecture uses substantially more calls, tokens, and runtime
   than one-shot controls.
4. The benchmark measures compliance and calibrated refusal, not wet-lab
   conversion, selectivity, yield, productivity, or long-term operability.
5. The same-model FlowPilot versus Qwen interval touches zero. The observed
   advantage is promising but should be validated on more independent
   protocols.
6. All infeasible cases were correctly blocked by all conditions, so this
   dataset does not distinguish methods on unsafe-screen errors.
7. Future confirmation should freeze a new external protocol set before model
   execution and, where possible, compare proposed screens with wet-lab data.
8. Twelve of 15 feasible FlowPilot cells used an engine fallback before the
   deterministic final validator. The 100% endpoint therefore establishes
   final-output compliance, not clean council convergence.

## Defensible claim

On this frozen matched benchmark, Qwen 27B inside FlowPilot produced more
reliable constraint-compliant decisions than the same Qwen model used
one-shot and GPT-5.6 Terra used one-shot.

## Claim to avoid

FlowPilot has been proven universally superior for batch-to-flow translation
or proven to predict higher experimental yields.
"""
    (output / "LIMITATIONS.md").write_text(text, encoding="utf-8")


def _write_pipeline_diagnostics(
    output: Path,
    diagnostic_rows: list[dict[str, Any]],
) -> None:
    feasible = [
        row for row in diagnostic_rows
        if row["scenario_kind"] == "feasible"
    ]
    fallback = [row for row in feasible if row["engine_fallback"]]
    clean = [row for row in feasible if not row["engine_fallback"]]
    text = f"""# FlowPilot Internal Pipeline Diagnostics

The primary benchmark scores the serialized final answer. This additional
audit examines how the complete FlowPilot condition reached that answer.

Of {len(feasible)} feasible cells, {len(clean)} completed through a clean
council path and {len(fallback)} used an engine fallback before deterministic
inventory enforcement and final validation. All {len(feasible)} final feasible
outputs passed the frozen joint endpoint, but the fallback rate means the
perfect endpoint cannot be interpreted as perfect council convergence.

The fallback concentration was:

- Suzuki: 3/3 feasible repeats.
- Photochemistry: 3/3 feasible repeats.
- Hydrogenolysis: 3/3 feasible repeats.
- Nitration: 0/3 feasible repeats.
- Multistep: 3/3 feasible repeats.

This diagnostic motivated the post-benchmark evidence-first policy change:
predicted intensification is now a soft screening hypothesis by default.
Aggressive process-value rejection remains available only when the caller
explicitly selects the `intensify` policy.

The frozen 90-cell scores were not recomputed after that change. A separately
named validation run is required to measure whether clean council convergence
improves while final constraint compliance is retained.
"""
    (output / "PIPELINE_DIAGNOSTICS.md").write_text(text, encoding="utf-8")


def _write_data_dictionary(output: Path) -> None:
    text = """# Data Dictionary

## `tables/cell_level_results.csv`

- `condition_id`: model/architecture condition.
- `scenario_id`: frozen chemistry and inventory scenario.
- `repeat`, `seed`: repeated-call identifiers.
- `expected_disposition`, `reported_disposition`: oracle and model decisions.
- `disposition_correct`: exact decision match.
- `critical_engineering_pass`: all critical oracle constraints passed.
- `joint_success`: conjunction of the previous two fields.
- `failure_type`: deterministic failure taxonomy.
- `failed_constraints`: failed oracle IDs and reasons.
- `runtime_s`, `llm_call_count`, `total_tokens`: operational burden.
- `design_input_sha256`: hash of the scored design input.

## `tables/scenario_summary.csv`

Aggregates three repeats for each condition and scenario. Scenario-level rates
are the inputs to the clustered comparison.

## `tables/failure_audit.csv`

Contains only cells that failed the joint endpoint and links each row to its
complete raw artifact directory.

## `tables/scenario_characteristics.csv`

Documents source record, controlled inventory change, hard constraints, and
oracle constraint IDs for all ten scenarios.

## `tables/flowpilot_internal_diagnostics.csv`

Records council fallback signals and deterministic final-validation status for
all 30 complete-FlowPilot cells.

## `input_integrity_summary.json`

Summarizes byte-level input identity across conditions.
"""
    (output / "DATA_DICTIONARY.md").write_text(text, encoding="utf-8")


def _write_readme(output: Path) -> None:
    text = """# FlowPilot Confirmatory Manuscript Package

This directory is a derived, checksummed presentation package for the frozen
90-cell confirmatory benchmark. It does not modify the raw benchmark.

## Start here

- `MANUSCRIPT_RESULTS.md`: concise results language.
- `METHODS.md`: study design and statistical unit.
- `SCORING.md`: exact endpoint definition.
- `FAILURE_AUDIT.md`: every failed cell.
- `PIPELINE_DIAGNOSTICS.md`: council fallback and final-validator audit.
- `LIMITATIONS.md`: defensible and unsupported claims.
- `DATA_DICTIONARY.md`: table definitions.

## Figures

- `06_scenario_success_heatmap`: all condition/scenario repeat outcomes.
- `07_failure_taxonomy`: false blocks and engineering failures.
- `08_scenario_level_pairwise_effect`: architecture effect by scenario.
- `09_flowpilot_internal_execution_path`: clean council versus fallback path.

All tables and figures can be regenerated with
`python -m ablation_test.scripts.build_confirmatory_manuscript_package
--study-dir
ablation_results/studies/fair_architecture_confirmatory_disposition_v1_20260730`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="stage2_confirmatory_attempt02_20260730",
    )
    parser.add_argument(
        "--analysis-name",
        default="stage2_confirmatory_analysis_20260730",
    )
    parser.add_argument(
        "--output-name",
        default="stage2_manuscript_package_20260730",
    )
    args = parser.parse_args()

    study = args.study_dir.resolve()
    run = study / args.run_name
    analysis = study / args.analysis_name
    output = study / args.output_name
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(run / "run_manifest.csv")
    if len(manifest_rows) != 90:
        raise ValueError(f"Expected 90 manifest rows, found {len(manifest_rows)}")
    if any(row["status"] != "completed" for row in manifest_rows):
        raise ValueError("All benchmark cells must be completed before packaging")

    cell_rows = build_cell_rows(manifest_rows)
    scenario_rows = build_scenario_rows(cell_rows)
    failure_rows = build_failure_rows(cell_rows)
    diagnostic_rows = build_flowpilot_diagnostic_rows(cell_rows)
    scenario_payload = _read_json(study / "protocol_scenarios.json")
    scenario_characteristics = build_scenario_characteristics(scenario_payload)
    input_audit = _read_json(run / "input_artifact_audit.json")
    integrity = build_input_integrity_summary(input_audit)
    status = _read_json(run / "stage2_status.json")
    condition_summary = _read_csv(
        analysis / "tables" / "condition_summary.csv"
    )
    pairwise = _read_csv(
        analysis / "tables" / "pairwise_scenario_cluster.csv"
    )

    _write_csv(tables / "cell_level_results.csv", cell_rows)
    _write_csv(tables / "scenario_summary.csv", scenario_rows)
    _write_csv(tables / "failure_audit.csv", failure_rows)
    _write_csv(
        tables / "flowpilot_internal_diagnostics.csv",
        diagnostic_rows,
    )
    _write_csv(
        tables / "scenario_characteristics.csv",
        scenario_characteristics,
    )
    (output / "input_integrity_summary.json").write_text(
        json.dumps(integrity, indent=2) + "\n",
        encoding="utf-8",
    )

    build_heatmap(scenario_rows, figures)
    build_failure_figure(cell_rows, figures)
    build_pairwise_scenario_figure(scenario_rows, figures)
    build_pipeline_diagnostics_figure(diagnostic_rows, figures)
    _write_methods(output, status, integrity)
    _write_scoring(output)
    _write_results(output, condition_summary, pairwise)
    _write_failure_audit(output, failure_rows)
    _write_pipeline_diagnostics(output, diagnostic_rows)
    _write_limitations(output)
    _write_data_dictionary(output)
    _write_readme(output)

    package_summary = {
        "schema_version": "flowpilot_confirmatory_manuscript_package_v1.0",
        "study_id": status["study_id"],
        "run_status": status["status"],
        "cell_count": len(cell_rows),
        "scenario_count": len(SCENARIO_ORDER),
        "condition_count": len(CONDITION_ORDER),
        "failed_cell_count": len(failure_rows),
        "input_integrity": integrity,
        "full_flowpilot_joint_success": sum(
            row["joint_success"]
            for row in cell_rows
            if row["condition_id"] == "qwen27b_full_flowpilot"
        ),
    }
    (output / "package_summary.json").write_text(
        json.dumps(package_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    print(json.dumps(package_summary, indent=2))


if __name__ == "__main__":
    main()
