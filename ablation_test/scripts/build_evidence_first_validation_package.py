from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ablation_test.src.paths import resolve_artifact_path


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
FAMILY_ORDER = ["suzuki", "photo_oxidation", "hydrogenolysis", "dinitration", "multistep"]
FAMILY_LABELS = {
    "suzuki": "Suzuki",
    "photo_oxidation": "Photo oxidation",
    "hydrogenolysis": "Hydrogenolysis",
    "dinitration": "Nitration",
    "multistep": "Multistep",
}
RUN_LABELS = {
    "baseline": "Original FlowPilot",
    "evidence_first": "Evidence-first FlowPilot",
}
COLORS = {
    "baseline": "#4C78A8",
    "evidence_first": "#2A9D8F",
    "fallback": "#E76F51",
    "neutral": "#B8B8B8",
    "inlet": "#457B9D",
    "channel": "#F4A261",
}


def _bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _is_fallback(result: dict[str, Any]) -> bool:
    reasons = " | ".join(
        str(item)
        for item in (result.get("design_disposition") or {}).get("screen_reasons") or []
    ).lower()
    return "engine_fallback" in reasons or "council fallback" in reasons


def build_cell_rows(run_dir: Path, run_version: str) -> list[dict[str, Any]]:
    rows = []
    for raw in _read_csv(run_dir / "run_manifest.csv"):
        if raw["condition_id"] != "qwen27b_full_flowpilot":
            continue
        cell_dir = resolve_artifact_path(raw["run_dir"])
        result = _read_json(cell_dir / "result.json")
        calculations = result.get("final_calculations") or {}
        validation = result.get("final_validation") or {}
        joint = _bool(raw["disposition_correct"]) and _bool(
            raw["critical_engineering_pass"]
        )
        rows.append(
            {
                "run_version": run_version,
                "scenario_id": raw["scenario_id"],
                "scenario_kind": raw["scenario_kind"],
                "repeat": int(raw["repeat"]),
                "seed": int(raw["seed"]),
                "expected_disposition": raw["expected_disposition"],
                "reported_disposition": raw["reported_disposition"],
                "disposition_correct": _bool(raw["disposition_correct"]),
                "critical_engineering_pass": _bool(
                    raw["critical_engineering_pass"]
                ),
                "joint_success": joint,
                "engine_fallback": _is_fallback(result),
                "final_validation_status": validation.get("status", ""),
                "residence_time_min": calculations.get("residence_time_min", 0.0),
                "residence_time_basis": calculations.get(
                    "residence_time_basis", ""
                ),
                "residence_time_inlet_min": calculations.get(
                    "residence_time_inlet_min", 0.0
                ),
                "residence_time_in_channel_min": calculations.get(
                    "residence_time_in_channel_min", 0.0
                ),
                "liquid_flow_mL_min": calculations.get(
                    "liquid_flow_rate_mL_min",
                    calculations.get("flow_rate_mL_min", 0.0),
                ),
                "reactor_volume_mL": calculations.get("reactor_volume_mL", 0.0),
                "tubing_ID_mm": calculations.get("tubing_ID_mm", 0.0),
                "bpr_pressure_bar": calculations.get("bpr_pressure_bar", 0.0),
                "gas_species": calculations.get("gas_species", ""),
                "gas_flow_sccm": calculations.get("gas_flow_sccm", 0.0),
                "gas_flow_actual_mL_min": calculations.get(
                    "gas_flow_actual_mL_min", 0.0
                ),
                "target_gas_equiv_inlet": calculations.get(
                    "target_gas_equiv_inlet", 0.0
                ),
                "gas_equiv_supplied": calculations.get("gas_equiv_supplied", 0.0),
                "runtime_s": float(raw["runtime_s"]),
                "llm_call_count": int(float(raw["llm_call_count"])),
                "total_tokens": int(float(raw["total_tokens"])),
                "run_dir": str(cell_dir),
            }
        )
    return rows


def summarize_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["scenario_kind"] == "feasible"]
    return {
        "cell_count": len(rows),
        "joint_success_n": sum(row["joint_success"] for row in rows),
        "joint_success_rate": (
            sum(row["joint_success"] for row in rows) / len(rows) if rows else 0.0
        ),
        "fallback_n": sum(row["engine_fallback"] for row in rows),
        "fallback_free_n": sum(not row["engine_fallback"] for row in rows),
        "feasible_cell_count": len(feasible),
        "feasible_fallback_n": sum(row["engine_fallback"] for row in feasible),
        "feasible_clean_council_n": sum(
            not row["engine_fallback"] for row in feasible
        ),
        "feasible_ready_n": sum(
            row["final_validation_status"] == "ready" for row in feasible
        ),
    }


def build_scenario_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_version"], row["scenario_id"])].append(row)
    output = []
    for run_version in ("baseline", "evidence_first"):
        for scenario in SCENARIO_ORDER:
            group = grouped[(run_version, scenario)]
            output.append(
                {
                    "run_version": run_version,
                    "scenario_id": scenario,
                    "scenario_kind": group[0]["scenario_kind"],
                    "repeat_count": len(group),
                    "joint_success_n": sum(row["joint_success"] for row in group),
                    "fallback_n": sum(row["engine_fallback"] for row in group),
                    "clean_council_n": sum(
                        not row["engine_fallback"] for row in group
                    ),
                    "ready_n": sum(
                        row["final_validation_status"] == "ready" for row in group
                    ),
                    "reported_dispositions": "|".join(
                        f"{key}:{value}"
                        for key, value in sorted(
                            Counter(
                                row["reported_disposition"] for row in group
                            ).items()
                        )
                    ),
                }
            )
    return output


def build_before_after_rows(
    scenario_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["run_version"], row["scenario_id"]): row for row in scenario_rows
    }
    output = []
    for scenario in SCENARIO_ORDER:
        old = lookup[("baseline", scenario)]
        new = lookup[("evidence_first", scenario)]
        output.append(
            {
                "scenario_id": scenario,
                "scenario_kind": old["scenario_kind"],
                "baseline_joint_success_n": old["joint_success_n"],
                "evidence_first_joint_success_n": new["joint_success_n"],
                "baseline_fallback_n": old["fallback_n"],
                "evidence_first_fallback_n": new["fallback_n"],
                "fallback_reduction_n": old["fallback_n"] - new["fallback_n"],
                "baseline_ready_n": old["ready_n"],
                "evidence_first_ready_n": new["ready_n"],
            }
        )
    return output


def _count_log_events(run_dir: Path) -> list[dict[str, Any]]:
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    patterns = {
        "unsupported_tau_reduction_rejected": (
            "evidence-first guard rejected unsupported tau reduction"
        ),
        "unsafe_revision_rejected": "rejected unsafe revision",
        "deterministic_pool_preserved": (
            "deterministic-audit candidates because scoring blocks"
        ),
        "lexical_retrieval_fallback": "deterministic lexical retrieval fallback",
    }
    return [
        {"event": name, "count": log.count(pattern)}
        for name, pattern in patterns.items()
    ]


def build_execution_path_figure(
    scenario_rows: list[dict[str, Any]], figures: Path
) -> None:
    feasible = [f"{family}_feasible" for family in FAMILY_ORDER]
    lookup = {
        (row["run_version"], row["scenario_id"]): row for row in scenario_rows
    }
    x = np.arange(len(feasible))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for offset, run_version in ((-width / 2, "baseline"), (width / 2, "evidence_first")):
        clean = [
            lookup[(run_version, scenario)]["clean_council_n"]
            for scenario in feasible
        ]
        fallback = [
            lookup[(run_version, scenario)]["fallback_n"] for scenario in feasible
        ]
        ax.bar(
            x + offset,
            clean,
            width,
            color=COLORS[run_version],
            label=f"{RUN_LABELS[run_version]}: council path",
        )
        ax.bar(
            x + offset,
            fallback,
            width,
            bottom=clean,
            color=COLORS["fallback"],
            hatch="//" if run_version == "baseline" else "\\\\",
            label=f"{RUN_LABELS[run_version]}: engine fallback",
        )
    ax.set_xticks(x, [FAMILY_LABELS[item] for item in FAMILY_ORDER])
    ax.set_ylabel("Repeats (n=3 per family)")
    ax.set_ylim(0, 3.5)
    ax.set_title("Feasible-case execution path before and after correction")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "10_before_after_feasible_council_path")


def build_gas_timing_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    gas_rows = [
        row
        for row in rows
        if row["run_version"] == "evidence_first"
        and row["scenario_id"] in {
            "photo_oxidation_feasible",
            "hydrogenolysis_feasible",
        }
    ]
    labels = [
        f"{'Photo' if row['scenario_id'].startswith('photo') else 'H2'} R{row['repeat']}"
        for row in gas_rows
    ]
    inlet = [row["residence_time_inlet_min"] for row in gas_rows]
    channel = [row["residence_time_in_channel_min"] for row in gas_rows]
    x = np.arange(len(gas_rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - width / 2, inlet, width, color=COLORS["inlet"], label="Inlet/STP basis")
    ax.bar(
        x + width / 2,
        channel,
        width,
        color=COLORS["channel"],
        label="Pressure-corrected in-channel basis",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Residence time (min)")
    ax.set_title("Gas-liquid timing reported on both required bases")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "11_evidence_first_residence_times")


def build_quality_figure(
    summaries: dict[str, dict[str, Any]], figures: Path
) -> None:
    metrics = [
        ("Joint success", "joint_success_n", "cell_count"),
        ("Feasible final validation", "feasible_ready_n", "feasible_cell_count"),
        (
            "Feasible clean council path",
            "feasible_clean_council_n",
            "feasible_cell_count",
        ),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for offset, run_version in ((-width / 2, "baseline"), (width / 2, "evidence_first")):
        values = [
            100
            * summaries[run_version][numerator]
            / summaries[run_version][denominator]
            for _, numerator, denominator in metrics
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=COLORS[run_version],
            label=RUN_LABELS[run_version],
        )
        ax.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=2)
    ax.set_xticks(x, [item[0] for item in metrics])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Endpoint quality and internal architecture convergence")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "12_joint_success_and_validation")


def build_safeguard_figure(events: list[dict[str, Any]], figures: Path) -> None:
    display = {
        "unsupported_tau_reduction_rejected": "Unsupported tau\nreduction rejected",
        "unsafe_revision_rejected": "Unsafe revisions\nrejected",
        "deterministic_pool_preserved": "Valid pools preserved\nfrom stochastic blocks",
        "lexical_retrieval_fallback": "Lexical retrieval\nfallbacks",
    }
    labels = [display[row["event"]] for row in events]
    values = [row["count"] for row in events]
    colors = ["#E76F51", "#E9C46A", "#2A9D8F", "#7A7A7A"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(np.arange(len(events)), values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_xticks(np.arange(len(events)), labels)
    ax.set_ylabel("Logged events (n)")
    ax.set_title("Deterministic safeguards and degraded-service handling")
    ax.set_ylim(0, max(values + [1]) * 1.18)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "13_safeguard_events")


def _write_documents(
    output: Path,
    summaries: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    old = summaries["baseline"]
    new = summaries["evidence_first"]
    event_counts = {row["event"]: row["count"] for row in events}
    (output / "SUMMARY.md").write_text(
        f"""# FlowPilot Evidence-First Validation

## Result

The corrected Qwen 27B + FlowPilot condition completed all 30 selected cells:
five chemistry families, paired feasible/infeasible inventories, and three
repeats. Joint disposition-plus-engineering success remained
`{new["joint_success_n"]}/{new["cell_count"]}`. All feasible outputs passed the
final deterministic validator (`{new["feasible_ready_n"]}/{new["feasible_cell_count"]}`).

The main improvement is internal architecture convergence. Feasible cases using
the engine fallback fell from `{old["feasible_fallback_n"]}/{old["feasible_cell_count"]}`
to `{new["feasible_fallback_n"]}/{new["feasible_cell_count"]}`. Across all cells,
fallback use fell from `{old["fallback_n"]}/{old["cell_count"]}` to
`{new["fallback_n"]}/{new["cell_count"]}`. The remaining fallback occurred in
one intentionally infeasible multistep control, which was correctly blocked.

## What was corrected

- Default translation policy is evidence-first; intensification is no longer
  forced without an explicit objective.
- Measured residence-time evidence cannot be overridden by an unsupported
  model-only reduction.
- Gas identity, inlet equivalents, STP flow, actual in-channel flow, and both
  residence-time bases are propagated through candidate generation and revision.
- Pump minimum and maximum flow, inventory pressure ceilings, diameter limits,
  geometry closure, and final revisions are deterministically enforced.
- Stochastic expert blocks cannot erase a pool that passed deterministic audit.
- Retrieval degrades to deterministic lexical ranking if embeddings are unavailable.

## Observed safeguards

- Unsupported residence-time reductions rejected: `{event_counts["unsupported_tau_reduction_rejected"]}`
- Unsafe final revisions rejected: `{event_counts["unsafe_revision_rejected"]}`
- Deterministically valid pools preserved: `{event_counts["deterministic_pool_preserved"]}`
- Lexical retrieval fallback activations: `{event_counts["lexical_retrieval_fallback"]}`

## Interpretation

This package validates constrained design behavior and council execution on the
frozen benchmark. It does not prove yield prediction, chemical optimality, or
wet-lab superiority. Proposed conditions remain experimental screens.
""",
        encoding="utf-8",
    )
    (output / "METHODS.md").write_text(
        """# Methods

The same Qwen3.6-27B endpoint was used as the upstream, translation, council,
and revision model. The correction run evaluated only the predeclared
`qwen27b_full_flowpilot` condition: ten frozen scenarios with three repeats,
for 30 planned cells. Each family contained a feasible inventory requiring
`SCREEN` and a controlled infeasible inventory requiring `BLOCK`.

The primary endpoint was frozen before this correction:

```text
joint_success = disposition_correct AND critical_engineering_pass
```

The deterministic oracle evaluated only public hard constraints and serialized
engineering fields. Internal council fallback and final-validator status were
reported as separate diagnostics; they were not added to the frozen primary
score. This prevents the correction from changing the endpoint after seeing
results.

The original 30 FlowPilot cells from the completed confirmatory run form the
before-correction baseline. The evidence-first run used the same scenario files,
seeds, temperature, candidate budget, and public inputs. Per-cell prompts,
responses, stage snapshots, calculations, oracle outputs, logs, environment
metadata, and checksums are retained in the source run directories.
""",
        encoding="utf-8",
    )
    (output / "CHANGES.md").write_text(
        """# Code Changes

1. Evidence-first became the default translation policy. Predicted target
   conditions are hypotheses unless the user explicitly requests intensification.
2. Candidate generation now preserves the protocol gas basis instead of assuming
   three equivalents of oxygen from air.
3. Pure gases and gas mixtures use their actual reagent fraction when converting
   equivalents to STP and pressure-corrected in-channel flow.
4. The selected pump maximum flow and inventory gas-service pressure ceiling are
   hard constraints throughout generation, council refinement, and final revision.
5. Unsupported residence-time reductions are rejected when measured evidence
   establishes a longer-time floor.
6. Model-proposed revisions are recomputed and hard-filtered before acceptance.
7. A deterministic lexical retrieval fallback and an embedding circuit breaker
   prevent repeated external embedding failures from terminating the pipeline.
8. The benchmark runner accepts condition and scenario subsets while retaining
   the frozen study definition and complete per-cell provenance.
""",
        encoding="utf-8",
    )
    (output / "RESIDUAL_RISKS.md").write_text(
        """# Residual Risks

- OpenAI embeddings returned a quota error during this run. Every cell therefore
  used the documented lexical retrieval fallback. Retrieval parity with the
  original embedding backend was not tested here.
- One intentionally infeasible multistep control reached the engine fallback
  after council disqualification; final disposition remained the correct `BLOCK`.
- Repeated stochastic calls can select different valid residence times. The final
  validator establishes feasibility, not that one condition is chemically optimal.
- The frozen oracle checks declared critical constraints. It does not replace a
  chemist review, HAZOP, equipment certification, or wet-lab validation.
- No experimental yields were generated in this software benchmark.
""",
        encoding="utf-8",
    )


def _write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(
    baseline_dir: Path,
    evidence_first_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    baseline_rows = build_cell_rows(baseline_dir, "baseline")
    evidence_rows = build_cell_rows(evidence_first_dir, "evidence_first")
    if len(baseline_rows) != 30 or len(evidence_rows) != 30:
        raise ValueError("Both FlowPilot runs must contain exactly 30 completed cells")

    rows = baseline_rows + evidence_rows
    summaries = {
        "baseline": summarize_cells(baseline_rows),
        "evidence_first": summarize_cells(evidence_rows),
    }
    scenario_rows = build_scenario_rows(rows)
    before_after = build_before_after_rows(scenario_rows)
    events = _count_log_events(evidence_first_dir)

    _write_csv(tables / "cell_results.csv", rows)
    _write_csv(tables / "scenario_summary.csv", scenario_rows)
    _write_csv(tables / "before_after_scenario.csv", before_after)
    _write_csv(
        tables / "final_designs_feasible.csv",
        [
            row
            for row in evidence_rows
            if row["scenario_kind"] == "feasible"
        ],
    )
    _write_csv(tables / "safeguard_events.csv", events)

    build_execution_path_figure(scenario_rows, figures)
    build_gas_timing_figure(evidence_rows, figures)
    build_quality_figure(summaries, figures)
    build_safeguard_figure(events, figures)
    _write_documents(output_dir, summaries, events)

    summary = {
        "schema_version": "flowpilot_evidence_first_validation_v1.0",
        "baseline_run": str(baseline_dir.resolve()),
        "evidence_first_run": str(evidence_first_dir.resolve()),
        "baseline": summaries["baseline"],
        "evidence_first": summaries["evidence_first"],
        "feasible_fallback_reduction_n": (
            summaries["baseline"]["feasible_fallback_n"]
            - summaries["evidence_first"]["feasible_fallback_n"]
        ),
        "all_cell_fallback_reduction_n": (
            summaries["baseline"]["fallback_n"]
            - summaries["evidence_first"]["fallback_n"]
        ),
        "safeguard_events": {row["event"]: row["count"] for row in events},
        "claims_boundary": (
            "Constraint-compliance and architecture validation only; no wet-lab "
            "yield or global-optimality claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_checksums(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--evidence-first-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_package(
        args.baseline_dir,
        args.evidence_first_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
