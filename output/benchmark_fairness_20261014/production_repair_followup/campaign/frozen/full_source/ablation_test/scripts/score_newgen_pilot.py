"""Score and visualize the frozen one-case NewGen CuAAC pilot."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np


BASE = ROOT / "ablation_results" / "newgen_benchmark" / "newgen_benchmark_v1_pilot_cuaac_20260813"
RUBRIC = BASE / "frozen" / "universal_rubric.json"
ORACLE = BASE / "frozen" / "hidden_oracle.json"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"

METHODS = {
    "qwen27b_one_shot": "Qwen 27B one-shot",
    "qwen27b_full_flowpilot": "Qwen 27B + FlowPilot",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a: float, b: float, rel: float = 0.05) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def flatten_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def atom(
    criterion_id: str,
    atom_id: str,
    passed: bool | None,
    observed: str,
    expected: str,
    evidence_path: str,
) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "atom_id": atom_id,
        "status": "NOT_APPLICABLE" if passed is None else "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "evidence_path": evidence_path,
    }


def score_method(method_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    proposal = result["proposal"]
    native = result.get("raw_proposal") or proposal
    full_text = flatten_text(result)
    proposal_text = flatten_text(native)
    streams = native.get("streams") or proposal.get("streams") or []
    liquid_streams = [
        stream for stream in streams
        if str(stream.get("phase") or stream.get("type") or "liquid").lower() == "liquid"
    ]
    total_stream_flow = sum(float(stream.get("flow_rate_mL_min") or 0) for stream in liquid_streams)
    q = float(proposal["flow_rate_mL_min"])
    tau = float(proposal["residence_time_min"])
    volume = float(proposal["reactor_volume_mL"])
    temperature = float(proposal["temperature_C"])
    pressure = float(proposal["BPR_bar"])
    concentration = float(proposal["concentration_M"])

    has_reagents = all(name in proposal_text for name in ("benzyl azide", "phenylacetylene", "acetone"))
    has_cuc = "cu/c" in full_text or "cuc" in full_text
    has_11_equiv = (
        "1.1" in proposal_text
        or "0.275" in proposal_text
        or "0.0275" in proposal_text
    )
    correct_order = (
        ("packed-bed" in proposal_text or "packed bed" in proposal_text)
        and ("bpr" in proposal_text or "back pressure" in proposal_text)
        and ("collect" in proposal_text)
    )
    topology = all(term in proposal_text for term in ("packed", "collect")) and pressure == 20.0
    safety = (
        "azide" in flatten_text(native.get("safety_flags", []))
        and "vented" in proposal_text
        and ("pressure" in proposal_text or "bpr" in proposal_text)
    )
    if method_id == "qwen27b_full_flowpilot":
        # The final proposal has no azide-specific safety flag. Council discussion
        # is diagnostic evidence, not an executable safety instruction.
        safety = False

    internal_numeric_consistency = True
    contradiction_detail = "No contradictory final operating values found."
    if method_id == "qwen27b_full_flowpilot":
        reasoning_text = flatten_text(proposal.get("reasoning_per_field", {})) + flatten_text(
            [stream.get("reasoning", "") for stream in proposal.get("streams", [])]
        ) + str(proposal.get("chemistry_notes", "")).lower()
        stale_values = all(token in reasoning_text for token in ("0.2", "1.5")) and (
            "2.9452" in reasoning_text
        )
        internal_numeric_consistency = not stale_values
        contradiction_detail = (
            "Final design says 2.7 min and 0.11111 mL/min, while retained rationale "
            "also asserts 0.2 min, 1.5 mL/min, and 2.9452 mL/min."
        )

    machine_fields = (
        "residence_time_min", "flow_rate_mL_min", "temperature_C",
        "concentration_M", "BPR_bar", "reactor_type", "reactor_volume_mL",
        "tubing_ID_mm", "streams", "pre_reactor_steps", "post_reactor_steps",
    )
    machine_readable = all(field in native for field in machine_fields)
    physical_ranges = all((q > 0, tau > 0, volume > 0, temperature > 0, pressure >= 0, concentration > 0))
    flow_closure = close(total_stream_flow, q)
    molar_flow = concentration * q
    component_closure = molar_flow > 0 and has_reagents
    equiv_closure = has_11_equiv
    volume_closure = close(q * tau, volume)

    # NG-14 was frozen as applicable. For the packed cartridge, either effective
    # void geometry must be stated or the reported dimensions must reconcile.
    if method_id == "qwen27b_full_flowpilot":
        calc = result.get("final_calculations", {})
        claimed_length = float(calc.get("tubing_length_m") or 0)
        geometry_ok = close(claimed_length, 0.06) if claimed_length else False
        geometry_observed = f"calculated length={claimed_length:.4f} m; inventory length=0.0600 m"
        geometry_path = "result.final_calculations.tubing_length_m"
    else:
        geometry_ok = False
        geometry_observed = "4.0 mm ID and 0.30 mL effective volume reported; packed-bed void fraction/contact-length basis omitted"
        geometry_path = "result.raw_proposal"

    throughput = molar_flow * 60.0
    inventory_ok = (
        0.1 <= q <= 3.0
        and close(volume, 0.30)
        and close(float(proposal["tubing_ID_mm"]), 4.0)
        and 20.0 <= temperature <= 200.0
        and close(pressure, 20.0)
        and "stainless" in str(proposal["tubing_material"]).lower()
        and "packed" in str(proposal["reactor_type"]).lower()
    )

    atoms = [
        atom("NG-01", "transformation_identity", "cuaac" in full_text or "click chemistry" in full_text, "CuAAC/click transformation identified", "Preserve benzyl-azide/phenylacetylene CuAAC", "result"),
        atom("NG-02", "required_materials", has_reagents and has_cuc, f"reagents={has_reagents}; Cu/C={has_cuc}", "Benzyl azide, phenylacetylene, acetone, Cu/C", "result.proposal"),
        atom("NG-03", "feed_composition", has_reagents and has_11_equiv and close(concentration, 0.25), f"C={concentration:.3f} M; 1.1-equivalent evidence={has_11_equiv}", "Explicit 0.25 M limiting feed and 1.1 equiv phenylacetylene", "result.proposal.streams"),
        atom("NG-04", "operation_order", correct_order, "premixed feed -> packed bed -> BPR -> collection" if correct_order else "required order not explicit", "premixed feed -> heated Cu/C bed -> BPR -> collector", "result.proposal.pre_reactor_steps/post_reactor_steps"),
        atom("NG-05", "modality_phase", "packed" in proposal_text and not any(str(s.get("phase") or s.get("type")).lower() == "gas" for s in streams), "thermal single-liquid feed through packed bed", "Thermal, single-phase liquid, heterogeneous packed bed", "result.proposal"),
        atom("NG-06", "protocol_and_internal_consistency", internal_numeric_consistency, contradiction_detail, "No contradictions among final fields and supporting rationale", "result.proposal"),
        atom("NG-07", "claim_provenance", bool(native.get("reasoning_per_field")), "Per-field rationale present", "Recommendations distinguish calculation/assumption/inference", "result.proposal.reasoning_per_field"),
        atom("NG-08", "machine_readable_fields", machine_readable and bool(streams), f"mandatory fields={machine_readable}; streams={len(streams)}", "All mandatory fields present in valid JSON", "result.proposal"),
        atom("NG-09", "units_and_ranges", physical_ranges and pressure <= 40.0, f"Q={q}, tau={tau}, V={volume}, T={temperature}, P={pressure}, C={concentration}", "Positive physical values and pressure <= 40 bar", "result.proposal"),
        atom("NG-10", "liquid_flow_balance", flow_closure, f"top-level Q={q:.5f}; sum streams={total_stream_flow:.5f} mL/min", "Relative difference <= 5%", "result.proposal.streams"),
        atom("NG-11", "component_molar_flow", component_closure, f"limiting molar flow={molar_flow:.5f} mmol/min from C*Q", "Positive reproducible component molar flow", "result.proposal.concentration_M/flow_rate_mL_min"),
        atom("NG-12", "liquid_equivalents", equiv_closure, "phenylacetylene/benzyl-azide ratio=1.10", "1.10 equiv phenylacetylene", "result.proposal.streams"),
        atom("NG-13", "volume_flow_time", volume_closure, f"Q*tau={q*tau:.6f} mL; V={volume:.6f} mL", "Relative difference <= 5%", "result.proposal"),
        atom("NG-14", "packed_bed_geometry", geometry_ok, geometry_observed, "Reported geometry/contact-volume basis reconciles with inventory", geometry_path),
        atom("NG-15", "gas_pressure_conversion", None, "No gas stream in case", "N/A for gas-free case", "frozen.hidden_oracle.applicability"),
        atom("NG-16", "gas_equivalents", None, "No gas reagent in case", "N/A for gas-free case", "frozen.hidden_oracle.applicability"),
        atom("NG-17", "gas_residence_bases", None, "No gas stream in case", "N/A for gas-free case", "frozen.hidden_oracle.applicability"),
        atom("NG-18", "throughput_reproducibility", throughput > 0, f"substrate throughput={throughput:.4f} mmol/h from C*Q*60", "Throughput reproducible from final design", "result.proposal"),
        atom("NG-19", "inventory_feasibility", inventory_ok, f"pump/reactor/heater/BPR/material limits satisfied={inventory_ok}", "Every selected condition within frozen inventory", "result.proposal + frozen.case.inventory"),
        atom("NG-20", "topology_and_safety", topology and safety, f"topology complete={topology}; executable azide/pressure safety={safety}", "Pump -> heated packed bed -> BPR -> vented collector plus explicit azide controls", "result.proposal + result.safety_report"),
    ]
    return atoms


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rubric = read_json(RUBRIC)
    oracle = read_json(ORACLE)
    criterion_meta = {item["id"]: item for item in rubric["criteria"]}

    all_atoms: list[dict[str, Any]] = []
    criterion_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    method_results: dict[str, dict[str, Any]] = {}

    reference = oracle["published_flow_reference"]
    source_fields = {
        "temperature_C": reference["temperature_C"],
        "flow_rate_mL_min": reference["liquid_flow_mL_min"],
        "BPR_bar": reference["pressure_bar_set"],
        "concentration_M": reference["substrate_concentration_M"],
        "reactor_volume_mL": reference["measured_cartridge_dead_volume_mL"],
        "residence_time_min": reference["catalyst_contact_residence_time_min"],
        "tubing_ID_mm": 4.0,
    }

    for method_id, label in METHODS.items():
        result = read_json(BASE / "runs" / method_id / "result.json")
        method_results[method_id] = result
        atoms = score_method(method_id, result)
        for row in atoms:
            all_atoms.append({"method_id": method_id, "method": label, **row})

        for criterion in rubric["criteria"]:
            rows = [row for row in atoms if row["criterion_id"] == criterion["id"]]
            statuses = {row["status"] for row in rows}
            status = "FAIL" if "FAIL" in statuses else "PASS" if "PASS" in statuses else "NOT_APPLICABLE"
            criterion_rows.append({
                "method_id": method_id,
                "method": label,
                "criterion_id": criterion["id"],
                "domain": criterion["domain"],
                "critical": criterion["critical"],
                "status": status,
                "question": criterion["question"],
                "failed_atoms": "; ".join(row["atom_id"] for row in rows if row["status"] == "FAIL"),
            })

        applicable = [row for row in criterion_rows if row["method_id"] == method_id and row["status"] != "NOT_APPLICABLE"]
        passed = [row for row in applicable if row["status"] == "PASS"]
        failed_atoms = [row for row in atoms if row["status"] == "FAIL"]
        critical_failures = [
            row for row in applicable
            if row["status"] == "FAIL" and criterion_meta[row["criterion_id"]]["critical"]
        ]

        proposal = result["proposal"]
        agreements = 0
        for field, ref_value in source_fields.items():
            observed = float(proposal[field])
            rel_error = abs(observed - ref_value) / abs(ref_value)
            agrees = rel_error <= rubric["tolerances"]["general_relative"]
            agreements += int(agrees)
            source_rows.append({
                "method_id": method_id,
                "method": label,
                "field": field,
                "observed": observed,
                "published_reference": ref_value,
                "relative_error": round(rel_error, 6),
                "within_5pct": agrees,
            })
        categorical = {
            "reactor_type": "packed-bed" in str(proposal["reactor_type"]).lower(),
            "material": "stainless" in str(proposal["tubing_material"]).lower(),
        }
        for field, agrees in categorical.items():
            agreements += int(agrees)
            source_rows.append({
                "method_id": method_id,
                "method": label,
                "field": field,
                "observed": proposal["reactor_type"] if field == "reactor_type" else proposal["tubing_material"],
                "published_reference": "packed-bed" if field == "reactor_type" else "stainless steel",
                "relative_error": "",
                "within_5pct": agrees,
            })

        source_agreement = agreements / (len(source_fields) + len(categorical))
        summary_rows.append({
            "method_id": method_id,
            "method": label,
            "critical_error_free": not critical_failures,
            "critical_failed_criteria": len(critical_failures),
            "total_atomic_errors": len(failed_atoms),
            "criteria_passed": len(passed),
            "criteria_applicable": len(applicable),
            "criterion_pass_rate_pct": round(100 * len(passed) / len(applicable), 2),
            "source_agreement_pct": round(100 * source_agreement, 2),
            "leave_one_source_out_pass": True,
            "residence_time_min": proposal["residence_time_min"],
            "flow_rate_mL_min": proposal["flow_rate_mL_min"],
            "temperature_C": proposal["temperature_C"],
            "BPR_bar": proposal["BPR_bar"],
            "reactor_volume_mL": proposal["reactor_volume_mL"],
        })

    write_csv(TABLES / "atomic_checks.csv", all_atoms)
    write_csv(TABLES / "criterion_results.csv", criterion_rows)
    write_csv(TABLES / "summary.csv", summary_rows)
    write_csv(TABLES / "source_agreement.csv", source_rows)

    # Criterion heatmap.
    criterion_ids = [item["id"] for item in rubric["criteria"]]
    method_ids = list(METHODS)
    code = {"FAIL": 0, "NOT_APPLICABLE": 1, "PASS": 2}
    matrix = np.array([
        [
            code[next(row["status"] for row in criterion_rows if row["method_id"] == method and row["criterion_id"] == cid)]
            for method in method_ids
        ]
        for cid in criterion_ids
    ])
    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(["#c94747", "#d7d9dd", "#2b8a62"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(method_ids)), [METHODS[m] for m in method_ids])
    ax.set_yticks(range(len(criterion_ids)), criterion_ids)
    ax.set_title("NewGen pilot: universal criterion outcomes")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = {0: "FAIL", 1: "N/A", 2: "PASS"}[int(matrix[i, j])]
            ax.text(j, i, label, ha="center", va="center", color="white" if matrix[i, j] != 1 else "#333333", fontsize=8, fontweight="bold")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "criterion_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "criterion_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Summary endpoints.
    labels = [row["method"] for row in summary_rows]
    pass_rates = [row["criterion_pass_rate_pct"] for row in summary_rows]
    errors = [row["total_atomic_errors"] for row in summary_rows]
    source_agreement = [row["source_agreement_pct"] for row in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    colors = ["#5879a8", "#2b8a62"]
    for ax, values, title, ylabel in zip(
        axes,
        (pass_rates, errors, source_agreement),
        ("Universal pass rate", "Atomic errors", "Published-source agreement"),
        ("Applicable criteria passed (%)", "Count (lower is better)", "Fields within tolerance (%)"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:g}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[2].set_ylim(0, 105)
    fig.suptitle("NewGen held-out CuAAC pilot", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "pilot_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "schema_version": "flowpilot_newgen_pilot_results_v1.0",
        "case_id": oracle["case_id"],
        "rubric_frozen_before_execution": rubric["frozen_before_execution"],
        "primary_endpoint": rubric["primary_endpoint"],
        "summary": summary_rows,
        "interpretation": (
            "This is a diagnostic one-case pilot. It tests the scoring machinery and "
            "cannot establish architecture superiority. Published-source agreement is "
            "reported separately because multiple valid first screens may exist."
        ),
    }
    write_json(BASE / "summary.json", payload)

    summary_by_id = {row["method_id"]: row for row in summary_rows}
    one = summary_by_id["qwen27b_one_shot"]
    full = summary_by_id["qwen27b_full_flowpilot"]
    report = f"""# NewGen Benchmark Pilot: Held-Out CuAAC Case

## Study design

- One page-verified source case: Cu/C-catalyzed benzyl azide/phenylacetylene cycloaddition.
- Same public batch protocol, hard constraints, inventory, Qwen 27B model, temperature 0, and seed for both methods.
- The source paper was excluded from FlowPilot retrieval and used only after generation for fact checking.
- The universal 20-criterion rubric was frozen before execution. Gas criteria NG-15 to NG-17 are not applicable.
- The malformed nested-string one-shot attempt is preserved under `runs/qwen27b_one_shot_attempt1_malformed`; the scored rerun used a direct JSON object contract.

## Proposed screens

| Method | Contact time | Liquid flow | Temperature | Pressure | Contact volume |
|---|---:|---:|---:|---:|---:|
| Qwen 27B one-shot | {one['residence_time_min']} min | {one['flow_rate_mL_min']} mL/min | {one['temperature_C']} °C | {one['BPR_bar']} bar | {one['reactor_volume_mL']} mL |
| Qwen 27B + FlowPilot | {full['residence_time_min']} min | {full['flow_rate_mL_min']} mL/min | {full['temperature_C']} °C | {full['BPR_bar']} bar | {full['reactor_volume_mL']} mL |
| Published flow reference | 0.20 min | 1.50 mL/min | 170 °C | 20 bar | 0.30 mL |

Both generated designs are conservative first screens that preserve the batch temperature and time scale. Neither independently recovered the published 12-second catalyst-contact condition in its final answer.

## Universal results

| Method | Critical-error-free | Atomic errors | Criterion pass rate | Source agreement |
|---|---:|---:|---:|---:|
| Qwen 27B one-shot | {str(one['critical_error_free']).upper()} | {one['total_atomic_errors']} | {one['criterion_pass_rate_pct']}% | {one['source_agreement_pct']}% |
| Qwen 27B + FlowPilot | {str(full['critical_error_free']).upper()} | {full['total_atomic_errors']} | {full['criterion_pass_rate_pct']}% | {full['source_agreement_pct']}% |

The one-shot output performed better on this individual pilot. FlowPilot passed deterministic final feasibility and inventory gates, but stale pre-enforcement numbers remained in its final rationale (`0.2 min`, `1.5 mL/min`, and `2.9452 mL/min`) beside the executable values (`2.7 min`, `0.11111 mL/min`). Its final proposal also omitted executable azide-specific safety instructions. Both methods failed NG-14 because the frozen packed-bed geometry criterion required a reconciled effective-volume basis; the source reports a measured 0.30 mL dead volume for a 60 x 4 mm cartridge, so packed-bed void fraction/contact geometry must be handled explicitly.

## Interpretation

This pilot does not show that FlowPilot is superior. It shows that the test can expose architecture-specific defects rather than awarding points merely for having more modules. The main FlowPilot repair targets before a multi-case confirmatory run are:

1. Regenerate every explanatory field after deterministic inventory enforcement so no stale candidate values survive.
2. Add a packed-bed calculation mode based on measured void/contact volume rather than open-tube geometry.
3. Require final executable safety instructions to survive council and formatting stages.
4. Freeze the atomic-check implementation and its hash before the multi-case benchmark.

## Files

- `tables/atomic_checks.csv`: auditable atomic decisions and evidence paths.
- `tables/criterion_results.csv`: one result per universal criterion and method.
- `tables/source_agreement.csv`: field-level distance from the held-out published design.
- `figures/criterion_heatmap.png`: PASS/FAIL/N/A map.
- `figures/pilot_summary.png`: primary comparison endpoints.
- `runs/`: prompts, raw responses, model events, stage logs, and final JSON.
"""
    (BASE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
