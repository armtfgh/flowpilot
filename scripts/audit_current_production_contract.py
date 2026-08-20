#!/usr/bin/env python3
"""Audit fresh production benchmark runs with the canonical release contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate.final_design_contract import build_final_design_contract
from flora_translate.schemas import ProcessTopology
from flora_translate.topology_semantics import normalize_topology_semantics


IDENTITIES = {
    "cuaac": "cuaac",
    "hydrogenolysis": "hydrogenolysis/debenzylation",
    "multistep": "oxidative amidation",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare(run_dir: Path) -> dict:
    result = _read(run_dir / "result.json")
    public = _read(run_dir / "input_public.json")
    topology = result.get("process_topology") or {}
    if not topology.get("unit_operations"):
        topology = result.get("diagnostic_topology") or topology
    result["process_topology"] = normalize_topology_semantics(
        ProcessTopology.model_validate(topology)
    ).model_dump(mode="json")
    result["diagram_render_manifest"] = {}
    result["batch_record"] = {
        "raw_text": public["protocol"],
        "reaction_description": public["protocol"],
    }
    result["inventory_snapshot"] = deepcopy(public["inventory"])
    # Remove only the semantic publication verdict written by the previous
    # contract pass. Numerical/inventory validation remains frozen.
    validation = result.get("final_validation") or {}
    checks = validation.get("checks") or {}
    semantic_check_names = {
        "chemistry_identity_preserved",
        "component_stoichiometry_complete",
        "topology_phase_consistent",
        "control_edges_not_in_process_path",
        "safety_controls_complete",
        "procedure_complete",
        "no_unsupported_operations",
        "diagram_matches_canonical_topology",
        "residence_time_basis_unambiguous",
        "mixing_regime_matches_topology",
        "final_design_contract_consistency",
    }
    for name in semantic_check_names:
        checks.pop(name, None)
    validation["checks"] = checks
    validation["status"] = "ready"
    validation["unresolved_reasons"] = [
        name for name, passed in checks.items() if passed is False
    ]
    result["final_validation"] = validation
    pre_contract_disposition = validation.get("design_disposition") or {}
    result["design_disposition"] = deepcopy(pre_contract_disposition)
    result["recommended_disposition"] = (
        pre_contract_disposition.get("recommended_disposition") or "SCREEN"
    )
    result.pop("final_design", None)
    return result


def _add_authorities(result: dict, case: str) -> None:
    result["intake_package"] = {
        "chemistry_identity_confirmation": {
            "transformation_family": IDENTITIES[case],
            "confirmed": True,
            "source": "frozen_benchmark_chemist_confirmation",
        }
    }
    accessories = result["inventory_snapshot"].setdefault("safety_accessories", [])
    accessories.append(
        {
            "equipment_id": f"benchmark_{case}_containment",
            "name": "Declared hazard-compatible shield and secondary containment",
            "type": "safety enclosure",
            "capabilities": ["shield_or_containment"],
        }
    )
    if case == "hydrogenolysis":
        accessories.append(
            {
                "equipment_id": "benchmark_h2_check_valve",
                "name": "Hydrogen-service non-return check valve",
                "type": "check valve",
                "capabilities": ["backflow_prevention"],
            }
        )


def _codes(contract: dict) -> list[str]:
    return [item["code"] for item in contract["consistency"]["issues"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = (args.output or campaign / "canonical_v2_postfix_audit").resolve()
    contracts = output / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(campaign.glob("*_qwen27b_full_flowpilot")):
        case = run_dir.name.split("_qwen27b", 1)[0]
        prepared = _prepare(run_dir)
        exact = build_final_design_contract(prepared)
        sensitivity_result = deepcopy(prepared)
        _add_authorities(sensitivity_result, case)
        sensitivity = build_final_design_contract(sensitivity_result)
        (contracts / f"{case}_exact_inventory.json").write_text(
            json.dumps(exact, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (contracts / f"{case}_complete_authority_inventory.json").write_text(
            json.dumps(sensitivity, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        rows.append(
            {
                "case": case,
                "exact_input_status": exact["status"],
                "exact_input_blockers": ";".join(_codes(exact)),
                "complete_authority_inventory_status": sensitivity["status"],
                "remaining_candidate_blockers": ";".join(_codes(sensitivity)),
                "canonical_sha256": sensitivity.get("canonical_sha256") or "",
            }
        )

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "flowpilot_fresh_production_contract_audit_v1.0",
        "campaign": str(campaign),
        "candidate_count": len(rows),
        "exact_input_executable": sum(row["exact_input_status"] == "executable" for row in rows),
        "complete_authority_inventory_executable": sum(
            row["complete_authority_inventory_status"] == "executable" for row in rows
        ),
        "rows": rows,
        "interpretation": {
            "exact_input": "Uses the inventory and authority fields exactly as supplied to the live run.",
            "complete_authority_inventory": (
                "Sensitivity analysis only: adds frozen chemist identity confirmation and "
                "the hazard-specific safety accessories absent from the old benchmark inventory."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    lines = [
        "# Fresh Canonical v2 Benchmark Audit",
        "",
        f"Campaign: `{campaign}`",
        "",
        f"- Executable with exact benchmark inputs: {summary['exact_input_executable']}/{len(rows)}",
        "- Executable after complete-authority/inventory sensitivity: "
        f"{summary['complete_authority_inventory_executable']}/{len(rows)}",
        "",
        "The sensitivity condition is not a replacement for measured inventory. It separates "
        "missing benchmark metadata from defects in the generated candidate.",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        blockers = row["remaining_candidate_blockers"] or "none"
        lines.append(
            f"- `{row['case']}`: exact `{row['exact_input_status']}`; sensitivity "
            f"`{row['complete_authority_inventory_status']}`; remaining `{blockers}`."
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
