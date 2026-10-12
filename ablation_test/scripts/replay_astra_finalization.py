#!/usr/bin/env python3
"""Offline regression replay; never changes or rescores the original campaign."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flora_translate.design_realizer import realize_executable_design
from flora_translate.design_calculator import DesignCalculator
from flora_translate.design_disposition import apply_design_disposition_gate
from flora_translate.final_design_contract import (
    _build_executable_process_graph, _gas_composition_and_equivalent_issues,
    build_final_design_contract,
)
from flora_translate.final_engineering import calculate_final_stages
from flora_translate.main import _build_translate_topology, _topology_matches_serialized_proposal
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.topology_semantics import normalize_topology_semantics


def replay_contract(result, proposal, batch, plan, inventory, constraints, validation, allocation, topology):
    """Rebuild artifacts and disposition; do not inherit an old pass/block decision."""
    calculations = asdict(DesignCalculator().run(
        batch, chemistry_plan=plan, proposal=proposal, inventory=inventory,
        target_flow_rate_mL_min=proposal.flow_rate_mL_min,
        target_tubing_ID_mm=proposal.tubing_ID_mm,
        target_residence_time_min=proposal.residence_time_min,
    ))
    calculations.update({
        "flow_rate_mL_min": proposal.flow_rate_mL_min,
        "reactor_volume_mL": proposal.reactor_volume_mL,
        "tubing_ID_mm": proposal.tubing_ID_mm,
        "residence_time_min": proposal.residence_time_min,
        "residence_time_inlet_min": proposal.residence_time_inlet_min,
        "residence_time_in_channel_min": proposal.residence_time_in_channel_min,
        "stage_calculations": list(proposal.stage_parameters or []),
    })
    calculations.update(proposal.multiphase_metrics or {})
    fresh = {
        "proposal": proposal.model_dump(), "batch_record": batch.model_dump(),
        "chemistry_plan": plan.model_dump(), "inventory_snapshot": inventory.model_dump(),
        "intake_package": deepcopy(result.get("intake_package")),
        "process_topology": topology.model_dump(), "inventory_allocation": allocation,
        "design_calculations": calculations,
        "final_stage_engineering": calculate_final_stages(proposal, batch, plan, inventory),
    }
    apply_design_disposition_gate(
        fresh, proposal=proposal, final_validation=validation, inventory=inventory,
        batch_record=batch, chemistry_plan=plan, hard_constraints=constraints,
        council_safety_report=deepcopy(result.get("council_safety_report_diagnostic") or result.get("safety_report")),
    )
    fresh["final_design"] = build_final_design_contract(fresh)
    return fresh


def replay(source, output, *, expected_runs=9, full_contract=False):
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a new output folder; existing replay evidence is immutable")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((source / "generation").glob("*/astra_flowpilot/repeat_*/result.json")):
        raw = path.read_bytes()
        result = json.loads(raw)
        batch = BatchRecord.model_validate(result["batch_record"])
        plan = ChemistryPlan.model_validate(result["chemistry_plan"])
        inventory = LabInventory.model_validate(result["inventory_snapshot"])
        constraints = (result.get("design_disposition") or {}).get("requirements", {}).get("source_text", "")
        proposal, realization, validation = realize_executable_design(
            FlowProposal.model_validate(result["proposal"]), batch_record=batch,
            chemistry_plan=plan, inventory=inventory, hard_constraints=constraints,
        )
        topology, allocation = compile_inventory_topology(
            _build_translate_topology(proposal, plan, batch, inventory),
            proposal=proposal, inventory=inventory,
        )
        topology = normalize_topology_semantics(topology)
        validation["checks"]["inventory_topology_assignment_complete"] = bool(allocation["checks"]["all_required_operations_assigned"])
        validation["checks"]["topology_matches_serialized_design"] = _topology_matches_serialized_proposal(topology, proposal)
        errors = _gas_composition_and_equivalent_issues(proposal.model_dump())
        graph = None
        try:
            graph = _build_executable_process_graph(topology.model_dump()).model_dump()
        except ValueError as exc:
            errors.append({"graph_error": str(exc)})
        rebuilt = None
        if full_contract:
            if proposal.stage_parameters:
                raise ValueError("This replay harness covers only the frozen single-stage Astra cases")
            rebuilt = replay_contract(result, proposal, batch, plan, inventory, constraints, validation, allocation, topology)
            errors.extend(rebuilt["final_design"]["consistency"]["issues"])
            if rebuilt["final_design"]["status"] != "executable":
                errors.append({"contract_error": "Final design remains blocked", "disposition": rebuilt["design_disposition"]})
        passed = all(validation["checks"].values()) and allocation["checks"]["all_required_operations_assigned"] and graph is not None and not errors
        row = {"case": path.parts[-4], "repeat": path.parts[-2], "source": str(path),
               "source_sha256": hashlib.sha256(raw).hexdigest(),
               "original_status": result["final_design"]["status"], "passed": passed,
               "errors": errors, "validation": validation, "allocation": allocation,
               "proposal": proposal.model_dump(), "graph": graph, "realization": realization,
               "rebuilt_result": rebuilt,
               "scope": "Offline finalization regression only. No model calls or judging; not an experimental approval."}
        target = output / (row["case"] + "_" + row["repeat"] + ".json")
        target.write_text(json.dumps(row, indent=2, default=str))
        rows.append({k: row[k] for k in ["case", "repeat", "original_status", "passed", "errors", "source_sha256"]})
    summary = {"runs": rows, "passed": sum(r["passed"] for r in rows), "total": len(rows),
               "full_contract": full_contract, "expected_runs": expected_runs}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(rows) != expected_runs or not all(r["passed"] for r in rows):
        raise SystemExit("Replay failed; inspect saved diagnostics before live generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-runs", type=int, default=9)
    parser.add_argument("--full-contract", action="store_true")
    args = parser.parse_args()
    replay(args.source.resolve(), args.output.resolve(), expected_runs=args.expected_runs, full_contract=args.full_contract)
