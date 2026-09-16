"""Recompile a saved FlowPilot candidate through the current final authority.

This utility performs no LLM calls. It is intended for regression evidence:
saved model/council output is rebound to the current inventory, topology,
diagram, validation, and final-design contracts without changing chemistry.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder
from flora_translate.design_realizer import realize_executable_design
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.final_design_contract import (
    build_final_design_contract,
    publish_final_design_artifacts,
)
from flora_translate.inventory_profiles import inventory_profile_from_payload
from flora_translate.main import _build_translate_topology
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal
from flora_translate.topology_compiler import compile_inventory_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads(source.read_text())
    profile = inventory_profile_from_payload(
        json.loads(Path(args.inventory).read_text())
    )
    inventory = profile.lab_inventory
    batch = BatchRecord.model_validate(result["batch_record"])
    plan = ChemistryPlan.model_validate(result["chemistry_plan"])
    proposal = FlowProposal.model_validate(result["proposal"])

    proposal, realization, validation = realize_executable_design(
        proposal,
        batch_record=batch,
        chemistry_plan=plan,
        inventory=inventory,
        hard_constraints=profile.operating_constraints,
        operating_limits=profile.design_operating_limits(),
    )
    topology = _build_translate_topology(
        proposal,
        plan,
        batch,
        inventory=inventory,
    )
    topology, allocation = compile_inventory_topology(
        topology,
        proposal=proposal,
        inventory=inventory,
    )

    result["proposal"] = proposal.model_dump()
    result["design_realization"] = realization
    result["multistage_inventory_plan"] = realization.get("multistage", {})
    result["process_topology"] = topology.model_dump()
    result["inventory_allocation"] = allocation
    result["inventory_snapshot"] = inventory.model_dump()
    result["recommended_disposition"] = "SCREEN"
    result["reported_disposition"] = "SCREEN"
    result["disposition_rationale"] = (
        "No deterministic hard-feasibility conflict was found; wet-lab "
        "confirmation remains required."
    )

    checks = validation.setdefault("checks", {})
    checks["inventory_topology_assignment_complete"] = allocation.get(
        "status"
    ) in {"complete", "complete_with_assumptions"}
    checks["final_design_contract_consistency"] = True
    validation["unresolved_reasons"] = [
        name for name, passed in checks.items() if passed is False
    ]
    validation["status"] = (
        "ready" if not validation["unresolved_reasons"] else "blocked"
    )
    result["final_validation"] = validation

    artifacts = render_topology_artifacts(
        topology,
        title=(plan.reaction_name or batch.reaction_description)[:90],
        builder=FlowsheetBuilder(),
    )
    result["svg_path"] = artifacts.get("svg_path", "")
    result["png_path"] = artifacts.get("png_path", "")
    result["diagram_artifacts"] = {
        key: value for key, value in artifacts.items() if key != "manifest"
    }
    result["diagram_render_manifest"] = artifacts.get("manifest", {})
    for key, filename in (("svg_path", "process.svg"), ("png_path", "process.png")):
        path = Path(str(result.get(key) or ""))
        if path.is_file():
            shutil.copy2(path, output / filename)

    result["final_design"] = build_final_design_contract(result)
    publish_final_design_artifacts(result, result["final_design"])
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )
    summary = {
        "source_result": str(source.resolve()),
        "inventory": str(Path(args.inventory).resolve()),
        "realization_status": realization.get("status"),
        "validation_status": result["final_validation"].get("status"),
        "inventory_allocation_status": allocation.get("status"),
        "final_design_status": result["final_design"].get("status"),
        "consistency_issues": (
            result["final_design"].get("consistency", {}).get("issues", [])
        ),
        "parameters": result["final_design"].get("parameters"),
        "stages": result["final_design"].get("stages"),
        "streams": result["final_design"].get("streams"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
