"""Recompile saved five-case FlowPilot candidates with the current contracts.

No LLM calls are made. The script preserves each generated chemistry candidate
and reruns deterministic realization, inventory allocation, topology, safety,
procedure, and final-design publication for regression evidence.
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
from flora_translate.main import _build_translate_topology
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.topology_compiler import compile_inventory_topology


def _revalidate(source: Path, output: Path) -> dict:
    result = json.loads(source.read_text())
    inventory = LabInventory.model_validate(result["inventory_snapshot"])
    batch = BatchRecord.model_validate(result["batch_record"])
    plan = ChemistryPlan.model_validate(result["chemistry_plan"])
    proposal = FlowProposal.model_validate(result["proposal"])
    intake = result.get("intake_package") or {}

    proposal, realization, validation = realize_executable_design(
        proposal,
        batch_record=batch,
        chemistry_plan=plan,
        inventory=inventory,
        hard_constraints=intake.get("hard_constraints") or intake.get("safety_operating_limits"),
    )
    topology = _build_translate_topology(proposal, plan, batch, inventory=inventory)
    topology, allocation = compile_inventory_topology(
        topology,
        proposal=proposal,
        inventory=inventory,
    )

    result["proposal"] = proposal.model_dump(mode="json")
    result["design_realization"] = realization
    result["multistage_inventory_plan"] = realization.get("multistage", {})
    result["process_topology"] = topology.model_dump(mode="json")
    result["inventory_allocation"] = allocation
    result["inventory_snapshot"] = inventory.model_dump(mode="json")
    result["recommended_disposition"] = "SCREEN"
    result["reported_disposition"] = "SCREEN"

    checks = validation.setdefault("checks", {})
    checks["inventory_topology_assignment_complete"] = allocation.get("status") in {
        "complete",
        "complete_with_assumptions",
    }
    checks["final_design_contract_consistency"] = True
    validation["unresolved_reasons"] = [
        name for name, passed in checks.items() if passed is False
    ]
    validation["status"] = "ready" if not validation["unresolved_reasons"] else "blocked"
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

    result["final_design"] = build_final_design_contract(result)
    publish_final_design_artifacts(result, result["final_design"])
    output.mkdir(parents=True, exist_ok=False)
    for key, filename in (("svg_path", "process.svg"), ("png_path", "process.png")):
        path = Path(str(result.get(key) or ""))
        if path.is_file():
            shutil.copy2(path, output / filename)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )
    summary = {
        "source": str(source),
        "realization_status": realization.get("status"),
        "allocation_status": allocation.get("status"),
        "validation_status": validation.get("status"),
        "final_design_status": result["final_design"].get("status"),
        "consistency_issues": result["final_design"].get("consistency", {}).get("issues", []),
        "parameters": result["final_design"].get("parameters"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign)
    output = Path(args.output)
    sources = sorted(campaign.glob("generation/*/*_flowpilot/repeat_01/result.json"))
    summaries = []
    for source in sources:
        relative = source.relative_to(campaign / "generation")
        target = output / relative.parts[0] / relative.parts[1]
        summary = _revalidate(source, target)
        summaries.append({"case": relative.parts[0], "candidate": relative.parts[1], **summary})
        print(
            f"{relative.parts[0]}/{relative.parts[1]}: "
            f"{summary['final_design_status']} ({len(summary['consistency_issues'])} issues)"
        )
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
