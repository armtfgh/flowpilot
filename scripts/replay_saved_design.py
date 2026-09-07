"""Re-run deterministic realization and final gates from an immutable saved run.

No LLM calls, new candidates, or inventory confirmations are made. The original
result is retained; newly compiled artifacts are autosaved as a separate run.
"""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_translate.component_identity import unique_components
from flora_translate.chemistry_contract import reconcile_reagent_facts
from flora_translate.design_calculator import DesignCalculator
from flora_translate.final_engineering import calculate_final_stages
from flora_translate.design_disposition import apply_design_disposition_gate
from flora_translate.design_realizer import realize_executable_design
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.final_design_contract import build_final_design_contract, publish_final_design_artifacts
from flora_translate.gui_autosave import autosave_gui_result
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.main import (
    _apply_final_design_guards, _build_translate_topology,
    _store_process_topology, _sync_stage_hardware_from_compiled_topology,
    _topology_matches_serialized_proposal,
)
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.topology_semantics import normalize_topology_semantics


def replay(source: Path, output: Path) -> Path:
    original_bytes = source.read_bytes()
    result = json.loads(original_bytes)
    if not result.get("proposal") or not result.get("chemistry_plan"):
        raise ValueError("This run stopped before a candidate existed; it cannot be replayed without a new design run.")
    original = copy.deepcopy(result)
    batch = BatchRecord.model_validate(result["batch_record"])
    plan = ChemistryPlan.model_validate(result["chemistry_plan"])
    inventory = LabInventory.model_validate(result["inventory_snapshot"])
    intake = result.get("intake_package") or {}
    constraints = {"inventory": intake.get("inventory_constraints"), "operating_limits": intake.get("operating_limits")}
    normalization = []
    reconcile_reagent_facts(batch, plan)
    for feed in [*plan.stream_logic, *(feed for stage in plan.stages for feed in stage.feed_streams)]:
        normalized = unique_components(feed.reagents)
        if normalized != feed.reagents:
            normalization.append({"stream_label": feed.stream_label, "before": list(feed.reagents), "after": normalized})
            feed.reagents = normalized
    if plan.canonical_contract:
        plan.canonical_contract = plan.canonical_contract.model_copy(update={
            "operations": [operation.model_copy(update={"species": unique_components(operation.species)})
                           for operation in plan.canonical_contract.operations],
        })
    proposal, realization, validation = realize_executable_design(
        FlowProposal.model_validate(result["proposal"]), batch_record=batch,
        chemistry_plan=plan, inventory=inventory, hard_constraints=constraints,
        operating_limits=intake.get("operating_limits"),
    )
    result["chemistry_plan"] = plan.model_dump(mode="json")
    result["proposal"] = proposal.model_dump(mode="json")
    result["design_realization"] = realization
    result["final_validation"] = validation
    result["multistage_inventory_plan"] = realization["multistage"]
    calculations = asdict(DesignCalculator().run(
        batch, chemistry_plan=plan, proposal=proposal, inventory=inventory,
        target_flow_rate_mL_min=proposal.flow_rate_mL_min,
        target_tubing_ID_mm=proposal.tubing_ID_mm,
        target_residence_time_min=proposal.residence_time_min,
    ))
    calculations.update({
        "flow_rate_mL_min": proposal.flow_rate_mL_min,
        "liquid_flow_rate_mL_min": proposal.flow_rate_mL_min,
        "reactor_volume_mL": proposal.reactor_volume_mL,
        "tubing_ID_mm": proposal.tubing_ID_mm,
        "residence_time_min": proposal.residence_time_min,
        "residence_time_inlet_min": proposal.residence_time_inlet_min,
        "residence_time_in_channel_min": proposal.residence_time_in_channel_min,
        "bpr_pressure_bar": proposal.BPR_bar,
        "stage_calculations": proposal.stage_parameters,
        "calculation_mode": "deterministic_post_council_realization",
    })
    calculations.update(proposal.multiphase_metrics or {})
    result["design_calculations"] = calculations
    result["design_calculations"]["annotation_scope"] = "lumped diagnostic; final engineering is in final_stage_engineering"
    result["final_stage_engineering"] = calculate_final_stages(proposal, batch, plan, inventory)
    _apply_final_design_guards(result)
    proposal = FlowProposal.model_validate(result["proposal"])
    requirements = _build_translate_topology(proposal, plan, batch, inventory)
    result["process_requirements_topology"] = requirements.model_dump(mode="json")
    topology, allocation = compile_inventory_topology(requirements, proposal=proposal, inventory=inventory)
    topology = normalize_topology_semantics(topology)
    result["inventory_allocation"] = allocation
    result["instrument_manifest"] = allocation["instrument_manifest"]
    _sync_stage_hardware_from_compiled_topology(proposal, topology)
    result["proposal"] = proposal.model_dump(mode="json")
    result["multistage_inventory_plan"]["stage_parameters"] = proposal.stage_parameters
    result["design_calculations"]["stage_calculations"] = proposal.stage_parameters
    validation["checks"]["topology_matches_serialized_design"] = _topology_matches_serialized_proposal(topology, proposal)
    validation["checks"]["inventory_topology_assignment_complete"] = allocation["checks"]["all_required_operations_assigned"]
    _store_process_topology(result, topology)
    result.pop("final_design", None)
    result.pop("diagram_render_manifest", None)
    result.pop("diagram_artifacts", None)
    apply_design_disposition_gate(
        result, proposal=proposal, final_validation=validation, inventory=inventory,
        batch_record=batch, chemistry_plan=plan, objective=intake.get("objective") or "",
        hard_constraints=constraints,
        council_safety_report=result.get("council_safety_report_diagnostic"),
    )
    contract = build_final_design_contract(result)
    artifacts = render_topology_artifacts(topology, base_dir=output / "diagrams")
    if contract["status"] == "executable":
        result["svg_path"], result["png_path"] = artifacts["svg_path"], artifacts["png_path"]
        result["diagram_artifacts"] = {k: v for k, v in artifacts.items() if k != "manifest"}
        result["diagram_render_manifest"] = artifacts["manifest"]
        contract = build_final_design_contract(result)
    if contract["status"] != "executable":
        result["svg_path"] = result["png_path"] = ""
        result["diagnostic_svg_path"], result["diagnostic_png_path"] = artifacts["svg_path"], artifacts["png_path"]
        result["diagnostic_topology"] = topology.model_dump(mode="json")
        result["recommended_disposition"] = result["reported_disposition"] = "BLOCK"
        result["disposition_rationale"] = "Deterministic replay still requires review; see final consistency issues."
    publish_final_design_artifacts(result, contract)
    result["replay_provenance"] = {
        "source_result": str(source.resolve()), "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "replayed_at": datetime.now(timezone.utc).isoformat(), "new_llm_calls": 0,
        "generation_usage_is_inherited_from_source": True,
        "source_status": original["final_design"]["status"],
        "source_issues": original["final_design"]["consistency"]["issues"],
        "normalization": normalization,
        "inventory_changed": inventory.model_dump(mode="json") != LabInventory.model_validate(original["inventory_snapshot"]).model_dump(mode="json"),
    }
    result["explanation_status"] = "pre_realization_audit_only"
    run_dir = autosave_gui_result(result, source="deterministic_replay", intake_package=intake)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_dir.name, "autosave_dir": str(run_dir),
        "status": contract["status"], "parameters": contract["parameters"],
        "issues": contract["consistency"]["issues"],
        "replay": result["replay_provenance"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert source.read_bytes() == original_bytes, "Original result must not be changed"
    print(json.dumps({k: summary[k] for k in ("run_id", "status", "issues")}, indent=2))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.source, args.output)
