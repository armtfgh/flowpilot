"""Unscored offline deterministic replay; no generation, judging or archive edit."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from flora_translate.design_calculator import DesignCalculator
from flora_translate.design_disposition import apply_design_disposition_gate
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.final_design_contract import build_final_design_contract, publish_final_design_artifacts
from flora_translate.final_engineering import calculate_final_stages
from flora_translate.main import _build_translate_topology, _topology_matches_serialized_proposal, _sync_stage_hardware_from_compiled_topology
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.topology_semantics import normalize_topology_semantics

ARCHIVE_ROOT = ROOT / "ablation_results/practical_input"
CAMPAIGNS = ("claude_practical_pilot_20261014", "claude_practical_repair_20261014")


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def archive_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for campaign in CAMPAIGNS
            for p in sorted((ARCHIVE_ROOT / campaign).rglob("*")) if p.is_file()}


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "flora_translate").rglob("*.py"))}


def diff_values(old, new, path="$", missing=object()):
    changes = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                changes.append({"path": path + "." + key, "before_present": key in old,
                                "after_present": key in new, "before": old.get(key), "after": new.get(key)})
            else: changes.extend(diff_values(old[key], new[key], path + "." + key))
    elif old != new:
        changes.append({"path": path, "before": old, "after": new})
    return changes


def metric_closure(calc):
    raw_expected = calc["flow_rate_mL_min"] * calc["concentration_M"] * calc["target_conversion"] * 60 / calc["reactor_volume_mL"]
    return {"recorded_sty": calc["space_time_yield_mol_L_h"], "expected_sty": raw_expected,
            "passed": math.isclose(calc["space_time_yield_mol_L_h"], raw_expected, abs_tol=0.000051)}


def physical_closure(contract, original):
    """Independent arithmetic over the published canonical recipe and setpoints."""
    streams = contract["streams"]
    comps = contract["stream_components"]
    recipes = {row["step_id"][5:]: row["instruction"] for row in contract["operating_procedure"] if row["step_id"].startswith("PREP-")}
    doses = []
    for comp in comps:
        if comp.get("concentration_M") is None:
            continue
        label = comp["stream_label"]
        stream = next(row for row in streams if row["stream_label"] == label)
        recipe = recipes[label]
        match = re.search(re.escape(comp["name"]) + r": ([0-9.eE+-]+) mmol", recipe)
        expected = comp["concentration_M"] * 10
        assert match is not None, (comp["name"], recipe)
        charge = float(match.group(1))
        assert math.isclose(charge, expected, rel_tol=1e-6)
        doses.append({"name": comp["name"], "stream": label, "basis_mL": 10,
                      "printed_charge_mmol": charge, "expected_charge_mmol": expected,
                      "concentration_M": comp["concentration_M"],
                      "molar_delivery_mmol_min": comp["concentration_M"] * stream["flow_rate_mL_min"], "passed": True})
    ref = next(row for row in comps if "limiting" in row["role"])
    refstream = next(row for row in streams if row["stream_label"] == ref["stream_label"])
    ref_flux = ref["concentration_M"] * refstream["flow_rate_mL_min"]
    params = contract["parameters"]
    gases = []
    for gas in (row for row in streams if row["phase"] == "gas"):
        active = gas["gas_flow_sccm"] / 22.412722 * gas["gas_reagent_mole_fraction"]
        actual = gas["gas_flow_sccm"] * (params["temperature_C"] + 273.15) / 273.15 * 1.01325 / params["pressure_absolute_bar"]
        assert math.isclose(active, gas["gas_active_molar_flow_mmol_min"], abs_tol=1e-7)
        assert math.isclose(actual, gas["gas_flow_actual_mL_min"], abs_tol=1e-6)
        assert math.isclose(active / ref_flux, gas["gas_active_equivalents"], abs_tol=1e-6)
        gases.append({"gas": gas["gas_active_species"], "active_flux_mmol_min": active,
                      "substrate_flux_mmol_min": ref_flux, "active_equivalents": active / ref_flux,
                      "calculated_actual_flow_mL_min": actual, "passed": True})
    q_liquid = sum(row["flow_rate_mL_min"] for row in streams if row["phase"] == "liquid")
    inlet = params["reactor_volume_mL"] / (q_liquid + sum(row["gas_flow_sccm"] for row in streams if row["phase"] == "gas"))
    channel = params["reactor_volume_mL"] / (q_liquid + sum(row["gas_flow_actual_mL_min"] for row in streams if row["phase"] == "gas"))
    assert math.isclose(inlet, params["residence_time_inlet_min"], abs_tol=6e-6)
    assert math.isclose(channel, params["residence_time_in_channel_min"], abs_tol=6e-6)
    graph = contract["process_graph"]["topology"]
    edges = [(edge["from_op"], edge["to_op"], edge["stream_type"]) for edge in graph["streams"]]
    oldedges = [(edge["from_op"], edge["to_op"], edge["stream_type"]) for edge in original["final_design"]["process_graph"]["topology"]["streams"]]
    assert edges == oldedges
    return {"basis": "Nominal geometric volume; gas times are indices, not measured residence distributions.",
            "doses": doses, "gases": gases, "inlet_index_min": inlet, "in_channel_index_min": channel,
            "graph_edges": edges, "graph_connectivity_unchanged": True,
            "canonical_parameters_unchanged": params == original["final_design"]["parameters"], "passed": True}


def replay(path):
    original = read(path)
    public = read(path.with_name("input_public.json"))
    result = deepcopy(original)
    batch = BatchRecord.model_validate(original["batch_record"])
    inventory = LabInventory.model_validate(original["inventory_snapshot"])
    # The archived proposal already passed council review and physical realization.
    # Recompilation must not rewrite its feed strings or pretend to renew review.
    plan = ChemistryPlan.model_validate(original["chemistry_plan"])
    proposal = FlowProposal.model_validate(original["proposal"])
    validation = deepcopy(original["final_validation"])
    validation["checks"] = deepcopy(original["final_design"]["consistency"]["validation_checks"])
    result["final_validation"] = validation
    result["design_disposition"] = deepcopy(original["final_validation"]["design_disposition"])
    for target in ("recommended_disposition", "reported_disposition"):
        result[target] = result["design_disposition"]["recommended_disposition"]
    result["disposition_rationale"] = result["design_disposition"]["rationale"]
    result["safety_report"] = deepcopy(original["council_safety_report_diagnostic"])
    result.pop("final_design", None)
    analogies = original.get("_analogies") or []
    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal, inventory=inventory,
        analogies=analogies, target_flow_rate_mL_min=proposal.flow_rate_mL_min,
        target_tubing_ID_mm=proposal.tubing_ID_mm, target_residence_time_min=proposal.residence_time_min)
    payload = asdict(calc)
    payload.update({"flow_rate_mL_min": proposal.flow_rate_mL_min, "liquid_flow_rate_mL_min": proposal.flow_rate_mL_min,
        "reactor_volume_mL": proposal.reactor_volume_mL, "tubing_ID_mm": proposal.tubing_ID_mm,
        "residence_time_min": proposal.residence_time_min, "residence_time_inlet_min": proposal.residence_time_inlet_min,
        "residence_time_in_channel_min": proposal.residence_time_in_channel_min, "bpr_pressure_bar": proposal.BPR_bar,
        "stage_calculations": list(proposal.stage_parameters or []), "calculation_mode": "deterministic_post_council_realization"})
    payload.update(proposal.multiphase_metrics or {})
    payload["annotation_scope"] = "lumped diagnostic; final engineering is in final_stage_engineering"
    result["design_calculations"] = payload
    result["final_stage_engineering"] = calculate_final_stages(proposal, batch, plan, inventory, analogies)
    abstract = _build_translate_topology(proposal, plan, batch, inventory)
    result["process_requirements_topology"] = abstract.model_dump()
    compiled, allocation = compile_inventory_topology(abstract, proposal=proposal, inventory=inventory)
    compiled = normalize_topology_semantics(compiled)
    _sync_stage_hardware_from_compiled_topology(proposal, compiled)
    assert proposal.model_dump() == FlowProposal.model_validate(original["proposal"]).model_dump(), "Artifact rebuild changed a reviewed proposal"
    result["inventory_allocation"] = allocation
    result["instrument_manifest"] = allocation.get("instrument_manifest", [])
    validation["checks"]["topology_matches_serialized_design"] = _topology_matches_serialized_proposal(abstract, proposal)
    validation["checks"]["inventory_topology_assignment_complete"] = allocation.get("checks", {}).get("all_required_operations_assigned", False)
    validation["unresolved_reasons"] = [key for key, value in validation["checks"].items() if not value]
    validation["status"] = "ready" if not validation["unresolved_reasons"] else "screen_required"
    apply_design_disposition_gate(result, proposal=proposal, final_validation=validation, inventory=inventory,
        batch_record=batch, chemistry_plan=plan, objective=public["objective"], hard_constraints=public["hard_constraints"],
        council_safety_report=original["council_safety_report_diagnostic"])
    result["process_topology"] = compiled.model_dump()
    case = path.parts[-4]
    folder = OUT / case
    artifacts = render_topology_artifacts(compiled, title=f"UNSCORED offline repaired design — {case}", base_dir=folder / "diagrams")
    result["diagram_artifacts"] = artifacts
    result["diagram_render_manifest"] = read(Path(artifacts["render_manifest_path"]))
    result["svg_path"] = artifacts["svg_path"]
    result["png_path"] = artifacts["png_path"]
    synthetic = deepcopy(result)
    synthetic["final_validation"]["checks"]["pump_flow_feasible"] = False
    synthetic_contract = build_final_design_contract(synthetic)
    assert synthetic_contract["status"] == "blocked"
    assert any(issue["code"] == "FINAL-PUMP_FLOW_FEASIBLE" for issue in synthetic_contract["consistency"]["issues"])
    result["final_design"] = build_final_design_contract(result)
    publish_final_design_artifacts(result, result["final_design"])
    result["offline_repair_provenance"] = {"kind": "unscored offline deterministic artifact recompilation from unchanged reviewed proposal", "fresh_generation": False,
        "model_or_judge_calls": 0, "source_result": str(path.relative_to(ROOT)), "source_sha256": sha(path),
        "benchmark_scores_updated": False, "source_model_proposal_preserved_in_archive": True, "reviewed_proposal_unchanged": result["proposal"] == original["proposal"],
        "restored_fields": {"final_validation.checks": "final_design.consistency.validation_checks", "design_disposition": "final_validation.design_disposition", "safety_report": "council_safety_report_diagnostic"},
        "no_new_council_approval": True}
    contract = result["final_design"]
    checks = {"case": case, "source_sha256": sha(path), "status": contract["status"],
        "disposition": result["recommended_disposition"], "contract_issues": contract["consistency"]["issues"],
        "validation_checks": result["final_validation"]["checks"],
        "canonical_sha256": contract.get("canonical_sha256"),
        "diagram_topology_sha256": artifacts["topology_sha256"], "render_status": artifacts["render_status"],
        "recipe": [row for row in contract["operating_procedure"] if row["step_id"].startswith("PREP")],
        "canonical_components": contract["stream_components"], "canonical_parameters": contract["parameters"],
        "physical_closure": physical_closure(contract, original),
        "synthetic_failed_pump_check_still_blocks": True,
        "fresh_render_topology_matches_manifest": artifacts["topology_sha256"] == result["diagram_render_manifest"]["topology_sha256"],
        "top_level_metric_closure": metric_closure(result["design_calculations"]),
        "stage_metric_closure": [metric_closure(stage["calculations"]) for stage in result["final_stage_engineering"]["stages"] if stage.get("status") == "calculated"],
        "mixer_rationales": [op.rationale for op in compiled.unit_operations if op.op_type == "mixer"],
        "original_proposal_unchanged": result["proposal"] == original["proposal"],
        "original_chemistry_plan_unchanged": result["chemistry_plan"] == original["chemistry_plan"],
        "original_batch_protocol_unchanged": result["batch_record"] == original["batch_record"],
        "original_inventory_unchanged": result["inventory_snapshot"] == original["inventory_snapshot"]}
    assert result["proposal"] == original["proposal"]
    assert result["chemistry_plan"] == original["chemistry_plan"]
    assert contract["status"] == "executable", contract["consistency"]["issues"]
    assert all(checks["validation_checks"].values())
    assert checks["top_level_metric_closure"]["passed"]
    assert all(row["passed"] for row in checks["stage_metric_closure"])
    write(folder / "replayed_result.json", result)
    write(folder / "changes_from_frozen_result.json", diff_values(original, result))
    write(folder / "verification.json", checks)
    return checks


def forbidden(*args, **kwargs): raise AssertionError("Network access is forbidden during offline replay")


def main():
    before = archive_hashes()
    sources = source_hashes()
    paths = sorted((ARCHIVE_ROOT / CAMPAIGNS[1] / "generation").glob("*/claude_repaired_flowpilot/repeat_01/result.json"))
    assert len(paths) == 3
    with patch("socket.socket.connect", forbidden), patch("socket.create_connection", forbidden):
        rows = [replay(path) for path in paths]
    assert before == archive_hashes(), "A frozen campaign archive changed"
    assert sources == source_hashes(), "Live sources changed during replay; rerun once stable"
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "unscored offline repairs after completed benchmark",
        "fresh_generation": False, "benchmark_judgments_changed": False, "model_or_judge_calls": 0,
        "archive_files_unchanged": len(before), "archive_sha256_before_and_after": before,
        "production_source_sha256": sources, "replay_script_sha256": sha(Path(__file__)), "cases": rows}
    write(OUT / "replay_summary.json", summary)
    lines = ["POST-BENCHMARK REPAIRS — UNSCORED OFFLINE ARTIFACT RECOMPILATION", "",
             "The paid benchmark was completed and verified before these additional source changes.",
             "Validation: the root agent reported 267 combined production regression tests passing on the final source. The 131 alias-focused and 61 STY/mixer-focused checks are overlapping subsets and are not added to that count.",
             "No model or judge was called. No benchmark result, packet, judgment, score or source archive was edited.",
             "The replay starts with each unchanged saved council-reviewed proposal, chemistry plan and physical realization.",
             "It restores the exact saved pre-publication validation-check snapshot and council safety diagnostic, recomputes engineering metrics, regenerates canonical components and topology, renders a fresh diagram, and executes the full final contract compiler and publisher.",
             "Source-proven alias normalization occurs in canonical component compilation; the raw model proposal is preserved for provenance.",
             "Re-running upstream reconciliation on already-reviewed archived proposals was rejected as a replay method because changed solvent/name labels triggered COUNCIL-DELIVERY-STATE-DRIFT. No review-integrity check was bypassed or set to true to suppress that issue. These artifacts are recompilations, not renewed council approvals.", "",
             "Fixes include a single source-proven H2 substrate charge instead of duplicate alias charges; graph-local mixer text without the stale separator/BPR ordering; and conditional space-time yield calculated as liquid molar productivity divided by total nominal reactor volume.",
             "Target conversion is still an assumption, not measured or predicted flow yield. Nominal gas times retain their declared reference bases.", ""]
    for row in rows:
        lines.append(f"{row['case']}: {row['status']}, disposition {row['disposition']}, {len(row['contract_issues'])} contract issues, recipe/gas/time/metric closures passed.")
        lines.append(f"  STY {row['top_level_metric_closure']['recorded_sty']} mol/L/h; unchanged canonical physical parameters and connectivity.")
    lines += ["", f"All {len(before)} files across both campaign archives had identical hashes before and after recompilation.",
              "Production source hashes are recorded in replay_summary.json and were stable throughout this replay.",
              "Separate synthetic failed-pump controls remained blocked for all three cases.",
              "Each case's replayed_result.json and verification.json are authoritative for this offline run; timestamped diagram directories are intermediate or current renders, and the current paths are named in replayed_result.json.",
              "Known scope limit: the H2 model's retained chief rationale contains the incorrect claim that 1.41 sccm is closer to a 1.0 sccm lower bound than 1.13 sccm. This arbitrary historical model prose was not changed and does not alter the now-coherent canonical setpoints and recipe. No fresh judge score was obtained for the additional repairs."]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n")
    manifest = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    write(OUT / "artifact_manifest.json", manifest)
    print(json.dumps([{k: r[k] for k in ("case", "status", "contract_issues", "top_level_metric_closure", "stage_metric_closure")} for r in rows], indent=2))


if __name__ == "__main__": main()
