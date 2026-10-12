#!/usr/bin/env python3
"""Software regression on frozen inputs, with explicit LLM test doubles.

No real model generation, judging, benchmark scores, or experimental approval.
Real inventory realization, council control flow, deterministic audit, topology
and final contract run unchanged. Network is forbidden for the entire replay.
"""
import argparse
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts.replay_astra_finalization import replay_contract
from flora_translate.design_realizer import realize_executable_design
from flora_translate.engine.council_v4 import chief, designer
from flora_translate.final_design_contract import publish_final_design_artifacts
from flora_translate.main import _build_translate_topology, _topology_matches_serialized_proposal
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.topology_semantics import normalize_topology_semantics


def forbid_network(*args, **kwargs):
    raise RuntimeError("Network forbidden in offline software regression")


def verify_case(source, scenario):
    raw = source.read_bytes()
    data = json.loads(raw)
    batch = BatchRecord.model_validate(data["batch_record"])
    plan = ChemistryPlan.model_validate(data["chemistry_plan"])
    inventory = LabInventory.model_validate(data["inventory_snapshot"])
    proposal = FlowProposal.model_validate(data["pre_council_proposal"])
    constraints = data.get("design_disposition", {}).get("requirements", {}).get("source_text", "")
    snapshots, reviews, selections = {}, [], []
    chief_signature = inspect.signature(chief._run_chief_llm)

    class Recorder:
        def snapshot(self, name, payload):
            snapshots[name] = deepcopy(payload)

        def stage_start(self, *args, **kwargs):
            pass

        def stage_end(self, *args, **kwargs):
            pass

    def framing(system, user, **kwargs):
        assert "Authoritative batch protocol:" in user
        assert batch.solvent in user and "actual_batch_time_min=" in user
        return json.dumps({"reaction_class": plan.reaction_class, "ambiguities": []})

    def score(**kwargs):
        rows = deepcopy(kwargs["candidates"])
        reviews.append(rows)
        result = {"blocked_by_scoring": [], "tool_calls": {}}
        for domain, field in [("chemistry", "combined_score"), ("kinetics", "kinetics_score"),
                              ("fluidics", "fluidics_score"), ("safety", "safety_score")]:
            result[domain + "_scores"] = [{"candidate_id": r["id"], field: .7,
                "reasoning": "SOFTWARE TEST DOUBLE, not scientific review or benchmark evidence",
                "verdict": "WARNING", "proposed_changes": {}, "concerns": [],
                "BPR_current_bar": r["BPR_bar"]} for r in rows]
        return result

    def select(**kwargs):
        chief_signature.bind(**kwargs)  # The double must obey the production API.
        selections.append(deepcopy(kwargs["candidates"]))
        if scenario == "missing_revision_approval" and len(selections) > 1:
            return {}, []
        reject = scenario == "reject_revision" and len(selections) > 1
        selected = None if reject else kwargs["candidates"][0]["id"]
        return {"selected_candidate_id": selected, "selection_rationale":
                "SOFTWARE TEST DOUBLE: reject" if reject else "SOFTWARE TEST DOUBLE: accept"}, []

    def revise(**kwargs):
        if scenario == "unchanged":
            return None
        winner = deepcopy(kwargs["winner"])
        # Fixed volume and concentration; a new flow must receive renewed review.
        winner["Q_mL_min"] *= 1.25
        return winner

    with ExitStack() as stack:
        for target, replacement in [("socket.socket.connect", forbid_network),
                ("flora_translate.engine.council_v4.designer.call_llm", framing),
                ("flora_translate.engine.council_v4.chief.run_domain_scoring", score),
                ("flora_translate.engine.council_v4.chief._run_chief_llm", select),
                ("flora_translate.engine.council_v4.chief.run_revision_stage", revise)]:
            stack.enter_context(patch(target, replacement))
        candidate, _ = chief.CouncilV4().run(proposal, batch, [], inventory,
            chemistry_plan=plan, objectives="de-risk the first flow screen", candidate_budget=2,
            max_loop_rounds=1, hard_constraints=constraints, intake_package=data.get("intake_package"),
            benchmark_recorder=Recorder(), execution_config={"enable_dfmea": False,
                "enable_preselection_refinement": False})
        realized, audit, validation = realize_executable_design(candidate.proposal,
            batch_record=batch, chemistry_plan=plan, inventory=inventory, hard_constraints=constraints)
        topology, allocation = compile_inventory_topology(
            _build_translate_topology(realized, plan, batch, inventory), proposal=realized, inventory=inventory)
        topology = normalize_topology_semantics(topology)
        validation["checks"]["inventory_topology_assignment_complete"] = bool(allocation["checks"]["all_required_operations_assigned"])
        validation["checks"]["topology_matches_serialized_design"] = _topology_matches_serialized_proposal(topology, realized)
        fresh = replay_contract({"intake_package": data.get("intake_package"),
            "safety_report": candidate.safety_report}, realized, batch, plan, inventory,
            constraints, validation, allocation, topology)
        publish_final_design_artifacts(fresh, fresh["final_design"])

    expected_reviews = 1 if scenario == "unchanged" else 2
    assert len(reviews) == expected_reviews, (scenario, len(reviews), candidate.deliberation_log.summary)
    assert len(selections) == expected_reviews, (scenario, len(selections), snapshots)
    if scenario in ("reject_revision", "missing_revision_approval"):
        assert fresh["final_design"]["status"] == "blocked"
        assert not fresh.get("operating_procedure")
        assert not fresh["final_design"].get("process_graph")
        assert realized.inventory_constraints.get("council_selection_block")
    else:
        assert fresh["final_design"]["status"] == "executable", fresh["final_design"]["consistency"]
        assert all(validation["checks"].values()), validation
        assert fresh["final_design"]["process_graph"]["topology"]["unit_operations"]
        assert not realized.engine_validated  # The test never validates kinetic predictions.
    if scenario != "unchanged":
        assert reviews[-1][0]["Q_mL_min"] != selections[0][0]["Q_mL_min"]
    return {"case": source.parts[-4], "scenario": scenario, "passed": True,
            "scope": __doc__, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "domain_review_rounds": len(reviews), "chief_selection_rounds": len(selections),
            "snapshots": snapshots, "realization": audit, "rebuilt_result": fresh}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    paths = sorted((args.source / "generation").glob("*/astra_flowpilot/repeat_01/result.json"))
    assert len(paths) == 3
    for path in paths:
        for scenario in ("unchanged", "accept_revision", "reject_revision", "missing_revision_approval"):
            result = verify_case(path, scenario)
            (args.output / (result["case"] + "_" + scenario + ".json")).write_text(json.dumps(result, indent=2, default=str))
            summary.append({k: result[k] for k in ("case", "scenario", "passed", "domain_review_rounds", "chief_selection_rounds", "source_sha256")})
    (args.output / "summary.json").write_text(json.dumps({"scope": __doc__, "runs": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
