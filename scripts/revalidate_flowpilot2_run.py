"""Replay saved model outputs through current code, with exact candidate matching.

No fresh generation is claimed. If any candidate's physical design changes,
recorded council judgments cannot be reused and this verification fails.
"""
import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--upstream-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.output / "run.log"), logging.StreamHandler()])
    original = json.loads(args.result.read_text())
    upstream = json.loads(args.upstream_snapshot.read_text())
    from flora_translate.main import translate
    from flora_translate.schemas import ChemistryPlan, FlowProposal
    from flora_translate.gui_autosave import autosave_gui_result
    audit = original["scientific_assessment"]
    calls = {c["role"]: c for c in audit["calls"]}
    expected = {c["candidate_id"]: c for c in calls["DrChemistry"]["request"]["context"]["candidates"]}
    compared = []
    def recorded(system, user_content, max_tokens):
        request = json.loads(user_content)
        role = request["role"]
        previous = calls[role]
        if system != previous["system"] or any(request["context"].get(k) != previous["request"]["context"].get(k)
                for k in ("objective", "objective_policy", "authority_labeled_intake")):
            raise AssertionError("Council policy or intake changed; fresh model review is required")
        for c in request["context"]["candidates"]:
            before = expected[c["candidate_id"]]
            for key in ("stage_parameters", "streams", "BPR_bar", "target_stage_screen_min"):
                if c[key] != before[key]:
                    raise AssertionError(f"Candidate {c['candidate_id']} {key} changed; fresh model review is required")
        compared.append(role)
        return calls[role]["raw_response"]
    runtime = dict(original["pipeline_runtime"])
    routing = runtime["model_routing"]
    options = {"design_policy": "scientific_v2", "candidate_budget": 12, **routing}
    (args.output / "request.json").write_text(json.dumps({"fresh_generation": False, "source_result": str(args.result),
        "upstream_snapshot": str(args.upstream_snapshot), "runtime_options": options}, indent=2))
    with ExitStack() as stack:
        stack.enter_context(patch("flora_translate.main.analyze_batch_chemistry", return_value=ChemistryPlan.model_validate(upstream["chemistry_plan"])))
        stack.enter_context(patch("flora_translate.main.VectorRetriever.retrieve", return_value=original["_analogies"]))
        stack.enter_context(patch("flora_translate.main.AnalogySelector.select", return_value=original["_analogies"]))
        stack.enter_context(patch("flora_translate.main.TranslationLLM.generate", return_value=FlowProposal.model_validate(original["engineering_history"]["before_council"]["proposal"])))
        stack.enter_context(patch("flora_translate.engine.llm_agents.call_llm", side_effect=recorded))
        result = translate(original["intake_package"]["raw_protocol"], intake_package=original["intake_package"], runtime_options=options)
    result["verification_provenance"] = {"fresh_generation": False, "source_result": str(args.result),
        "type": "full-pipeline deterministic replay of saved model outputs", "physically_identical_candidates_verified_for": compared,
        "annotations": "Recomputed without assumed kinetics, reaction heat or conversion. Model outputs are unchanged."}
    result["scientific_assessment"]["verification_provenance"] = result["verification_provenance"]
    for call in result["scientific_assessment"]["calls"]:
        call["replayed_from_saved_response"] = True
    audit_path = Path(result["scientific_assessment"]["archive_path"]) / "audit.json"
    audit_path.write_text(json.dumps(result["scientific_assessment"], indent=2, default=str))
    archive = autosave_gui_result(result, intake_package=result["intake_package"], source="scientific_verified")
    result["autosave_dir"] = str(archive)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    summary = {"status": result["final_design"]["status"], "gui_archive": str(archive),
               "fresh_generation": False, "candidate_identity_checks": len(compared) * 12,
               "selected_candidate_id": result["scientific_assessment"]["selected_candidate_id"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "executable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
