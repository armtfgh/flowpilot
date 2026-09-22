"""Append-only feature-off/on comparison with frozen upstream generation.

The council is live in BOTH arms. No collaborator failure or desired oxygen
answer is added to the historical chemist input. This tests a council change,
not generation-model variance or experimentally demonstrated safety.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(result):
    final = result.get("final_design", {})
    review = result.get("scientific_assessment", {}).get("backflow_review", {})
    gas = next((s for s in result.get("proposal", {}).get("streams", []) if s["phase"] == "gas"), {})
    return {"final_contract_status": final.get("status"), "gas": gas.get("contents"),
        "gas_STP_mL_min": gas.get("gas_flow_sccm"), "reagent_fraction": gas.get("gas_reagent_mole_fraction"),
        "oxygen_equivalents": gas.get("molar_equiv"), "introduction_stage": gas.get("introduction_stage"),
        "stages": result.get("result_report", {}).get("stages"),
        "selected_candidate_id": result.get("scientific_assessment", {}).get("selected_candidate_id"),
        "council_action": review.get("decision", {}).get("action"),
        "revision_status": review.get("application_status"), "revision_error": review.get("revision_error"),
        "reported_backflow_findings": result.get("backflow_assessment", {}).get("findings", []),
        "laboratory_execution_status": result.get("backflow_assessment", {}).get("laboratory_execution_status", "not_assessed_by_feature"),
        "selection_preserved": result.get("scientific_assessment", {}).get("selected_design_preserved"),
        "stage_closure": bool(result.get("result_report", {}).get("stages")) and all(s["closure"] for s in result["result_report"]["stages"]),
        "figure_generated": bool(result.get("png_path") and Path(result["png_path"]).is_file())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/khu_revised_six_20260915/presentation/figure5_set1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--endpoint")
    parser.add_argument("--arm", default="both", choices=["both", "off", "on"])
    parser.add_argument("--offline-selection-test", action="store_true")
    parser.add_argument("--physics-profile", type=Path, help="Opt-in transient assumptions JSON; uses existing fluidics/safety reviewers in the on arm")
    args = parser.parse_args()
    physics_profile = json.loads(args.physics_profile.read_text()) if args.physics_profile else None
    if physics_profile is not None:
        from flora_translate.engine.council_v4.transient_flow import TransientProfile
        physics_profile = TransientProfile.model_validate(physics_profile).model_dump()
    out = args.output or ROOT / "outputs" / ("council_backflow_ab_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / "result.json").read_text())
    package = json.loads((args.source / "intake_package.json").read_text())
    proposal = source["engineering_history"]["before_council"]["proposal"]
    plan = source["chemistry_plan"]
    from flora_translate.schemas import ChemistryPlan, FlowProposal
    from flora_translate.engine import llm_agents
    from flora_translate.gui_autosave import autosave_gui_result
    import flora_translate.main as pipeline

    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        dump(manifest_path, {"source": str(args.source), "source_result_sha256": digest(args.source / "result.json"),
            "source_intake_sha256": digest(args.source / "intake_package.json"),
            "baseline_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "upstream": "Frozen archived chemistry plan and translation proposal; no new upstream calls",
            "downstream_model": args.model, "downstream_provider": args.provider,
            "comparison": "identical protocol, answers, inventory, upstream plan/proposal, retrieval and council model; feature flag differs",
            "new_failure_feedback_in_input": False, "offline_selection_test": args.offline_selection_test,
            "physics_profile": physics_profile,
            "physics_comparison_scope": "If enabled: combined topology plus physics tools versus neither; not an isolated estimate of marginal physics benefit",
            "wet_lab_validation": False, "repeats_per_arm": 1})
        dump(out / "intake_package.json", package)
        dump(out / "frozen_generation.json", {"chemistry_plan": plan, "proposal": proposal, "analogies": source.get("_analogies", [])})
        (out / "protocol.txt").write_text(package["raw_protocol"])
        snapshot = out / "source_snapshot"
        for path in [*ROOT.glob("flora_translate/*.py"), *ROOT.glob("flora_translate/engine/council_v4/*.py"), Path(__file__)]:
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        baseline = out / "baseline_source"
        for relative in ("flora_translate/engine/council_v4/scientific.py", "flora_translate/engine/council_v4/chief.py",
                         "flora_translate/main.py", "flora_translate/design_realizer.py", "flora_translate/pipeline_runtime.py"):
            target = baseline / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT))
    else:
        saved = json.loads(manifest_path.read_text())
        assert saved["source_result_sha256"] == digest(args.source / "result.json")
        assert saved["source_intake_sha256"] == digest(args.source / "intake_package.json")
        assert saved["downstream_provider"] == args.provider
        assert saved["downstream_model"] == args.model and saved["offline_selection_test"] == args.offline_selection_test
        assert saved.get("physics_profile") == physics_profile
    llm_agents.set_llm_runtime_overrides(capture_content=True, temperature=0)
    print("OUTPUT", out, flush=True)
    summaries = {}
    for enabled in (False, True):
        arm = "on" if enabled else "off"
        if args.arm not in ("both", arm):
            continue
        folder = out / arm
        folder.mkdir(exist_ok=False)
        started = time.monotonic()
        status = {"arm": arm, "status": "running"}
        dump(folder / "status.json", status)
        print("START", arm, flush=True)
        runtime = {"design_policy": "scientific_v2", "candidate_budget": 12,
            "council_backflow_review": enabled, "downstream_model": args.model, "downstream_provider": args.provider}
        if args.endpoint:
            runtime["model_endpoints"] = {args.model: args.endpoint}
        if enabled and physics_profile is not None:
            runtime["council_physics_profile"] = physics_profile
        dump(folder / "runtime.json", runtime)
        def observer(event):
            with (folder / "llm_calls.jsonl").open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        llm_agents.set_llm_observer(observer)
        handler = logging.FileHandler(folder / "run.log")
        console_handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
        for h in console_handlers:
            logging.getLogger().removeHandler(h)
        logging.getLogger().addHandler(handler)
        try:
            with (folder / "console.log").open("w") as console, redirect_stdout(console), redirect_stderr(console), \
                    patch.object(pipeline, "analyze_batch_chemistry", side_effect=lambda *a, **k: ChemistryPlan.model_validate(deepcopy(plan))), \
                    patch.object(pipeline.VectorRetriever, "retrieve", return_value=source.get("_analogies", [])), \
                    patch.object(pipeline.AnalogySelector, "select", return_value=source.get("_analogies", [])), \
                    patch.object(pipeline.TranslationLLM, "generate", side_effect=lambda *a, **k: FlowProposal.model_validate(deepcopy(proposal))):
                if args.offline_selection_test:
                    from contextlib import nullcontext
                    def fake(system, user, tokens):
                        q = json.loads(user)
                        dump(folder / (q["role"] + "_request.json"), {"system": system, "request": q})
                        if q.get("phase") == "tool_request":
                            return json.dumps({"tool_name": "simulate_flow_transients", "candidate_ids": list(range(1, 13))})
                        if q["role"] == "FlowOperability":
                            answer = {"action": "propose_pure_oxygen", "justification": "Synthetic pipeline wiring test; not an actual LLM conclusion.",
                                "addressed_finding_ids": [q["assessment"]["findings"][0]["finding_id"]], "required_controls": ["Branch protection"],
                                "required_confirmations": ["Pressure and oxygen service"], "limitations": ["Synthetic selection"]}
                        elif q["role"].startswith("Dr"):
                            answer = {"reviews": [{"candidate_id": n, "recommendation": "acceptable", "hard_violation": False,
                                "justification": "Synthetic review", "uncertainties": ["Not laboratory validation"]} for n in range(1, 13)]}
                            if "physics_assessment" in q["response_schema"]:
                                r = q["context"]["physics_tool_results"][q["role"]]
                                ids = [row["evidence_id"] for c in r["candidate_results"] for row in c.get("representative_simulations", [])]
                                answer["physics_assessment"] = {"tool_result_id": r["result_id"], "evidence_ids": ids[:1],
                                    "proposed_alternatives": [], "limitations": ["Synthetic wiring test; unmeasured dynamics"]}
                        elif q["role"] == "Skeptic":
                            answer = {"vetoes": [], "assessment": "Synthetic review", "required_measurements": ["pressure"]}
                        else:
                            answer = {"candidate_id": 1, "justification": "Synthetic selection", "objective_alignment": "Wiring test",
                                "alternatives": [{"candidate_id": n, "reason_not_selected": "Synthetic comparison"} for n in (2, 3)],
                                "answer_impacts": [], "limitations": ["Synthetic selection"], "next_measurements": ["pressure"]}
                        return json.dumps(answer)
                    context = patch.object(llm_agents, "call_llm", side_effect=fake)
                else:
                    from contextlib import nullcontext
                    context = nullcontext()
                with context:
                    result = pipeline.translate(package["raw_protocol"], intake_package=deepcopy(package), runtime_options=runtime)
                gui = autosave_gui_result(result, intake_package=package, source="backflow_" + arm,
                                          user_input=package["raw_protocol"])
                dump(folder / "result.json", result)
                dump(folder / "summary.json", summarize(result))
                dump(folder / "process_topology.json", result.get("process_topology"))
                dump(folder / "backflow_assessment.json", result.get("backflow_assessment"))
                dump(folder / "council_audit.json", result.get("scientific_assessment"))
                physics = result.get("scientific_assessment", {}).get("physics_review")
                if physics:
                    shutil.copytree(physics["archive_path"], folder / "physics_tools")
                    dump(folder / "physics_review.json", physics)
                for key, name in (("png_path", "topology.png"), ("svg_path", "topology.svg")):
                    if result.get(key) and Path(result[key]).is_file():
                        shutil.copy2(result[key], folder / name)
                report = result.get("result_report", {})
                csv_rows(folder / "stages.csv", report.get("stages", []))
                csv_rows(folder / "streams.csv", report.get("streams", []))
                status.update(status="completed", gui_run_id=gui.name, final_contract_status=result["final_design"]["status"])
                summaries[arm] = summarize(result)
        except Exception as exc:
            (folder / "failure.txt").write_text(traceback.format_exc())
            status.update(status="failed", error=str(exc))
        finally:
            status["elapsed_seconds"] = time.monotonic() - started
            dump(folder / "status.json", status)
            logging.getLogger().removeHandler(handler)
            for h in console_handlers:
                logging.getLogger().addHandler(h)
            handler.close()
            llm_agents.set_llm_observer(None)
        print("FINISH", json.dumps(status), flush=True)
    for arm in ("off", "on"):
        path = out / arm / "summary.json"
        if path.exists():
            summaries[arm] = json.loads(path.read_text())
    dump(out / "comparison.json", summaries)
    if len(summaries) == 2:
        rows = []
        for arm in ("off", "on"):
            result = json.loads((out / arm / "result.json").read_text())
            for c in result["scientific_assessment"]["candidates"]:
                p = c["proposal"]
                gas = next(s for s in p["streams"] if s["phase"] == "gas")
                rows.append({"arm": arm, "candidate_id": c["candidate_id"], "gas": " / ".join(gas["contents"]),
                    "gas_inlet_STP_mL_min": gas["gas_flow_sccm"], "oxygen_fraction": gas["gas_reagent_mole_fraction"],
                    "oxygen_equiv": gas["molar_equiv"], "BPR_bar": p["BPR_bar"],
                    "stage1_nominal_min": p["stage_parameters"][0]["residence_time_min"],
                    "stage2_nominal_inlet_min": p["stage_parameters"][1]["residence_time_min"]})
        csv_rows(out / "paired_candidates.csv", rows)
    print("COMPLETE", out, flush=True)
    statuses = [json.loads(p.read_text()) for p in out.glob("*/status.json")]
    if any(s["status"] != "completed" for s in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
