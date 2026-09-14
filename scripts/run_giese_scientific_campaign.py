"""Instrumented full pipeline run of the supplied Giese/oxidation case."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import traceback
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oxygen-stage", type=int, choices=(1, 2), default=2,
                        help="Requested oxygen introduction reactor. Stage 1 is the user's revised variant.")
    parser.add_argument("--replay-generation", type=Path, help="Reuse recorded upstream/proposal text only; run fresh council. Explicitly logged, not a fresh full-model repeat.")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out / "run.log")])
    from flora_translate.intake_agent import intake_context_block, IntakeAgent
    from flora_translate.main import translate
    from flora_translate.gui_autosave import autosave_gui_result
    from flora_translate.engine.llm_agents import set_llm_observer, set_llm_runtime_overrides
    from flora_translate.schemas import DesignInputPackage, IntakeAnswer
    source = Path("outputs/flowpilot2/20260907_requested_three_protocols/attempt_01/03_giese_oxidation/intake_package.json")
    package = DesignInputPackage.model_validate_json(source.read_text())
    package.inventory_constraints["standard_reactor_connectors_available"] = True
    if args.oxygen_stage == 1:
        if args.replay_generation:
            replay_request = json.loads((args.replay_generation.parent / "request.json").read_text())
            if replay_request.get("requested_oxygen_stage") != 1:
                raise ValueError("The changed Stage-1 instruction cannot reuse Stage-2 generation")
        package = IntakeAgent().analyze(existing_package=package, use_llm=False, answers=[
            IntakeAnswer(question_id="Q-GAS-003", answer="Introduce oxygen at the inlet of Stage 1, into Reactor 1.",
                         source="user_clarification_20260908")])
        if not package.ready_for_design or package.engineering_requirements["gas"].get("introduction_stage") != 1:
            raise ValueError("Revised intake is not ready or does not bind oxygen to Reactor 1")
        shutil.copy2(source, out / "original_intake_package.json")
    # A proposed flow-feed adaptation is not a modified batch fact or user answer.
    gas = package.engineering_requirements["gas"]
    gas.update(species="O2", reagent_mole_fraction=1.0, batch_species="air",
               identity_source="inventory_screening_proposal", requires_chemist_confirmation=True,
               rationale="KHU v4 declares an O2 MFC, not an air-calibrated device. Pure O2 is proposed for review; the user has not confirmed this change.")
    if args.oxygen_stage == 1:
        gas["introduction_conflict_note"] = "The user requests evaluating oxygen at the Reactor 1 inlet. This departs from the source batch's argon-protected Giese stage. Assess chemical compatibility explicitly; do not silently reroute oxygen to Stage 2. The request is not experimental evidence of compatibility."
    package.hypotheses.append(f"Proposed inventory adaptation, not confirmed: pure O2 delivery at Stage {args.oxygen_stage} instead of batch air. Requires laboratory approval and safety review; do not infer equivalent selectivity or performance.")
    raw = package.model_dump()
    runtime = {"design_policy": "scientific_v2", "candidate_budget": 12,
               "upstream_model": "claude-opus-4-6", "downstream_model": "claude-sonnet-4-6"}
    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2, default=str))
    save("intake_package.json", raw)
    save("request.json", {"intake_package": raw, "runtime_options": runtime, "original_package_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "gas_change_is_user_confirmed": False, "fresh_upstream_and_council": args.replay_generation is None,
                          "requested_oxygen_stage": args.oxygen_stage,
                          "generation_replayed_from": str(args.replay_generation) if args.replay_generation else None})
    snapshot = out / "source_snapshot"
    for base in (Path("flora_translate"), Path("flowpilot_webapp/frontend/src")):
        for file in base.rglob("*"):
            if file.suffix not in {".py", ".txt", ".tsx", ".ts"} or "__pycache__" in file.parts:
                continue
            dest = snapshot / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    (out / "prompt.txt").write_text(intake_context_block(package))
    set_llm_runtime_overrides(capture_content=True)
    def observe(event):
        with (out / "llm_calls.jsonl").open("a") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
    set_llm_observer(observe)
    started = time.monotonic()
    before = {p for base in ("outputs/scientific_pipeline", "outputs/scientific_council") for p in Path(base).glob("*")}
    success = False
    try:
        with ExitStack() as stack:
            if args.replay_generation:
                events = [json.loads(line) for line in args.replay_generation.read_text().splitlines()]
                reused = [next(e for e in events if e["api_name"] == "chemistry_agent"),
                          next(e for e in reversed(events) if e["api_name"].startswith("translation_llm"))]
                save("reused_generation_records.json", reused)
                for module, event in zip(("flora_translate.chemistry_agent", "flora_translate.translation_llm"), reused):
                    stack.enter_context(patch(module + ".call_model_text", return_value=SimpleNamespace(text=event["response_text"], stop_reason=event.get("stop_reason"), finish_reason=None)))
            result = translate(package.raw_protocol, intake_package=raw, runtime_options=runtime)
        save("result.json", result)
        archive = Path(autosave_gui_result(result, intake_package=raw, source="scientific_giese", user_input=package.raw_protocol))
        shutil.copytree(archive, out / "gui_export")
        assessment = result.get("scientific_assessment", {})
        summary = {"status": result.get("final_design", {}).get("status"), "gui_archive": str(archive),
                   "requested_oxygen_stage": args.oxygen_stage,
                   "candidate": assessment.get("selected_candidate_id"), "candidates": assessment.get("candidate_count"),
                   "stages": result.get("proposal", {}).get("stage_parameters"),
                   "elapsed_seconds": time.monotonic()-started, "gas_change_is_user_confirmed": False}
        save("summary.json", summary)
        success = summary["status"] == "executable"
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        (out / "failure.txt").write_text(traceback.format_exc())
        save("failure.json", {"error": str(error), "rejections": getattr(error, "rejections", []), "elapsed_seconds": time.monotonic()-started})
        print(type(error).__name__, str(error)[:500], flush=True)
    finally:
        after = {p for base in ("outputs/scientific_pipeline", "outputs/scientific_council") for p in Path(base).glob("*")}
        for path in sorted(after-before):
            shutil.copytree(path, out / "model_artifacts" / path.parent.name / path.name)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
