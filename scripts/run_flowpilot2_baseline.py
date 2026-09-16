"""Run or replay the private scientific policy against an immutable saved intake."""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/gui_runs/20260907_145525_webapp/result.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true", help="Verify candidate generation only; no fresh model calls")
    parser.add_argument("--upstream", default="claude-opus-4-6")
    parser.add_argument("--downstream", default="claude-sonnet-4-6")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.output / "run.log"), logging.StreamHandler()])
    raw = args.source.read_bytes()
    old = json.loads(raw)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    runtime = {"design_policy": "scientific_v2", "candidate_budget": 12,
               "upstream_model": args.upstream, "downstream_model": args.downstream}
    save("request.json", {"intake_package": old["intake_package"], "runtime_options": runtime,
                           "source_sha256": hashlib.sha256(raw).hexdigest(), "fresh_generation": not args.replay})
    start = time.monotonic()
    try:
        if args.replay:
            from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
            from flora_translate.chemistry_contract import reconcile_chemistry_plan
            from flora_translate.engine.council_v4.scientific import build_screen_pool
            b = BatchRecord.model_validate(old["batch_record"])
            plan, audit = reconcile_chemistry_plan(b, ChemistryPlan.model_validate(old["chemistry_plan"]), scientific=True)
            pool, rejected = build_screen_pool(FlowProposal.model_validate(old["proposal"]), b, plan, LabInventory.model_validate(old["inventory_snapshot"]))
            result = {"source_evidence": plan.scientific_context, "reconciliation": audit, "candidates": pool, "rejected": rejected}
            summary = {"fresh_generation": False, "candidate_count": len(pool), "source_issues": plan.scientific_context["issues"]}
        else:
            from flora_translate.main import translate
            result = translate(old["intake_package"]["raw_protocol"], intake_package=old["intake_package"], runtime_options=runtime)
            from flora_translate.gui_autosave import autosave_gui_result
            archive = autosave_gui_result(result, intake_package=old["intake_package"], source="scientific_check",
                                         user_input=old["intake_package"]["raw_protocol"])
            a = result.get("scientific_assessment") or result.get("proposal", {}).get("scientific_design", {})
            summary = {"fresh_generation": True, "status": result.get("final_design", {}).get("status"),
                       "gui_archive": str(archive),
                       "candidate_count": a.get("candidate_count"), "selected_candidate_id": a.get("selected_candidate_id"),
                       "selected_design_preserved": a.get("selected_design_preserved"),
                       "stage_parameters": result.get("proposal", {}).get("stage_parameters"),
                       "issues": result.get("final_design", {}).get("consistency", {}).get("issues", [])}
        save("result.json", result)
        summary["elapsed_seconds"] = time.monotonic() - start
        save("summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        assert args.source.read_bytes() == raw, "Baseline was mutated"
        return 0
    except Exception:
        (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        logging.exception("Baseline check failed; failure preserved")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
