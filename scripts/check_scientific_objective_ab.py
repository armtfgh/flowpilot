"""Controlled intake A/B: frozen upstream, fresh council, full downstream closure.

This isolates objective/hypothesis changes from upstream sampling variation.
It is not an independent full-model repeat or a wet-lab yield benchmark.
"""
import argparse
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-snapshot", type=Path, required=True)
    parser.add_argument("--throughput-probe", action="store_true", help="Synthetic objective-sensitivity test, not an original user run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.output / "run.log"), logging.StreamHandler()])
    names = ["20260907_164042_webapp", "20260907_165126_webapp"]
    sources = [Path("outputs/gui_runs") / n / "result.json" for n in names]
    originals = [json.loads(p.read_text()) for p in sources]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
    from flora_translate.main import translate
    from flora_translate.schemas import ChemistryPlan, FlowProposal
    from flora_translate.gui_autosave import autosave_gui_result
    frozen = originals[0]
    upstream = json.loads(args.upstream_snapshot.read_text())
    summary = []
    cases = list(zip(("A_balanced", "B_yield"), originals, sources))
    if args.throughput_probe:
        probe = deepcopy(originals[0])
        pkg = probe["intake_package"]
        pkg["screening_priority"] = "throughput_priority"
        pkg["objective"] = "Prioritize feed throughput in the first exploratory screen. Compare shortened stage holds with batch-anchored controls while preserving sequential chemistry, inventory compatibility and operating limits. Do not claim product throughput before measuring final amide yield."
        for answer in pkg["answers"]:
            if answer["question_id"] == "Q-OBJ-001":
                answer["answer"] = pkg["objective"]
                answer["source"] = "synthetic_objective_sensitivity_probe"
        cases = [("C_throughput_probe", probe, sources[0])]
    for label, original, source in cases:
        out = args.output / label
        out.mkdir()
        def save(name, value):
            (out / name).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        intake = original["intake_package"]
        options = {"design_policy": "scientific_v2", "candidate_budget": 12,
                   **frozen["pipeline_runtime"]["model_routing"]}
        provenance = {"type": "controlled intake A/B; frozen upstream and translation, fresh six-agent council",
                      "source": str(source), "frozen_upstream_source": str(args.upstream_snapshot),
                      "upstream_snapshot_sha256": hashlib.sha256(args.upstream_snapshot.read_bytes()).hexdigest(),
                      "fresh_upstream": False, "fresh_council": True, "runtime_options": options}
        provenance["synthetic_objective_probe"] = args.throughput_probe
        save("request.json", {"intake_package": intake, **provenance})
        start = time.monotonic()
        try:
            with ExitStack() as stack:
                stack.enter_context(patch("flora_translate.main.analyze_batch_chemistry", side_effect=lambda *a, **k: ChemistryPlan.model_validate(upstream["chemistry_plan"])))
                stack.enter_context(patch("flora_translate.main.VectorRetriever.retrieve", return_value=frozen["_analogies"]))
                stack.enter_context(patch("flora_translate.main.AnalogySelector.select", return_value=frozen["_analogies"]))
                stack.enter_context(patch("flora_translate.main.TranslationLLM.generate", side_effect=lambda *a, **k: FlowProposal.model_validate(frozen["engineering_history"]["before_council"]["proposal"])))
                result = translate(intake["raw_protocol"], intake_package=intake, runtime_options=options)
            result["verification_provenance"] = provenance
            archive = autosave_gui_result(result, intake_package=result["intake_package"], source="objective_ab")
            save("result.json", result)
            a = result["scientific_assessment"]
            # Verify actual sent requests, not merely the saved answer field.
            assert len(a["calls"]) == 6
            assert all(c["request"]["context"]["objective_policy"] == a["objective_policy"] for c in a["calls"])
            assert all(c["request"]["context"]["objective"] == intake["objective"] for c in a["calls"])
            assert all(all(h in c["request"]["context"]["authority_labeled_intake"] for h in intake["hypotheses"]) for c in a["calls"])
            assert a["selected_design_preserved"]
            assert result["final_design"]["status"] == "executable"
            row = {"label": label, "status": result["final_design"]["status"], "priority": a["objective_policy"]["priority"],
                   "pool_sha256": a["pool_sha256"], "selected": a["selected_candidate_id"],
                   "stages": result["final_design"]["stages"], "alignment": a["chief"]["objective_alignment"],
                   "gui_archive": str(archive), "elapsed_seconds": time.monotonic() - start, **provenance}
            save("summary.json", row)
            summary.append(row)
        except Exception:
            (out / "failure.txt").write_text(traceback.format_exc())
            logging.exception("A/B case failed; preserving evidence")
            summary.append({"label": label, "status": "failed", "elapsed_seconds": time.monotonic() - start})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(label, summary[-1]["status"], flush=True)
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
    return 0 if all(r["status"] == "executable" for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
