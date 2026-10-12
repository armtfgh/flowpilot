#!/usr/bin/env python3
"""Fresh, fixed-size development pilot after the audited production repairs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts import run_astra_repair_pilot as pilot

base_configure = pilot.configure


def configure():
    base_configure()
    pilot.campaign.SEED_NAMESPACE = "astra-verified-repair-development-20261012-v3"
    pilot.evaluation.configure = configure


pilot.configure = configure


def source_files():
    for directory in ("flora_translate", "ablation_test/src", "ablation_test/scripts"):
        yield from sorted((ROOT / directory).rglob("*.py"))


def verify_source(output):
    expected = pilot.read(output / "frozen/full_source_manifest.json")
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    if expected != actual:
        raise RuntimeError("Production/evaluation source changed after freeze; start a new development round")


def initialize(output, budget):
    if output.exists():
        raise RuntimeError("Use a new directory; no existing campaign may be replaced")
    pilot.initialize(output)
    frozen = output / "frozen"
    hashes = {}
    for path in source_files():
        relative = path.relative_to(ROOT)
        target = frozen / "full_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    pilot.write(frozen / "full_source_manifest.json", hashes)
    manifest = pilot.read(frozen / "campaign_manifest.json")
    manifest.update(benchmark_name="Astra verified-repair development pilot v3",
        evaluation_packet_version="flowpilot_newgen_2_0_blinded_outcome_v1.1",
        budget_stop_threshold_usd=budget, development_cases=True,
        evidence_scope="Cases were inspected during development. A fresh matched run is development evidence, not independent confirmation.",
        stopping_rule="Exactly six fresh outcomes and eighteen judgments. Retain every result; no score-dependent retry or exclusion. Stop on infrastructure failure or observed budget threshold.")
    pilot.write(frozen / "campaign_manifest.json", manifest)
    (frozen / "PREREGISTRATION.md").write_text(
        "# Verified-repair Astra development pilot\n\n"
        "Three previously examined cases, two architectures, one fresh generation per cell. "
        "Both generators use Astra medium reasoning with the same authoritative input. "
        "FlowPilot uses the legacy two-candidate configuration and source-excluded lexical retrieval. "
        "These cases informed software repairs, so this is development evidence. "
        "The fixed 14-criterion rubric and the Qwen, GPT-4o and Claude judge panel are retained. "
        "Both architectures use the v1.1 outcome adapter and its common arithmetic worksheet. "
        "The previous campaign used an earlier adapter; its scores will not be pooled with this run. "
        "All generation, packet construction and evaluation source is frozen before generation. "
        "Run exactly six generation cells and eighteen accepted judge assessments. "
        "No low-score retries, selective exclusions, or production-code changes during the run. "
        "Retain failed attempts, schema retries, blocked designs and costs. "
        "A higher score is an observed development result, not a stopping-rule-based claim of general superiority. "
        "Any further production change requires a separately frozen development round; every round remains visible. "
        "Independent confirmation requires a prospectively fixed repeated comparison and preferably additional cases.\n\n"
        f"Observed returned-usage stop threshold: ${budget:.2f}. In-flight calls can add cost before it is observed. "
        "Rates follow the archived same-day pilot estimate; no local hardware cost is assigned.\n")
    (frozen / "CONFIGURATION_SCOPE.md").write_text(
        "Legacy production path; two initial candidates; lexical source-excluded retrieval; Astra medium. "
        "Repaired production code and v1.1 evaluation adapter, both frozen before generation. "
        "This is a development pilot and is not the scientific_v2 twelve-candidate GUI configuration.\n")


def workers(output, phase, items):
    processes, handles = [], []
    budget = pilot.read(output / "frozen/campaign_manifest.json")["budget_stop_threshold_usd"]
    try:
        for item in items:
            handle = (output / f"verified_{phase}_{item}.log").open("a", buffering=1)
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--output", str(output), "--phase", phase, "--item", str(item)], cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            state = pilot.progress(output, phase)
            if state["estimated_api_cost_usd"] >= budget:
                raise RuntimeError("Observed campaign usage reached the frozen spending threshold")
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A worker failed; preserve attempts and inspect before further calls")
            time.sleep(5)
        codes = [p.returncode for p in processes]
        pilot.write(output / f"verified_{phase}_exit_codes.json", codes)
        if any(codes):
            raise RuntimeError(f"Incomplete {phase} phase; all attempts retained")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("init", "generate", "prepare", "judge", "report", "run"), default="run")
    parser.add_argument("--item")
    parser.add_argument("--budget-usd", type=float, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    configure()
    if args.phase == "init":
        initialize(output, args.budget_usd)
        return
    verify_source(output)
    if args.phase == "generate":
        pilot.campaign.run_generation(output, [pilot.campaign.selected_cases()[int(args.item)]], ("astra",), 0)
    elif args.phase == "prepare":
        pilot.evaluation.prepare(output)
    elif args.phase == "judge":
        if args.item == "qwen":
            from flora_translate.engine import llm_agents
            llm_agents._bounded_local_messages = lambda system, messages, max_tokens: pilot.evaluation.exact_local_context(
                system, messages, max_tokens=max_tokens, audit_path=output / "judge_context_checks.jsonl")
        pilot.campaign.run_judges(output, (args.item,), 0)
    elif args.phase == "report":
        pilot.report(output)
    else:
        try:
            workers(output, "generate", range(3))
            verify_source(output)
            pilot.evaluation.prepare(output)
            workers(output, "judge", pilot.campaign.JUDGES)
            verify_source(output)
            pilot.report(output)
        except Exception as exc:
            pilot.write(output / "pilot_interruption.json", {"at": pilot.now(), "error_type": type(exc).__name__, "message": str(exc)})
            pilot.progress(output, "interrupted")
            raise


if __name__ == "__main__":
    main()
