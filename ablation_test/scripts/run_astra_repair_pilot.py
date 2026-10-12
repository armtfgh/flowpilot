#!/usr/bin/env python3
"""One-repeat Astra repair pilot, preserving delivered-output evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts import run_astra_matched_benchmark as astra
from ablation_test.scripts import evaluate_astra_delivered_outputs as evaluation
from ablation_test.scripts.build_newgen_2_0_report import aggregate

campaign = astra.campaign


def configure():
    astra.configure()
    campaign.REPEAT_IDS = ("repeat_01",)
    campaign.SEED_NAMESPACE = "astra-inventory-repair-pilot-20261012-v2"
    evaluation.configure = configure


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def usage(output):
    rows = []
    for path in sorted((output / "generation").rglob("llm_events.jsonl")):
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # A running writer may have an incomplete last line.
            u = e.get("usage") or {}
            i, o = u.get("input_tokens", 0), u.get("output_tokens", 0)
            d = u.get("input_tokens_details") or {}
            c, w = d.get("cached_tokens", 0), d.get("cache_write_tokens", 0)
            fi, fo = (2, 1.5) if i > 272000 else (1, 1)
            cost = (fi * (max(0, i-c-w)*10 + c + w*12.5) + fo*o*50) / 1e6
            rows.append({"scope": "generation", "source": str(path.relative_to(output)),
                         "model": e.get("model"), "returned_model": e.get("returned_model"),
                         "response_id": e.get("response_id"), "input_tokens": i,
                         "output_tokens": o, "estimated_usd": cost})
    for path in sorted((output / "judgments").rglob("telemetry.json")):
        try:
            events = read(path)
        except json.JSONDecodeError:
            continue
        for e in events:
            u = e.get("usage") or {}
            i = u.get("input_tokens", u.get("prompt_tokens", 0))
            o = u.get("output_tokens", u.get("completion_tokens", 0))
            model = e.get("model", "")
            if "claude" in model:
                cost = (i*3 + o*15 + u.get("cache_read_input_tokens", 0)*.3
                        + u.get("cache_creation_input_tokens", 0)*3.75) / 1e6
            elif model == "gpt-4o":
                c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                cost = ((i-c)*2.5 + c*1.25 + o*10) / 1e6
            else:
                cost = 0.0  # Local Qwen has no per-call API charge.
            rows.append({"scope": "judging", "source": str(path.relative_to(output)),
                         "model": model, "input_tokens": i, "output_tokens": o,
                         "estimated_usd": cost})
    totals = Counter()
    for r in rows:
        totals[r["scope"]] += r["estimated_usd"]
    return {"generation_usd": totals["generation"], "judging_usd": totals["judging"],
            "total_usd": sum(totals.values()), "recorded_calls": len(rows), "calls": rows,
            "scope": "Recorded API usage, including retained retries; excludes local hardware, taxes, and unreturned billable failures."}


def completed_judgments(output):
    """Count a successful format retry once, rather than its failed root call."""
    valid = set()
    for path in (output / "judgments").rglob("status.json"):
        try:
            if read(path).get("status") == "valid":
                valid.add(tuple(path.relative_to(output / "judgments").parts[:2]))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return len(valid)


def progress(output, phase):
    generations = []
    for path in sorted((output / "generation").glob("*/*/repeat_*/run_summary.json")):
        try:
            d = read(path)
            result = read(path.with_name("result.json")) if path.with_name("result.json").exists() else {}
        except json.JSONDecodeError:
            continue
        generations.append({"case": path.parts[-4], "condition": path.parts[-3],
                            "status": d.get("status"),
                            "final_contract_status": (result.get("final_design") or {}).get("status")})
    valid = completed_judgments(output)
    costs = usage(output)
    state = {"updated_at": now(), "phase": phase, "planned_generation": 6,
             "completed_generation": sum(r["status"] == "completed" for r in generations),
             "planned_judgments": 18, "valid_judgments": valid, "runs": generations,
             "estimated_api_cost_usd": costs["total_usd"], "recorded_api_calls": costs["recorded_calls"]}
    write(output / "pilot_progress.json", state)
    write(output / "usage_so_far.json", costs)
    print(json.dumps(state), flush=True)
    return state


def initialize(output):
    configure()
    output.mkdir(parents=True, exist_ok=True)
    cases = campaign.selected_cases()
    baseline = ROOT / "ablation_results/manuscript_benchmark/astra_matched_streaming_20261001/frozen"
    old = {r["case_id"]: r["design_input_sha256"] for r in read(baseline / "execution_plan.json")["cells"]}
    assert old == {c.case_id: c.design_input_sha256 for _, c in cases}
    astra.freeze(output, cases)
    frozen = output / "frozen"
    manifest = read(frozen / "campaign_manifest.json")
    manifest.update(benchmark_name="GPT-6 Astra inventory-repair one-repeat pilot",
                    repeat_count=1, candidate_count=6, planned_judgments=18,
                    independent_call_policy="One fresh generation per case and architecture. No temperature or seed sent; saved seeds are bookkeeping only.",
                    pilot=True, within_case_repeatability_estimated=False,
                    estimated_api_budget_usd=[10, 12], budget_stop_threshold_usd=12,
                    budget_note="Supervisor stops at $12 of observed returned usage; concurrent in-flight calls can add charges before observation.")
    write(frozen / "campaign_manifest.json", manifest)
    (frozen / "PREREGISTRATION.md").write_text(
        "# Astra repair pilot\n\nOne fresh generation for each of three frozen chemistry cases in two architectures, giving six outcomes and eighteen independent judge assessments. "
        "Astra medium reasoning is used for both one-shot and all FlowPilot model roles. FlowPilot retains the legacy policy setting, two initial candidates, original revision limits, and held-out-source lexical retrieval. Single-stage candidates are now realized on the authoritative inventory before council review, and the reviewed configuration is preserved through delivery. Explicit council rejections are retained. "
        "The repaired production code is frozen before generation. The cases, authoritative inputs, fourteen-criterion rubric, and Qwen3.6/GPT-4o/Sonnet4.6 judges match the previous completed Astra campaign. "
        "Raw delivered one-shot values and actual final FlowPilot outputs are evaluated without benchmark-side numerical repair. "
        "No low-score retries, outcome-dependent exclusions, or changes to production code are allowed during this pilot. Existing generation and judge format-failure retries remain recorded. "
        "One repeat cannot estimate generation repeatability. All outcomes, including blocked designs, are retained. "
        "This is a repair diagnostic and not a test of the GUI twelve-candidate configuration.\n\n"
        "API cost estimate: $10-12. The supervisor checks observed usage every 15 seconds and stops at $12; this is not a billing-system hard cap and in-flight calls may add cost. "
        "No score is produced for an infrastructure-interrupted partial campaign.\n")
    for name in [Path(__file__).name, "evaluate_astra_delivered_outputs.py"]:
        src = Path(__file__).with_name(name)
        dst = frozen / "source_code" / name
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise RuntimeError("Pilot driver changed after freezing")
        shutil.copy2(src, dst)
    write(frozen / "pilot_source_checksums.json", {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (frozen / "source_code").glob("*.py")})
    write(frozen / "pricing.json", {
        "date": "2026-10-12", "currency": "USD", "basis": "per million tokens, standard API rates",
        "astra": {"input": 10, "cached_input": 1, "cache_write": 12.5, "output": 50,
                  "source": "https://developers.openai.com/api/docs/models/gpt-6-astra"},
        "gpt4o": {"input": 2.5, "cached_input": 1.25, "output": 10,
                  "source": "https://developers.openai.com/api/docs/models/gpt-4o"},
        "sonnet46": {"input": 3, "output": 15, "cache_read": .3, "cache_write_5m": 3.75,
                     "source": "https://platform.claude.com/docs/en/models/sonnet-4-6/overview"},
        "qwen": "Local server; no API charge included; hardware cost not estimated."})
    readiness = ROOT / "tmp/astra_readiness_20261012"
    if not (output / "preflight.json").exists():
        for name in ["preflight.json", "preflight_llm_events.json", "endpoint_health.json"]:
            shutil.copy2(readiness / name, output / name)
        write(frozen / "preflight_provenance.json", {
            "reused_from": str(readiness), "reason": "Successful live preflight earlier on the same day with unchanged provider code and endpoints.",
            "excluded_from_pilot_cost": True})
    assert len(read(frozen / "execution_plan.json")["cells"]) == 6
    progress(output, "initialized")


def run_workers(output, phase, items):
    processes, handles = [], []
    try:
        for item in items:
            handle = (output / f"pilot_{phase}_{item}.log").open("a", buffering=1)
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--output", str(output), "--phase", phase, "--item", str(item)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            state = progress(output, phase)
            if state["estimated_api_cost_usd"] >= 12:
                raise RuntimeError("Observed API spending reached the pilot budget threshold; no further calls authorized by this runner")
            for error in (output / "generation").glob("*/*/repeat_*/error.json"):
                text = str(read(error).get("error", "")).lower()
                if any(x in text for x in ["no credits remaining", "insufficient_quota", "exceeded your current quota"]):
                    raise RuntimeError("Pilot interrupted by API credit exhaustion")
            time.sleep(15)
        codes = [p.returncode for p in processes]
        write(output / f"pilot_{phase}_exit_codes.json", codes)
        progress(output, phase + "_finished")
        if phase == "judge" and any(codes):
            raise RuntimeError("Judge calls remain incomplete; see retained worker logs")
        return codes
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
        for h in handles:
            h.close()


def report(output):
    target = output / "report"
    aggregate(output, target)
    scores = list(csv.DictReader((target / "tables/candidate_consensus_scores.csv").open()))
    key = read(output / "frozen/candidate_key_confidential.json")["candidates"]
    costs = usage(output)
    write(target / "api_usage.json", costs)
    pairs = []
    for case in campaign.SELECTED_CASES:
        values = {r["architecture"]: float(r["mean_score_0_1"]) for r in scores if r["case"] == case}
        assert set(values) == {"One-shot", "FlowPilot"}
        pairs.append({"case": case, "one_shot": values["One-shot"], "flowpilot": values["FlowPilot"],
                      "difference": values["FlowPilot"] - values["One-shot"]})
    contracts = []
    for cell in key:
        run = Path(cell["run_directory"])
        result = read(run / "result.json") if (run / "result.json").exists() else {}
        final = result.get("final_design") or {}
        contracts.append({"case": cell["case"], "architecture": cell["architecture"],
                          "status": final.get("status"), "issues": (final.get("consistency") or {}).get("issues", [])})
        artifacts = (result.get("diagram_artifacts") or {}).get("run_dir")
        if artifacts and Path(artifacts).is_dir():
            shutil.copytree(artifacts, output / "topologies" / cell["candidate_id"], dirs_exist_ok=True)
    summary = {"completed_at": now(), "outcomes": len(scores), "judgments": 18, "repeats_per_case": 1,
               "case_results": pairs, "one_shot_mean": statistics.mean(r["one_shot"] for r in pairs),
               "flowpilot_mean": statistics.mean(r["flowpilot"] for r in pairs),
               "contracts": contracts, "estimated_api_cost_usd": costs["total_usd"],
               "scope": "One-repeat repair pilot. No within-case repeat SD or laboratory validation."}
    write(target / "pilot_summary.json", summary)
    lines = ["# Astra repair pilot results", "", "One fresh design per architecture and chemistry; three judges per design. Six outcomes and eighteen judgments.", "",
             "| Chemistry | One-shot | FlowPilot | Difference |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['case']} | {r['one_shot']:.4f} | {r['flowpilot']:.4f} | {r['difference']:+.4f} |" for r in pairs]
    lines += ["", f"Mean across the three cases: one-shot {summary['one_shot_mean']:.4f}; FlowPilot {summary['flowpilot_mean']:.4f}.",
              f"Recorded API usage estimate: ${costs['total_usd']:.2f}, including generation and paid judges, excluding local hardware.", "",
              "Only one generation per case and architecture was performed, so repeatability cannot be estimated. Scores are judge assessments. Software contract status is distinct from experimental success.", "",
              "The frozen inputs, code, configuration, prompts, full responses, failed attempts, intermediate snapshots, calculations, diagrams, judge rationales, and token usage are retained in this campaign. "
              "Production sources were not changed during the pilot. The configuration retains the legacy two-candidate policy."]
    (target / "PILOT_RESULTS.md").write_text("\n".join(lines) + "\n")
    progress(output, "complete")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["init", "generate", "prepare", "judge", "report", "all"], default="all")
    parser.add_argument("--item")
    args = parser.parse_args()
    output = args.output.resolve()
    configure()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.phase in ["init", "all"]:
        initialize(output)
    if args.phase == "init":
        return
    if args.phase == "generate":
        campaign.run_generation(output, [campaign.selected_cases()[int(args.item)]], ("astra",), 0)
    elif args.phase == "prepare":
        evaluation.prepare(output)
        assert len(read(output / "frozen/candidate_key_confidential.json")["candidates"]) == 6
    elif args.phase == "judge":
        if args.item == "qwen":
            from flora_translate.engine import llm_agents
            llm_agents._bounded_local_messages = lambda system, messages, max_tokens: evaluation.exact_local_context(
                system, messages, max_tokens=max_tokens, audit_path=output / "judge_context_checks.jsonl")
        campaign.run_judges(output, (args.item,), 0)
    elif args.phase == "report":
        report(output)
    elif args.phase == "all":
        try:
            run_workers(output, "generate", range(3))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--phase", "prepare"], check=True)
            run_workers(output, "judge", campaign.JUDGES)
            report(output)
        except Exception as e:
            write(output / "pilot_interruption.json", {"at": now(), "error_type": type(e).__name__, "message": str(e)})
            progress(output, "interrupted")
            raise


if __name__ == "__main__":
    main()
