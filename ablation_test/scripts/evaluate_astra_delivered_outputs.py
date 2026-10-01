#!/usr/bin/env python3
"""Judge unmodified delivered outputs, retaining a packet-normalization audit."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ablation_test.scripts.run_astra_matched_benchmark import configure, campaign
from ablation_test.scripts.run_manuscript_five_case_benchmark import deterministic_sheet
from ablation_test.src.newgen_2_outcome import build_outcome_packet
from ablation_test.src.newgen_2_outcome import judge_prompt


def exact_local_context(system, messages, *, max_tokens, audit_path):
    """Never silently trim a judge's evidence using a character-count heuristic."""
    from ablation_test.scripts.run_newgen_2_0_benchmark import JUDGES
    bundle = JUDGES["qwen"]
    routed = copy.deepcopy(messages)
    if routed and isinstance(routed[-1].get("content"), str):
        routed[-1]["content"] = "/no_think\n" + routed[-1]["content"]
    payload = {"model": bundle["model"], "messages": [{"role": "system", "content": system}, *routed],
               "chat_template_kwargs": {"enable_thinking": False}, "return_token_strs": False}
    request = urllib.request.Request(
        bundle["base_url"].removesuffix("/v1") + "/tokenize",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        tokenized = json.load(response)
    count = tokenized["count"]
    limit = tokenized.get("max_model_len", 32768)
    allowed = count + max_tokens + 2048 <= limit
    with audit_path.open("a") as handle:
        handle.write(json.dumps({"input_tokens": count, "output_allowance": max_tokens,
                                 "context_limit": limit, "safety_margin": 2048,
                                 "full_prompt_fits": allowed, "truncated": False,
                                 "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}) + "\n")
    if not allowed:
        raise RuntimeError(f"Complete judge prompt cannot fit: {count} input + {max_tokens} output + 2048 margin > {limit}; evidence was not truncated")
    return system, messages


def delivered_result(result):
    value = copy.deepcopy(result)
    if value.get("variant") == "general_one_shot" and isinstance(value.get("raw_proposal"), dict):
        value["proposal"] = copy.deepcopy(value["raw_proposal"])
    return value


def delivered_packet(**kwargs):
    return build_outcome_packet(**{**kwargs, "result": delivered_result(kwargs["result"])})


def delivered_verification(result, metrics, case):
    result = delivered_result(result)
    sheet = deterministic_sheet(result, metrics, case)
    if case.expected_features.get("gas_required"):
        old = sheet["top_level_volume_flow_time_closure"]
        sheet["top_level_volume_flow_time_closure"] = {
            "assessable": False,
            "reason": "A single liquid-flow-times-tau verdict is not valid for a gas process without matching the declared time basis. Check labeled STP turnover, channel turnover, empty-bed space time, and liquid-holdup contact time separately against the delivered parameters.",
            "arithmetic_only": {k: v for k, v in old.items() if k not in {"assessable", "passes_10_percent", "relative_error"}},
            "passes_10_percent": None,
        }
    sheet["evaluation_provenance"] = "Unmodified delivered parameters; no numerical repair by benchmark normalization. Basis-ambiguous gas residence-time closure is judge-assessed, not assigned an automatic pass/fail."
    return sheet


def prepare(output):
    if any((output / "judgments").glob("*/*/status.json")):
        raise RuntimeError("Do not replace evaluation packets after judging has started")
    configure()
    changes = []
    for path in sorted((output / "generation").glob("*/astra_one_shot/repeat_*/result.json")):
        result = campaign.read_json(path)
        raw, normalized = result.get("raw_proposal") or {}, result.get("proposal") or {}
        changes.append({"result": str(path.relative_to(output)),
                        "changed_fields": {k: {"raw": v, "normalized": normalized.get(k)}
                                           for k, v in raw.items() if v != normalized.get(k)}})
    campaign.write_json(output / "normalization_audit.json", changes)
    campaign.build_outcome_packet = delivered_packet
    campaign.deterministic_sheet = delivered_verification
    campaign.build_packets(output, campaign.selected_cases())
    rubric = campaign.read_json(output / "frozen/outcome_rubric.json")
    for path in sorted((output / "packets").glob("*.json")):
        system, user = judge_prompt(rubric, campaign.read_json(path))
        exact_local_context(system, [{"role": "user", "content": user}], max_tokens=12000,
                            audit_path=output / "judge_context_preflight.jsonl")
    amendments = output / "frozen/amendments"
    amendments.mkdir(parents=True, exist_ok=True)
    target = amendments / Path(__file__).name
    if target.exists() and campaign.sha256_file(target) != campaign.sha256_file(Path(__file__)):
        raise RuntimeError("Frozen evaluation adapter differs")
    shutil.copy2(Path(__file__), target)
    campaign.write_json(amendments / "delivered_output_evaluation.json", {
        "made_before_judging": True,
        "generation_changes": False,
        "rubric_changes": False,
        "judge_prompt_changes": False,
        "reason": "Legacy one-shot Pydantic normalization rewrote declared residence times/bases and could discard extra fields. Legacy mechanical Q_liquid*tau checks ignored gas time bases. Evaluate actual raw one-shot JSON versus actual delivered FlowPilot final contract, without automatic repairs to either. Use exact server token counts for the Qwen judge and never truncate evidence using the legacy character-count heuristic.",
        "scope": "Packet construction and a basis-ambiguous deterministic diagnostic only. Historical published scores are not silently replaced.",
        "evaluation_adapter_sha256": campaign.sha256_file(target),
    })
    manifest = campaign.read_json(output / "frozen/campaign_manifest.json")
    manifest["evaluation_amendment"] = "amendments/delivered_output_evaluation.json"
    campaign.write_json(output / "frozen/campaign_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=("prepare", "judge", "report", "all"), default="all")
    parser.add_argument("--judge", choices=("qwen", "openai", "claude"))
    args = parser.parse_args()
    configure()
    output = args.output.resolve()
    if args.phase in {"prepare", "all"}:
        prepare(output)
    if args.phase in {"judge", "all"}:
        if args.judge:
            if args.judge == "qwen":
                from flora_translate.engine import llm_agents
                llm_agents._bounded_local_messages = lambda system, messages, max_tokens: exact_local_context(
                    system, messages, max_tokens=max_tokens, audit_path=output / "judge_context_checks.jsonl",
                )
            campaign.run_judges(output, (args.judge,), 1)
        else:
            import subprocess
            processes, files = [], []
            try:
                for judge in campaign.JUDGES:
                    handle = (output / f"worker_judging_{judge}.log").open("a")
                    files.append(handle)
                    processes.append(subprocess.Popen([
                        sys.executable, "-u", str(Path(__file__).resolve()), str(output),
                        "--phase", "judge", "--judge", judge,
                    ], stdout=handle, stderr=subprocess.STDOUT))
                codes = [p.wait() for p in processes]
                campaign.write_json(output / "judge_worker_exit_codes.json", codes)
                if any(codes):
                    raise RuntimeError("One or more judge workers failed; retain attempts and resume failed judge only")
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                        p.wait()
                for handle in files:
                    handle.close()
    if args.phase in {"report", "all"}:
        campaign.build_repeated_report(output, output / "report")
        from ablation_test.scripts.summarize_astra_benchmark import summarize
        summarize(output)


if __name__ == "__main__":
    main()
