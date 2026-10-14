"""Verify a completed repair campaign offline; never generates or judges.

Run only after run_practical_repair_benchmark.py --phase report has finished.
Writes verification.json and artifact_manifest.json inside the chosen campaign.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_practical_repair_benchmark as driver
from ablation_test.src.newgen_2_outcome import _scrub, _pipeline_outcome, _one_shot_outcome, packet_gate, validate_response
from ablation_test.src.outcome_verification import resolves_json_path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def equal_float(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def no_network(*args, **kwargs):
    raise AssertionError("Network access is forbidden in final verification")


def exact_usage_cost(provider, usage):
    """Independent decimal implementation of the frozen per-million prices."""
    n = lambda key: Decimal(usage.get(key, 0) or 0)
    if provider == "anthropic":
        one_hour = Decimal((usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0) or 0)
        five_min = max(Decimal(0), n("cache_creation_input_tokens") - one_hour)
        return (n("input_tokens") * 3 + n("output_tokens") * 15 +
                n("cache_read_input_tokens") * Decimal("0.3") + five_min * Decimal("3.75") + one_hour * 6) / 1000000
    assert provider == "openai", provider
    inputs = Decimal(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    outputs = Decimal(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = Decimal(details.get("cached_tokens", 0) or 0)
    assert 0 <= cached <= inputs
    return ((inputs - cached) * Decimal("2.5") + cached * Decimal("1.25") + outputs * 10) / 1000000


def verify(output):
    required = ["report/summary.json", "accepted_judgments.json", "prepared_manifest.json",
                "frozen/candidate_key_confidential.json", "paid_usage.jsonl"]
    assert all((output / name).is_file() for name in required), "Campaign is not complete; run report first"
    driver.configure(output)
    driver.verify(output)  # Frozen source, case facts, execution plan, prior sources and prepared outputs.
    frozen = read(output / "frozen/campaign_manifest.json")
    assert frozen["generation_outcomes"] == 9 and frozen["planned_judgments"] == 30
    assert frozen["archived_outcomes_reevaluated"] == 6
    assert frozen["spending_guard_usd"] == 10
    plan = read(output / "frozen/execution_plan.json")["cells"]
    assert len(plan) == 9
    expected_runs = {driver.campaign.run_directory(output, row).resolve() for row in plan}
    actual_runs = {p.parent.resolve() for p in (output / "generation").rglob("run_summary.json")}
    assert actual_runs == expected_runs
    run_statuses = {}
    for path in sorted(actual_runs):
        status = read(path / "run_summary.json")["status"]
        assert status in {"completed", "failed"}, status
        run_statuses[str(path.relative_to(output))] = status

    keys = read(output / "frozen/candidate_key_confidential.json")["candidates"]
    assert len(keys) == 15 and len({c["candidate_id"] for c in keys}) == 15
    assert Counter(c["cohort"] for c in keys) == {"fresh": 9, "archived": 6}
    assert {Path(c["run_directory"]).resolve() for c in keys if c["cohort"] == "fresh"} == expected_runs
    assert {p.stem for p in (output / "packets").glob("*.json")} == {c["candidate_id"] for c in keys}
    assert Counter((c["cohort"], c["architecture"]) for c in keys) == {
        ("fresh", "Practical one-shot"): 3, ("fresh", "Guided one-shot"): 3,
        ("fresh", "FlowPilot"): 3, ("archived", "Practical one-shot"): 3, ("archived", "FlowPilot"): 3,
    }
    integrity = read(output / "packet_integrity.json")
    assert len(integrity) == 15
    assert {item["candidate_id"] for item in integrity} == {c["candidate_id"] for c in keys}
    for item in integrity:
        packet_path = output / "packets" / (item["candidate_id"] + ".json")
        assert sha(packet_path) == item["sha256"] and packet_path.stat().st_size == item["bytes"]

    rubric = read(output / "frozen/outcome_rubric.json")
    summary = read(output / "report/summary.json")
    accepted = read(output / "accepted_judgments.json")
    judges = tuple(driver.campaign.JUDGES)
    assert set(judges) == {"openai", "qwen"}
    expected_pairs = {(c["candidate_id"], judge) for c in keys for judge in judges}
    accepted_by_pair = {(r["candidate_id"], r["judge"]): r for r in accepted}
    assert len(accepted) == len(accepted_by_pair) == 30 and set(accepted_by_pair) == expected_pairs
    saved_rows = {(r["candidate_id"], r["judge"]): r for r in summary["results"]}
    assert len(summary["results"]) == len(saved_rows) == 30 and set(saved_rows) == expected_pairs

    rows, delivery, attempt_counts, evidence_counts = [], [], {}, Counter()
    for candidate in keys:
        cid = candidate["candidate_id"]
        packet_path = output / "packets" / f"{cid}.json"
        packet = read(packet_path)
        assert packet["candidate_id"] == cid and not packet_gate(packet)
        run = Path(candidate["run_directory"])
        source = read(run / "result.json") if (run / "result.json").exists() else None
        if source is not None and candidate["generation_status"] == "completed":
            is_pipeline = source.get("variant") == "full" or isinstance(source.get("final_design"), dict)
            expected_outcome = _scrub(_pipeline_outcome(source) if is_pipeline else _one_shot_outcome(source))
            assert packet["delivered_final_outcome"] == expected_outcome, f"Packet outcome differs from source {cid}"
            if not is_pipeline:
                assert packet["delivered_final_outcome"]["chemistry_and_operations"]["chemistry_notes"] == _scrub(source["raw_delivered_text"])
        elif candidate["generation_status"] == "failed":
            assert packet["delivered_final_outcome"]["delivery_status"] == "generation_failed"
            assert packet["delivered_final_outcome"]["reported_disposition"] == "GENERATION_FAILED"
            assert not packet["delivered_final_outcome"]["parameters"]
        if candidate["cohort"] == "archived":
            assert sha(run / "result.json") == candidate["result_sha256"]
            assert sha(Path(candidate["original_run_directory"]) / "result.json") == candidate["result_sha256"]
        elif candidate["generation_status"] == "completed":
            public = read(run / "input_public.json")
            if candidate["architecture"] == "FlowPilot":
                assert source["batch_record"]["raw_text"] == public["protocol"]
                assert source["intake_package"]["chemistry_identity_confirmation"] == {}
                assert source["intake_package"]["inventory_constraints"] == public["inventory"]
            else:
                label = driver.campaign.safe_name(candidate["case"])
                prompt_kind = "guided" if candidate["architecture"] == "Guided one-shot" else "practical"
                expected_prompt = (output / "frozen/practical_inputs" / label / f"{prompt_kind}_request.txt").read_text()
                prompt = read(run / "prompt.json")
                assert prompt["system"] == driver.base.SYSTEM and prompt["user"] == expected_prompt
                assert prompt["prompt_sha256"] == hashlib.sha256(expected_prompt.encode()).hexdigest()
        if candidate["cohort"] == "fresh":
            final = (source or {}).get("final_design") or {}
            delivery.append({"candidate_id": cid, "case": candidate["case"], "architecture": candidate["architecture"],
                             "generation_status": candidate["generation_status"],
                             "delivery_status": final.get("status", (source or {}).get("output_validity", "no_delivery")),
                             "delivery_issues": (final.get("consistency") or {}).get("issues", [])})
        for judge in judges:
            pair = (cid, judge)
            statuses = sorted((output / "judgments" / judge / cid).rglob("status.json"))
            valid_paths = [p for p in statuses if read(p).get("status") == "valid"]
            assert len(valid_paths) == 1, f"Expected exactly one valid accepted assessment {pair}: {valid_paths}"
            status_path = valid_paths[0]
            parsed_path = status_path.with_name("parsed_response.json")
            accepted_row = accepted_by_pair[pair]
            assert str(status_path.relative_to(output)) == accepted_row["status_path"]
            assert sha(parsed_path) == accepted_row["response_sha256"]
            response = read(parsed_path)
            errors = validate_response(response, rubric, packet)
            assert not errors, (pair, errors)
            assert all(resolves_json_path(packet, path) for criterion in response["criterion_scores"]
                       for path in criterion["evidence_paths"]), f"Unresolved cited evidence {pair}"
            assert response == driver.base.resolved_response(output, judge, cid, rubric, packet)
            applicable = [r for r in response["criterion_scores"] if r["applicability"] == "APPLICABLE"]
            score = sum(r["score"] for r in applicable) / (4 * len(applicable))
            flags = sum(bool(r["critical_error"]) for r in response["criterion_scores"])
            row = {"candidate_id": cid, "cohort": candidate["cohort"], "case": candidate["case"],
                   "architecture": candidate["architecture"], "judge": judge, "score": score, "critical_flags": flags}
            expected = saved_rows[pair]
            for key, value in row.items():
                equal_float(value, expected[key]) if key == "score" else None
                if key != "score": assert value == expected[key], (pair, key)
            rows.append(row)
            attempt_counts[f"{judge}/{cid}"] = len(statuses)
            evidence_counts["criterion_rows"] += len(response["criterion_scores"])
            evidence_counts["applicable_criterion_rows"] += len(applicable)
            evidence_counts["resolved_citation_paths"] += sum(len(r["evidence_paths"]) for r in response["criterion_scores"])

    recomputed_groups = {}
    for judge in judges:
        recomputed_groups[judge] = {}
        for cohort in ("fresh", "archived"):
            recomputed_groups[judge][cohort] = {}
            for arm in driver.ARMS:
                group = [r for r in rows if (r["judge"], r["cohort"], r["architecture"]) == (judge, cohort, arm)]
                if not group: continue
                assert len(group) == 3 and len({r["case"] for r in group}) == 3
                computed = {"mean": statistics.fmean(r["score"] for r in group),
                            "critical_flags": sum(r["critical_flags"] for r in group), "n": len(group)}
                saved = summary["groups"][judge][cohort][arm]
                equal_float(computed["mean"], saved["mean"])
                assert computed["critical_flags"] == saved["critical_flags"] and computed["n"] == saved["n"]
                recomputed_groups[judge][cohort][arm] = computed
    assert summary["generation_outcomes"] == 9 and summary["archived_outcomes_reevaluated"] == 6
    assert summary["accepted_judgments"] == 30 and summary["primary_judge"] == "openai"

    ledger = driver.base.ledger_rows(output)
    by_request = defaultdict(list)
    for row in ledger: by_request[row["request_id"]].append(row)
    total = Decimal(0)
    paid = 0
    failed = 0
    for request_id, events in by_request.items():
        reserves = [r for r in events if r["event"] == "reserved"]
        terminal = [r for r in events if r["event"] in {"completed", "failed"}]
        assert len(reserves) == len(terminal) == 1, f"Unsettled or duplicated request {request_id}"
        event = terminal[0]
        assert event["provider"] == reserves[0]["provider"]
        if event["event"] == "failed": failed += 1; continue
        assert event["requested_model"] == reserves[0]["model"]
        assert event["requested_model"] == {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o"}[event["provider"]]
        exact = exact_usage_cost(event["provider"], event["usage"])
        equal_float(float(exact), event["usd"])
        equal_float(float(exact), driver.base.usage_cost(event["provider"], event["usage"]))
        total += exact
        paid += 1
    assert total <= Decimal("10"), total
    cost = driver.base.cost(output)
    equal_float(float(total), cost["estimated_paid_usd"])
    assert cost["pending_reserved_usd"] == 0 and cost["recorded_paid_calls"] == paid
    assert summary["usage"] == cost, "Reported usage is stale"

    driver.verify(output)
    deltas = {}
    for judge in judges:
        fresh = recomputed_groups[judge]["fresh"]
        deltas[judge] = {
            "fresh_flowpilot_minus_practical_one_shot": fresh["FlowPilot"]["mean"] - fresh["Practical one-shot"]["mean"],
            "fresh_guided_minus_practical_one_shot": fresh["Guided one-shot"]["mean"] - fresh["Practical one-shot"]["mean"],
            "fresh_flowpilot_minus_guided_one_shot": fresh["FlowPilot"]["mean"] - fresh["Guided one-shot"]["mean"],
        }
    verification = {
        "verified_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "checker_sha256": sha(Path(__file__)), "network_calls": 0,
        "frozen_source_case_inputs_old_archive_and_prepared_manifest_verified": True,
        "fresh_generation_attempts": 9, "archived_outcomes": 6, "unique_packets": 15,
        "unique_valid_accepted_judgments": 30, "run_statuses": run_statuses,
        "attempt_counts_including_invalid_attempts": attempt_counts,
        "all_packet_outcomes_match_unchanged_recorded_sources": True,
        "all_one_shot_prose_retained_without_numerical_repair": True,
        "score_and_equal_case_group_means_reproduced": True,
        "evidence_structure": dict(evidence_counts),
        "evidence_verification_scope": "Schema, applicability, citation-path resolution and exact packet/source provenance verified. This does not establish that every judge observation, chemical interpretation or factual conclusion is correct; scientific assessment still requires content audit.",
        "fresh_delivery": delivery, "recomputed_groups": recomputed_groups, "computed_differences": deltas,
        "cost": {**cost, "independent_decimal_usd": str(total), "failed_requests_unreported_cost_unknown": failed},
        "limits": ["One output per case-arm; no generation-repeat uncertainty estimate.",
                   "Development cases, without an independent chemistry holdout.",
                   "Guidance changes prompt content and token count together; token count alone is not isolated.",
                   "Scores and flags remain LLM assessments, not laboratory outcomes.",
                   "Returned-usage API estimate excludes local hardware and unreported server work after failed requests."],
    }
    write(output / "verification.json", verification)
    # Exclude self and ephemeral lock. verification.json is included, with no
    # reverse reference to the manifest hash, so there is no hash cycle.
    manifest = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                if p.is_file() and p.name not in {"artifact_manifest.json", "paid_usage.lock"}}
    write(output / "artifact_manifest.json", manifest)
    assert all(sha(output / name) == digest for name, digest in manifest.items())
    print(json.dumps({"verified": str(output), "judgments": 30, "paid_usd": str(total),
                      "fresh_flowpilot_delivery": [r for r in delivery if r["architecture"] == "FlowPilot"],
                      "computed_differences": deltas, "hashed_artifacts": len(manifest)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=driver.DEFAULT)
    args = parser.parse_args()
    with patch("socket.socket.connect", no_network), patch("socket.create_connection", no_network):
        verify(args.output.resolve())


if __name__ == "__main__":
    main()
