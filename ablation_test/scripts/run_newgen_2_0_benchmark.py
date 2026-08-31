#!/usr/bin/env python3
"""Build and execute the blinded NewGen 2.0 three-judge benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.holistic_audit import build_case_context
from ablation_test.src.llm_judge import parse_json_object
from ablation_test.src.newgen_2_outcome import (
    build_outcome_packet, candidate_id, judge_prompt, packet_gate, read_json,
    normalize_response, response_schema, stable_seed, validate_response, write_json,
)
from ablation_test.src.providers import activate_bundle, endpoint_health
from flora_translate.engine.llm_agents import (
    call_model_text, clear_llm_observer, clear_llm_runtime_overrides,
    set_llm_observer, set_llm_runtime_overrides,
)

RUBRIC = ROOT / "ablation_test/benchmarks/newgen_2_0_outcome_rubric.json"
SOURCE = ROOT / "deliverables/flowpilot_newgen_qwen_openai_20260813/source_data/run_metrics.csv"
OLD_AUDIT = ROOT / "ablation_results/newgen_benchmark/newgen_holistic_error_audit_v2_20260814"
DEFAULT_OUTPUT = ROOT / "ablation_results/newgen_benchmark/newgen_2_0_20260818"

JUDGES = {
    "qwen": {"provider": "ollama", "model": "/models/Qwen3.6-27B", "base_url": "http://10.13.24.169:8000/v1", "upstream_mode": "always", "family": "qwen"},
    "openai": {"provider": "openai", "model": "gpt-4o", "upstream_mode": "never", "family": "openai"},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6", "upstream_mode": "never", "family": "anthropic"},
}
ORACLES = {
    "CuAAC": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_cuaac/hidden_oracle.json",
    "Hydrogenolysis": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_hydrogenolysis/hidden_oracle.json",
    "Two-stage amidation": ROOT / "ablation_test/benchmarks/newgen_benchmark_v1_pilot_multistep/hidden_oracle.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def load_inventory(run_dir: Path, public: dict[str, Any]) -> dict[str, Any]:
    if isinstance(public.get("inventory"), dict):
        return public["inventory"]
    path = run_dir / "input_inventory.json"
    return read_json(path) if path.is_file() else {}


def schema_for_provider(schema: dict[str, Any], judge: str) -> dict[str, Any]:
    if judge != "claude":
        return schema
    # Anthropic structured outputs currently accept minItems only as 0 or 1.
    # The local response validator still enforces the exact 14-row contract, so
    # removing this provider keyword weakens neither acceptance nor scoring.
    unsupported = {"minItems", "maxItems", "maxLength", "minimum", "maximum"}
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in unsupported}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(schema)


def _old_audits_by_run() -> dict[str, dict[str, Any]]:
    key_path = OLD_AUDIT / "frozen/candidate_key_confidential.json"
    if not key_path.is_file():
        raise FileNotFoundError("Run the frozen deterministic audit before NewGen 2.0")
    mapping = {}
    for row in read_json(key_path)["candidates"]:
        audit = OLD_AUDIT / "deterministic_evidence/audits" / f"{row['candidate_id']}.json"
        mapping[row["run_directory"]] = read_json(audit)
    return mapping


def build_packets(output: Path, selected_judges: list[str] | None = None) -> None:
    selected_judges = selected_judges or list(JUDGES)
    frozen, packets = output / "frozen", output / "packets"
    frozen.mkdir(parents=True, exist_ok=True)
    if packets.exists():
        shutil.rmtree(packets)
    packets.mkdir(parents=True)
    shutil.copy2(RUBRIC, frozen / "outcome_rubric.json")
    shutil.copy2(SOURCE, frozen / "source_manifest.csv")
    source_dir = frozen / "source_code"
    source_dir.mkdir(exist_ok=True)
    for source in (
        ROOT / "ablation_test/src/newgen_2_outcome.py",
        ROOT / "ablation_test/scripts/run_newgen_2_0_benchmark.py",
        ROOT / "ablation_test/scripts/build_newgen_2_0_report.py",
        ROOT / "ablation_test/benchmarks/NEWGEN_2_0_PROTOCOL.md",
    ):
        if source.is_file():
            shutil.copy2(source, source_dir / source.name)
    old_audits = _old_audits_by_run()
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    contexts: dict[str, dict[str, Any]] = {}
    key_rows, metrics = [], []
    for row in rows:
        run_dir = Path(row["run_directory"])
        public = read_json(run_dir / "input_public.json")
        if row["case"] not in contexts:
            contexts[row["case"]] = build_case_context(
                case_label=row["case"], public_input=public,
                inventory=load_inventory(run_dir, public), oracle=read_json(ORACLES[row["case"]]),
            )
        cid = candidate_id(str(run_dir))
        has_gas = row["case"] == "Hydrogenolysis"
        is_multistage = row["case"] == "Two-stage amidation"
        packet = build_outcome_packet(
            candidate=cid, case_context=contexts[row["case"]],
            result=read_json(run_dir / "result.json"),
            deterministic_verification=old_audits[str(run_dir)],
            has_gas=has_gas, is_multistage=is_multistage,
        )
        errors = packet_gate(packet)
        if errors:
            raise RuntimeError(f"Packet gate failed for {cid}: {errors}")
        path = packets / f"{cid}.json"
        write_json(path, packet)
        key_rows.append({
            "candidate_id": cid, "case": row["case"], "generator_model": row["model"],
            "generator_family": "qwen" if "qwen" in row["model"].lower() else "openai",
            "architecture": row["architecture"], "run_directory": str(run_dir),
            "result_sha256": sha256(run_dir / "result.json"),
        })
        metrics.append({"candidate_id": cid, "bytes": path.stat().st_size, "gate_errors": errors})
    write_json(frozen / "candidate_key_confidential.json", {"candidates": key_rows})
    write_json(frozen / "packet_metrics.json", metrics)
    write_json(frozen / "campaign_manifest.json", {
        "schema_version": "flowpilot_newgen_2_0_campaign_v1.0",
        "benchmark_name": "NewGen 2.0", "created_at": datetime.now().isoformat(),
        "candidate_count": len(key_rows), "criteria_per_candidate": 14,
        "judge_count": len(selected_judges), "planned_calls": len(key_rows) * len(selected_judges),
        "selected_judges": selected_judges,
        "architecture_label_withheld": True, "architecture_blinding_claimed": False,
        "generator_identity_withheld": True,
        "equal_criterion_weights": True, "rubric_frozen_before_judging": True,
        "rubric_sha256": sha256(frozen / "outcome_rubric.json"),
        "source_manifest_sha256": sha256(frozen / "source_manifest.csv"),
        "judges": {
            name: {k: v for k, v in JUDGES[name].items() if k != "base_url"}
            for name in selected_judges
        },
    })


def health(output: Path, judges: list[str]) -> None:
    status = {judge: endpoint_health(JUDGES[judge]) for judge in judges}
    write_json(output / "endpoint_health.json", status)
    failed = [name for name, row in status.items() if not row.get("reachable") or not row.get("model_advertised")]
    if failed:
        raise RuntimeError(f"Unavailable judges: {failed}")


def _valid_existing(call_dir: Path, rubric: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any] | None:
    statuses = [call_dir / "status.json", *(call_dir / "attempts").glob("attempt_*/status.json")]
    for status_path in sorted((p for p in statuses if p.is_file()), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        parsed_path = status_path.with_name("parsed_response.json")
        status = read_json(status_path)
        if status.get("status") == "valid" and parsed_path.is_file() and not validate_response(read_json(parsed_path), rubric, packet):
            return status
    return None


def _latest_status(call_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    statuses = [call_dir / "status.json", *(call_dir / "attempts").glob("attempt_*/status.json")]
    paths = [p for p in statuses if p.is_file()]
    if not paths:
        return None, None
    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    return read_json(path), path


def call_judge(output: Path, judge: str, candidate: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    cid = candidate["candidate_id"]
    packet = read_json(output / "packets" / f"{cid}.json")
    call_dir = output / "judgments" / judge / cid
    existing = _valid_existing(call_dir, rubric, packet)
    if existing:
        return existing
    previous, previous_path = _latest_status(call_dir)
    if not (call_dir / "status.json").is_file():
        write_dir = call_dir
    else:
        count = len(list((call_dir / "attempts").glob("attempt_*"))) + 2
        write_dir = call_dir / "attempts" / f"attempt_{count}"
    system, user = judge_prompt(rubric, packet)
    feedback = []
    if previous and previous.get("status") != "valid":
        feedback = previous.get("validation_errors") or [previous.get("error", "Previous response was invalid")]
    elif previous_path and previous_path.with_name("parsed_response.json").is_file():
        feedback = validate_response(read_json(previous_path.with_name("parsed_response.json")), rubric, packet)
    if feedback:
        user += "\n\nVALIDATOR FEEDBACK FROM PRIOR IMMUTABLE ATTEMPT:\n- " + "\n- ".join(map(str, feedback)) + "\nReturn a fresh complete corrected object."
    if previous_path:
        prior_raw_path = previous_path.with_name("raw_response.json")
        if prior_raw_path.is_file():
            prior_raw = read_json(prior_raw_path)
            if prior_raw.get("finish_reason") == "length":
                user += (
                    "\n\nThe prior response was truncated at the output-token ceiling. "
                    "Return the complete 14-row object concisely: at most one evidence path, "
                    "one observed value, and 25 words in each explanatory string per row. "
                    "Do not repeat the protocol, rubric, or reasoning."
                )
    schema = schema_for_provider(response_schema(), judge)
    attempt = "base" if write_dir == call_dir else write_dir.name
    seed = stable_seed("newgen-2.0", judge, cid, attempt)
    write_json(write_dir / "prompt.json", {"system": system, "user": user, "json_schema": schema, "stable_seed": seed})
    events: list[dict[str, Any]] = []
    set_llm_observer(events.append)
    set_llm_runtime_overrides(temperature=0.0, seed=seed, json_mode=True, json_schema=schema, capture_content=False)
    started = time.perf_counter()
    bundle = JUDGES[judge]
    try:
        with activate_bundle(bundle):
            # Local Qwen can use more tokens than the commercial judges to
            # serialize the same 14-row contract. A 7,500-token ceiling caused
            # otherwise valid judgments to be cut mid-JSON for larger packets.
            max_tokens = 12000 if bundle["provider"] == "ollama" else 7500
            response = call_model_text(model=bundle["model"], provider=bundle["provider"], api_name="newgen_2_0_judge", max_tokens=max_tokens, system=system, user_content=user)
        write_json(write_dir / "raw_response.json", {"provider": response.provider, "model": response.model, "usage": response.usage, "stop_reason": response.stop_reason, "finish_reason": response.finish_reason, "text": response.text})
        write_json(write_dir / "telemetry.json", events)
        try:
            parsed = normalize_response(parse_json_object(response.text))
            errors = validate_response(parsed, rubric, packet)
            write_json(write_dir / "parsed_response.json", parsed)
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
        status = {"status": "valid" if not errors else "invalid", "validation_errors": errors, "duration_seconds": round(time.perf_counter() - started, 3), "provider": bundle["provider"], "model": bundle["model"], "stable_seed": seed}
    except Exception as exc:
        status = {"status": "request_failed", "error": f"{type(exc).__name__}: {exc}", "duration_seconds": round(time.perf_counter() - started, 3), "provider": bundle["provider"], "model": bundle["model"], "stable_seed": seed}
    finally:
        clear_llm_observer(); clear_llm_runtime_overrides()
    write_json(write_dir / "status.json", status)
    return status


def run(output: Path, judges: list[str], offset: int, limit: int | None) -> None:
    rubric = read_json(output / "frozen/outcome_rubric.json")
    candidates = read_json(output / "frozen/candidate_key_confidential.json")["candidates"][offset:]
    if limit is not None:
        candidates = candidates[:limit]
    total, current, counts = len(candidates) * len(judges), 0, {}
    for judge in judges:
        for candidate in candidates:
            current += 1
            status = call_judge(output, judge, candidate, rubric)
            for _ in range(2):
                if status["status"] == "valid":
                    break
                status = call_judge(output, judge, candidate, rubric)
            counts[status["status"]] = counts.get(status["status"], 0) + 1
            print(f"NewGen 2.0 {current}/{total} {judge}/{candidate['candidate_id']}: {status['status']}", flush=True)
    write_json(output / f"execution_summary_{'_'.join(judges)}_{offset}_{limit or 'all'}.json", counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("packets", "health", "judge", "all"), default="all")
    parser.add_argument("--judges", nargs="+", choices=tuple(JUDGES), default=list(JUDGES))
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"packets", "all"} or not (output / "frozen/campaign_manifest.json").is_file():
        build_packets(output, args.judges)
    if args.phase in {"health", "judge", "all"} and not args.skip_health:
        health(output, args.judges)
    if args.phase in {"judge", "all"}:
        run(output, args.judges, args.candidate_offset, args.candidate_limit)
    print(output)


if __name__ == "__main__":
    main()
