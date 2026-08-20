"""Run the NewGen v2 holistic criterion-level error audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.holistic_audit import (
    audit_prompt,
    build_case_context,
    build_domain_record,
    read_json,
    response_schema,
    stable_seed,
    validate_response,
    write_json,
)
from ablation_test.src.llm_judge import parse_json_object
from ablation_test.src.providers import activate_bundle, endpoint_health
from flora_translate.engine.llm_agents import (
    call_model_text,
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)


RUBRIC_PATH = ROOT / "ablation_test" / "benchmarks" / "newgen_holistic_error_audit_v2.json"
SOURCE_MANIFEST = ROOT / "deliverables" / "flowpilot_newgen_qwen_openai_20260813" / "source_data" / "run_metrics.csv"
DEFAULT_OUTPUT = ROOT / "ablation_results" / "newgen_benchmark" / "newgen_holistic_error_audit_v2_20260814"

JUDGES = {
    "qwen": {
        "provider": "ollama",
        "model": "/models/Qwen3.6-27B",
        "base_url": "http://10.13.24.169:8000/v1",
        "upstream_mode": "always",
        "family": "qwen",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-5.4-2026-03-05",
        "upstream_mode": "never",
        "family": "openai",
    },
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "upstream_mode": "never",
        "family": "anthropic",
    },
}

CASE_ORACLES = {
    "CuAAC": ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_cuaac" / "hidden_oracle.json",
    "Hydrogenolysis": ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_hydrogenolysis" / "hidden_oracle.json",
    "Two-stage amidation": ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_multistep" / "hidden_oracle.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def _candidate_id(run_directory: str) -> str:
    digest = hashlib.sha256(f"holistic-v2:{run_directory}".encode()).hexdigest().upper()
    return f"H-{digest[:7]}"


def _schema_for_judge(schema: dict[str, Any], judge: str) -> dict[str, Any]:
    """Keep one logical contract while honoring provider schema subsets."""
    if judge != "claude":
        return schema
    unsupported = {"maxItems", "maxLength"}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in unsupported}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(schema)


def _load_inventory(run_dir: Path, public: dict[str, Any]) -> dict[str, Any]:
    if isinstance(public.get("inventory"), dict):
        return public["inventory"]
    path = run_dir / "input_inventory.json"
    return read_json(path) if path.is_file() else {}


def build_packets(output: Path) -> None:
    frozen = output / "frozen"
    packets = output / "packets"
    frozen.mkdir(parents=True, exist_ok=True)
    if packets.exists():
        shutil.rmtree(packets)
    packets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUBRIC_PATH, frozen / "rubric.json")
    shutil.copy2(SOURCE_MANIFEST, frozen / "source_manifest.csv")
    rubric = read_json(RUBRIC_PATH)
    rows = list(csv.DictReader(SOURCE_MANIFEST.open(encoding="utf-8")))
    case_contexts: dict[str, dict[str, Any]] = {}
    key_rows: list[dict[str, Any]] = []
    packet_metrics: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_directory"])
        public = read_json(run_dir / "input_public.json")
        inventory = _load_inventory(run_dir, public)
        if row["case"] not in case_contexts:
            case_contexts[row["case"]] = build_case_context(
                case_label=row["case"],
                public_input=public,
                inventory=inventory,
                oracle=read_json(CASE_ORACLES[row["case"]]),
            )
            write_json(packets / "cases" / f"{_safe(row['case'])}.json", case_contexts[row["case"]])
        result_path = run_dir / "result.json"
        result = read_json(result_path)
        candidate_id = _candidate_id(str(run_dir))
        for domain in rubric["domains"]:
            packet = build_domain_record(
                result,
                candidate_id=candidate_id,
                case_label=row["case"],
                domain=domain,
            )
            packet_path = packets / "candidates" / candidate_id / f"{domain}.json"
            write_json(packet_path, packet)
            packet_metrics.append({
                "candidate_id": candidate_id,
                "domain": domain,
                "bytes": packet_path.stat().st_size,
                "coverage_complete": packet["record_index"]["full_result_coverage"]["coverage_complete"],
                "uncovered_keys": packet["record_index"]["full_result_coverage"]["uncovered_meaningful_keys"],
                "truncation_markers": packet_path.read_text(encoding="utf-8").count("[truncated]")
                    + packet_path.read_text(encoding="utf-8").count("[depth-limited]"),
                "identity_label_hits": len(re.findall(
                    r"flowpilot|flow\s*agent|general_one_shot|\bqwen|\bgpt|\bopenai\b|\bclaude|\banthropic\b",
                    packet_path.read_text(encoding="utf-8"),
                    flags=re.IGNORECASE,
                )),
            })
        key_rows.append({
            "candidate_id": candidate_id,
            "case": row["case"],
            "generator_model": row["model"],
            "generator_family": "qwen" if "qwen" in row["model"].lower() else "openai",
            "architecture": row["architecture"],
            "run_directory": str(run_dir),
            "result_sha256": _sha256(result_path),
        })
    incomplete = [
        row for row in packet_metrics
        if not row["coverage_complete"] or row["truncation_markers"] or row["identity_label_hits"]
    ]
    if incomplete:
        raise RuntimeError(f"Packet coverage/truncation gate failed: {incomplete}")
    write_json(frozen / "candidate_key_confidential.json", {"candidates": key_rows})
    write_json(frozen / "packet_metrics.json", packet_metrics)
    write_json(frozen / "campaign_manifest.json", {
        "schema_version": "newgen_holistic_error_audit_campaign_v2.0",
        "created_at": datetime.now().isoformat(),
        "rubric_sha256": _sha256(frozen / "rubric.json"),
        "source_manifest_sha256": _sha256(frozen / "source_manifest.csv"),
        "candidate_count": len(key_rows),
        "domain_count": len(rubric["domains"]),
        "planned_judge_calls": len(key_rows) * len(rubric["domains"]) * len(JUDGES),
        "judge_repeats": 1,
        "composite_score": False,
        "pairwise_ranking": False,
        "architecture_blinding_claimed": False,
        "generator_identity_withheld": True,
        "judges": {
            key: {name: value for name, value in bundle.items() if name != "base_url"}
            for key, bundle in JUDGES.items()
        },
    })


def check_health(output: Path, judges: list[str]) -> None:
    health = {judge: endpoint_health(JUDGES[judge]) for judge in judges}
    write_json(output / "endpoint_health.json", health)
    failed = [judge for judge, value in health.items() if not value.get("reachable") or not value.get("model_advertised")]
    if failed:
        raise RuntimeError(f"Unavailable judges: {failed}")


def _resolved_status(
    call_dir: Path, rubric: dict[str, Any], domain: str, candidate_id: str
) -> dict[str, Any] | None:
    paths = [call_dir / "status.json", *(call_dir / "attempts").glob("attempt_*/status.json")]
    for path in sorted((item for item in paths if item.is_file()), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        status = read_json(path)
        parsed_path = path.with_name("parsed_response.json")
        if status.get("status") == "valid" and parsed_path.is_file():
            if not validate_response(read_json(parsed_path), rubric, domain, candidate_id):
                return status
    return None


def _latest_unresolved_status(call_dir: Path) -> dict[str, Any] | None:
    paths = [call_dir / "status.json", *(call_dir / "attempts").glob("attempt_*/status.json")]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return read_json(max(existing, key=lambda path: path.stat().st_mtime_ns))


def _resolved_parsed_response(output: Path, judge: str, domain: str, candidate_id: str) -> dict[str, Any] | None:
    root = output / "judgments" / judge / domain / candidate_id
    pairs = [(root / "status.json", root / "parsed_response.json")]
    pairs += [
        (path, path.with_name("parsed_response.json"))
        for path in (root / "attempts").glob("attempt_*/status.json")
    ]
    valid = [
        (status_path, parsed_path) for status_path, parsed_path in pairs
        if status_path.is_file() and parsed_path.is_file() and read_json(status_path).get("status") == "valid"
    ]
    if not valid:
        return None
    _, parsed = max(valid, key=lambda pair: pair[0].stat().st_mtime_ns)
    return read_json(parsed)


def _qwen_hierarchical_evidence_packet(
    output: Path, candidate_id: str, packet: dict[str, Any]
) -> dict[str, Any]:
    """Fit Qwen's 32k window without truncating unique evidence records.

    Sections duplicated in other domains are represented by Qwen's own validated
    criterion findings from those complete raw-section audits. Evidence-specific
    sections remain verbatim in this final pass.
    """
    unique_sections = {
        "chemistry_plan", "_analogies", "safety_report", "design_disposition",
        "confidence", "recommended_disposition", "reported_disposition",
        "disposition_rationale", "design_space", "council_messages", "deliberation_log",
    }
    domain_findings: dict[str, Any] = {}
    for domain in (
        "chemistry_protocol", "numerical_engineering", "process_inventory",
        "safety_operations", "system_consistency",
    ):
        response = _resolved_parsed_response(output, "qwen", domain, candidate_id)
        if response is None:
            continue
        domain_findings[domain] = [
            {
                "criterion_id": item["criterion_id"], "status": item["status"],
                "severity": item["severity"], "evidence_paths": item["evidence_paths"],
                "observed_values": item["observed_values"],
                "expected_or_correct": item["expected_or_correct"],
                "explanation": item["explanation"],
            }
            for item in response["criterion_findings"]
        ]
    return {
        "schema_version": "newgen_holistic_candidate_hierarchical_evidence_v2.0",
        "candidate_id": packet["candidate_id"], "case_label": packet["case_label"],
        "audit_domain": packet["audit_domain"],
        "record_policy": {
            "scope": "Hierarchical complete-record audit for a 32k-context judge.",
            "unique_sections": "Evidence/provenance-specific sections below are supplied verbatim without truncation.",
            "duplicated_sections": "Other result sections were supplied verbatim in the five named prior domain audits; their validated findings are supplied below.",
            "missing_section_meaning": "A null section means the generator did not provide that record.",
        },
        "record_index": packet["record_index"],
        "record_sections": {
            key: value for key, value in packet["record_sections"].items() if key in unique_sections
        },
        "prior_validated_domain_audits": domain_findings,
    }


def _call(
    *,
    output: Path,
    judge: str,
    domain: str,
    candidate: dict[str, Any],
    case_context: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    call_dir = output / "judgments" / judge / domain / candidate_id
    resolved = _resolved_status(call_dir, rubric, domain, candidate_id)
    if resolved:
        return resolved
    base_status = call_dir / "status.json"
    if base_status.is_file():
        attempts = sorted((call_dir / "attempts").glob("attempt_*"))
        write_dir = call_dir / "attempts" / f"attempt_{len(attempts) + 2}"
    else:
        write_dir = call_dir
    packet = read_json(output / "packets" / "candidates" / candidate_id / f"{domain}.json")
    if judge == "qwen" and domain == "evidence_uncertainty":
        packet = _qwen_hierarchical_evidence_packet(output, candidate_id, packet)
    system, user = audit_prompt(rubric, domain, case_context, packet)
    previous = _latest_unresolved_status(call_dir)
    repair_errors = previous.get("validation_errors") or [] if previous and previous.get("status") == "invalid" else []
    if previous and previous.get("status") == "valid":
        latest_paths = [call_dir / "status.json", *(call_dir / "attempts").glob("attempt_*/status.json")]
        latest_paths = [path for path in latest_paths if path.is_file()]
        if latest_paths:
            latest = max(latest_paths, key=lambda path: path.stat().st_mtime_ns)
            parsed_path = latest.with_name("parsed_response.json")
            if parsed_path.is_file():
                repair_errors = validate_response(read_json(parsed_path), rubric, domain, candidate_id)
    if repair_errors:
        user += (
            "\n\nVALIDATOR FEEDBACK FROM THE PREVIOUS IMMUTABLY STORED ATTEMPT:\n- "
            + "\n- ".join(str(item) for item in repair_errors)
            + "\nReturn a fresh complete object that fixes every listed contract error; do not discuss the feedback."
        )
    schema = _schema_for_judge(response_schema(rubric, domain), judge)
    attempt_id = "base" if write_dir == call_dir else write_dir.name
    seed = stable_seed("holistic-v2", judge, domain, candidate_id, "audit-pass-1", attempt_id)
    write_json(write_dir / "prompt.json", {
        "system": system,
        "user": user,
        "json_schema": schema,
        "stable_seed": seed,
        "candidate_id": candidate_id,
        "domain": domain,
    })
    events: list[dict[str, Any]] = []
    set_llm_observer(events.append)
    set_llm_runtime_overrides(temperature=0.0, seed=seed, json_mode=True, json_schema=schema, capture_content=False)
    started = time.perf_counter()
    bundle = JUDGES[judge]
    try:
        with activate_bundle(bundle):
            response = call_model_text(
                model=bundle["model"],
                provider=bundle["provider"],
                api_name=f"newgen_holistic_{domain}",
                max_tokens=3000 if judge == "qwen" else 7000,
                system=system,
                user_content=user,
            )
        write_json(write_dir / "raw_response.json", {
            "provider": response.provider,
            "model": response.model,
            "usage": response.usage,
            "stop_reason": response.stop_reason,
            "finish_reason": response.finish_reason,
            "text": response.text,
        })
        write_json(write_dir / "telemetry.json", events)
        try:
            parsed = parse_json_object(response.text)
            errors = validate_response(parsed, rubric, domain, candidate_id)
        except Exception as exc:
            parsed = None
            errors = [f"{type(exc).__name__}: {exc}"]
        if parsed is not None:
            write_json(write_dir / "parsed_response.json", parsed)
        status = {
            "status": "valid" if not errors else "invalid",
            "validation_errors": errors,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "provider": bundle["provider"],
            "model": bundle["model"],
            "stable_seed": seed,
        }
    except Exception as exc:
        status = {
            "status": "request_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "provider": bundle["provider"],
            "model": bundle["model"],
            "stable_seed": seed,
        }
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()
    write_json(write_dir / "status.json", status)
    return status


def run_audit(
    output: Path,
    judges: list[str],
    domains: list[str] | None,
    candidate_offset: int,
    candidate_limit: int | None,
) -> None:
    rubric = read_json(output / "frozen" / "rubric.json")
    key = read_json(output / "frozen" / "candidate_key_confidential.json")
    candidates = key["candidates"][candidate_offset:]
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]
    selected_domains = domains or list(rubric["domains"])
    case_contexts = {
        path.stem: read_json(path)
        for path in (output / "packets" / "cases").glob("*.json")
    }
    total = len(judges) * len(selected_domains) * len(candidates)
    current = 0
    counts = {"valid": 0, "invalid": 0, "request_failed": 0}
    for judge in judges:
        for domain in selected_domains:
            for candidate in candidates:
                current += 1
                case_key = _safe(candidate["case"])
                status = _call(
                    output=output, judge=judge, domain=domain, candidate=candidate,
                    case_context=case_contexts[case_key], rubric=rubric,
                )
                # Preserve every malformed/request-failed response, then retry in
                # an immutable child attempt. Only a validated response resolves.
                for _ in range(2):
                    if status["status"] == "valid":
                        break
                    status = _call(
                        output=output, judge=judge, domain=domain, candidate=candidate,
                        case_context=case_contexts[case_key], rubric=rubric,
                    )
                counts[status["status"]] = counts.get(status["status"], 0) + 1
                print(f"audit {current}/{total} {judge}/{domain}/{candidate['candidate_id']}: {status['status']}", flush=True)
    suffix = "_".join(judges) + f"_offset{candidate_offset}_limit{candidate_limit if candidate_limit is not None else 'all'}"
    write_json(output / f"execution_summary_{suffix}.json", counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("packets", "health", "audit", "all"), default="all")
    parser.add_argument("--judges", nargs="+", choices=tuple(JUDGES), default=list(JUDGES))
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"packets", "all"} or not (output / "frozen" / "campaign_manifest.json").is_file():
        build_packets(output)
    rubric = read_json(output / "frozen" / "rubric.json")
    unknown = sorted(set(args.domains or []) - set(rubric["domains"]))
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}")
    if args.phase in {"health", "audit", "all"} and not args.skip_health:
        check_health(output, args.judges)
    if args.phase in {"audit", "all"}:
        run_audit(output, args.judges, args.domains, args.candidate_offset, args.candidate_limit)
    print(output)


if __name__ == "__main__":
    main()
