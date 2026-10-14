"""Architecture-neutral records and contracts for the NewGen 2.0 benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


IDENTITY_RE = re.compile(
    r"flow\s*pilot|flow\s*agent|general[_ -]?one[_ -]?shot|\bqwen\b|\bgpt(?:-[\w.]+)?\b|"
    r"\bopenai\b|\bclaude\b|\banthropic\b",
    re.IGNORECASE,
)
ALWAYS_IDS = {f"UO-{number:02d}" for number in range(1, 15)} - {"UO-08", "UO-09"}
SEVERITIES = {"NONE", "MINOR", "MAJOR", "CRITICAL"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def candidate_id(run_directory: str) -> str:
    digest = hashlib.sha256(f"newgen-2.0:{run_directory}".encode()).hexdigest().upper()
    return f"N2-{digest[:8]}"


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        return IDENTITY_RE.sub("[generator withheld]", value)
    return value


def _normal_stages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _one_shot_outcome(result: dict[str, Any]) -> dict[str, Any]:
    proposal = dict(result.get("proposal") or result.get("raw_proposal") or {})
    return {
        "parameters": {key: value for key, value in proposal.items() if key not in {
            "streams", "stage_parameters", "pre_reactor_steps", "post_reactor_steps",
            "chemistry_notes", "reasoning_per_field", "safety_flags", "inventory_selection",
            "inventory_constraints", "evidence_calibration", "literature_analogies",
        }},
        "streams": proposal.get("streams") or [],
        "stages": _normal_stages(proposal.get("stage_parameters")),
        "topology": {
            "reactor_type": proposal.get("reactor_type"),
            "mixer_type": proposal.get("mixer_type"),
            "mixing_order_reasoning": proposal.get("mixing_order_reasoning"),
            "inventory_selection": proposal.get("inventory_selection"),
        },
        "chemistry_and_operations": {
            "chemistry_notes": proposal.get("chemistry_notes"),
            "pre_reactor_steps": proposal.get("pre_reactor_steps"),
            "post_reactor_steps": proposal.get("post_reactor_steps"),
        },
        "safety_controls": proposal.get("safety_flags") or [],
        "evidence_and_reasoning": {
            "reasoning_per_field": proposal.get("reasoning_per_field"),
            "evidence_calibration": proposal.get("evidence_calibration"),
            "literature_analogies": proposal.get("literature_analogies"),
        },
        "reported_disposition": result.get("reported_disposition"),
        "disposition_rationale": result.get("disposition_rationale"),
    }


def _pipeline_outcome(result: dict[str, Any]) -> dict[str, Any]:
    final = result.get("final_design") or {}
    proposal = result.get("proposal") or {}
    topology = result.get("process_topology") or final.get("process_graph") or {}
    executable = final.get("status") == "executable"
    return {
        "parameters": final.get("parameters") or proposal,
        "streams": final.get("streams") or proposal.get("streams") or [],
        "stages": _normal_stages(final.get("stages") or proposal.get("stage_parameters")),
        "topology": topology,
        "chemistry_and_operations": {
            "chemistry_identity": final.get("chemistry_identity") if executable else result.get("chemistry_plan"),
            "stream_components": final.get("stream_components") if executable else None,
            "operating_procedure": final.get("operating_procedure") if executable else None,
            "pre_reactor_steps": None if executable else proposal.get("pre_reactor_steps"),
            "post_reactor_steps": None if executable else proposal.get("post_reactor_steps"),
            "pid_description": topology.get("pid_description") if isinstance(topology, dict) else None,
        },
        "safety_controls": final.get("safety") if executable else result.get("safety_report") or proposal.get("safety_flags") or [],
        "evidence_and_reasoning": {
            "reasoning_per_field": proposal.get("reasoning_per_field"),
            "literature_analogies": topology.get("literature_support") if isinstance(topology, dict) else None,
            "confidence": result.get("confidence"),
            "validation_experiments": final.get("validation_experiments") if executable else [],
            "canonical_sha256": final.get("canonical_sha256") if executable else None,
        },
        "reported_disposition": result.get("reported_disposition") or result.get("recommended_disposition"),
        "disposition_rationale": result.get("disposition_rationale"),
    }


def build_outcome_packet(
    *, candidate: str, case_context: dict[str, Any], result: dict[str, Any],
    deterministic_verification: dict[str, Any], has_gas: bool, is_multistage: bool,
) -> dict[str, Any]:
    final = result.get("final_design")
    # A blocked or screen-only production result is still a pipeline outcome.
    # Routing it through the one-shot normalizer drops the final contract and
    # can make the delivered result look more complete than it was.
    pipeline_outcome = result.get("variant") == "full" or isinstance(final, dict)
    outcome = _pipeline_outcome(result) if pipeline_outcome else _one_shot_outcome(result)
    packet = {
        "schema_version": "flowpilot_newgen_2_0_blinded_outcome_v1.0",
        "candidate_id": candidate,
        "case_context": {
            "case_label": case_context.get("case_label"),
            "protocol": case_context.get("protocol"),
            "objective": case_context.get("objective"),
            "hard_constraints": case_context.get("hard_constraints"),
            "authoritative_inventory": case_context.get("authoritative_inventory"),
            "held_out_source_fact_check": case_context.get("held_out_source_fact_check"),
            "source_policy": case_context.get("source_policy"),
            "applicability": {"gas_process": has_gas, "multistage_process": is_multistage},
        },
        "delivered_final_outcome": outcome,
        "deterministic_verification_sheet": deterministic_verification,
        "evaluation_boundary": (
            "Evaluate only this delivered outcome. Internal architecture, intermediate reasoning, "
            "council traces, validators, and generator identity are withheld from primary scoring."
        ),
    }
    return _scrub(packet)


def packet_gate(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    text = json.dumps(packet, ensure_ascii=False)
    if IDENTITY_RE.search(text):
        errors.append("generator or architecture identity leaked")
    for key in ("candidate_id", "case_context", "delivered_final_outcome", "deterministic_verification_sheet"):
        if key not in packet:
            errors.append(f"missing packet field: {key}")
    outcome = packet.get("delivered_final_outcome") or {}
    for key in ("parameters", "streams", "stages", "topology", "chemistry_and_operations", "safety_controls"):
        if key not in outcome:
            errors.append(f"missing common outcome field: {key}")
    if "[truncated]" in text or "[depth-limited]" in text:
        errors.append("packet contains truncation marker")
    return errors


def response_schema() -> dict[str, Any]:
    row = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "criterion_id", "applicability", "score", "critical_error", "severity",
            "evidence_paths", "observed_values", "expected_or_correct", "rationale",
            "required_correction", "confidence",
        ],
        "properties": {
            "criterion_id": {"type": "string", "enum": [f"UO-{i:02d}" for i in range(1, 15)]},
            "applicability": {"type": "string", "enum": ["APPLICABLE", "NOT_APPLICABLE"]},
            "score": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
            "critical_error": {"type": "boolean"},
            "severity": {"type": "string", "enum": sorted(SEVERITIES)},
            "evidence_paths": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 5},
            "observed_values": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 5},
            "expected_or_correct": {"type": "string", "maxLength": 600},
            "rationale": {"type": "string", "maxLength": 800},
            "required_correction": {"type": "string", "maxLength": 600},
            "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["candidate_id", "criterion_scores", "overall_executability", "cross_criterion_notes"],
        "properties": {
            "candidate_id": {"type": "string"},
            "criterion_scores": {"type": "array", "items": row, "minItems": 14, "maxItems": 14},
            "overall_executability": {"type": "string", "enum": ["EXECUTABLE", "SCREEN_ONLY", "BLOCKED"]},
            "cross_criterion_notes": {"type": "array", "items": {"type": "string", "maxLength": 400}, "maxItems": 5},
        },
    }


def validate_response(response: dict[str, Any], rubric: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    candidate = packet["candidate_id"]
    if response.get("candidate_id") != candidate:
        errors.append(f"candidate_id must equal {candidate}")
    rows = response.get("criterion_scores")
    if not isinstance(rows, list):
        return errors + ["criterion_scores must be a list"]
    expected = {item["criterion_id"] for item in rubric["criteria"]}
    ids = [row.get("criterion_id") for row in rows if isinstance(row, dict)]
    if len(rows) != 14 or set(ids) != expected or len(ids) != len(set(ids)):
        errors.append("criterion_scores must contain each UO-01 through UO-14 exactly once")
    flags = packet["case_context"]["applicability"]
    allowed_na = set()
    if not flags["gas_process"]:
        allowed_na.add("UO-08")
    if not flags["multistage_process"]:
        allowed_na.add("UO-09")
    required_keys = {
        "criterion_id", "applicability", "score", "critical_error", "severity",
        "evidence_paths", "observed_values", "expected_or_correct", "rationale",
        "required_correction", "confidence",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index} must be an object")
            continue
        cid = row.get("criterion_id")
        missing = required_keys - set(row)
        if missing:
            errors.append(f"{cid}: missing keys {sorted(missing)}")
            continue
        applicable = row["applicability"] == "APPLICABLE"
        if not applicable:
            if cid not in allowed_na:
                errors.append(f"{cid}: NOT_APPLICABLE is not allowed")
            if row["score"] is not None or row["critical_error"] or row["severity"] != "NONE":
                errors.append(f"{cid}: N/A requires null score, no critical error, and NONE severity")
            continue
        if cid in allowed_na:
            errors.append(f"{cid}: case applicability requires NOT_APPLICABLE")
        score = row["score"]
        if type(score) is not int or not 0 <= score <= 4:
            errors.append(f"{cid}: applicable score must be integer 0..4")
            continue
        severity = row["severity"]
        if severity not in SEVERITIES:
            errors.append(f"{cid}: invalid severity")
        if row["critical_error"] and not (score <= 1 and severity == "CRITICAL"):
            errors.append(f"{cid}: critical_error requires score <=1 and CRITICAL severity")
        if score == 4 and (severity != "NONE" or row["critical_error"]):
            errors.append(f"{cid}: score 4 requires NONE severity")
        if score <= 2:
            if not row["evidence_paths"] or not row["observed_values"]:
                errors.append(f"{cid}: score <=2 requires evidence paths and observed values")
            for field in ("expected_or_correct", "rationale", "required_correction"):
                if not str(row[field]).strip():
                    errors.append(f"{cid}: score <=2 requires {field}")
    return errors


def judge_prompt(rubric: dict[str, Any], packet: dict[str, Any]) -> tuple[str, str]:
    system = """You are an independent forensic evaluator of continuous-flow chemistry designs.
Score only the delivered final outcome. Do not infer or reward the hidden architecture, model identity,
number of agents, verbosity, internal validation, or likely generation method. Apply the frozen rubric
literally and equally. The published flow reference is supporting evidence, not the only valid answer.
Differences from it are not defects unless they contradict protocol facts, physics, inventory, safety,
or lack a defensible basis. Treat the deterministic verification sheet as authoritative for calculations
it explicitly tests; if you reject one of its conclusions, identify the exact erroneous input or formula.
Do not provide chain-of-thought. Return only the requested JSON object."""
    user = (
        "Evaluate this blinded candidate against every fixed criterion exactly once.\n\n"
        "SCORING RULES:\n"
        "- 4 correct, complete, executable, supported; no material defect.\n"
        "- 3 correct overall; only a minor defect unlikely to change execution or interpretation.\n"
        "- 2 one material but recoverable defect; revision is required before execution.\n"
        "- 1 major or multiple material deficiencies; substantial redesign is required.\n"
        "- 0 fundamentally wrong, unsafe, chemically invalid, or physically impossible.\n"
        "Missing required information is score 0 or 1, never N/A. N/A is allowed only for UO-08 "
        "when gas_process=false and UO-09 when multistage_process=false. A critical error must match "
        "a criterion-specific critical condition, have score 0 or 1, and severity CRITICAL. Cite exact "
        "JSON paths. Judge all architectures by the same final-output standard.\n\n"
        f"FROZEN RUBRIC:\n{json.dumps(rubric, ensure_ascii=False, sort_keys=True)}\n\n"
        f"BLINDED PACKET:\n{json.dumps(packet, ensure_ascii=False, sort_keys=True)}"
    )
    return system, user
