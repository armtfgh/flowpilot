from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


OUTCOME_PARAMETER_FIELDS = (
    "residence_time_min",
    "residence_time_inlet_min",
    "residence_time_in_channel_min",
    "residence_time_basis",
    "flow_rate_mL_min",
    "temperature_C",
    "concentration_M",
    "BPR_bar",
    "reactor_type",
    "reactor_volume_mL",
    "tubing_material",
    "tubing_ID_mm",
    "wavelength_nm",
)

BLIND_PATTERNS = (
    (re.compile(r"flow\s*pilot", re.I), "the design system"),
    (re.compile(r"flow\s*agent", re.I), "the design system"),
    (re.compile(r"\bqwen[^\s,;:\]\[}\{\"]*", re.I), "a language model"),
    (re.compile(r"\bgpt[-\w.]*", re.I), "a language model"),
    (re.compile(r"\bopenai\b", re.I), "a model provider"),
    (re.compile(r"\bclaude[-\w.]*", re.I), "a language model"),
    (re.compile(r"\banthropic\b", re.I), "a model provider"),
    (re.compile(r"\bone[- ]shot\b", re.I), "single-pass"),
    (re.compile(r"council", re.I), "deliberation_review"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", (text or "").strip(), flags=re.I)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def _blind_text(text: str) -> str:
    value = text
    for pattern, replacement in BLIND_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def blind_value(value: Any, *, max_string: int = 2400, max_list: int = 30, depth: int = 0) -> Any:
    if depth > 7:
        return "[depth-limited]"
    if isinstance(value, str):
        text = _blind_text(value)
        return text if len(text) <= max_string else text[:max_string] + " [truncated]"
    if isinstance(value, list):
        return [blind_value(item, max_string=max_string, max_list=max_list, depth=depth + 1) for item in value[:max_list]]
    if isinstance(value, dict):
        return {
            _blind_text(str(key)): blind_value(item, max_string=max_string, max_list=max_list, depth=depth + 1)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _native_proposal(result: dict[str, Any]) -> dict[str, Any]:
    final = result.get("final_design")
    proposal = dict(result.get("proposal") or {})
    if isinstance(final, dict) and final.get("status") == "executable":
        parameters = dict(final.get("parameters") or {})
        return {
            "parameters": parameters,
            "streams": final.get("streams") or proposal.get("streams") or [],
            "stages": final.get("stages") or proposal.get("stage_parameters") or [],
            "process_graph": final.get("process_graph") or {},
            "pre_reactor_steps": proposal.get("pre_reactor_steps") or [],
            "post_reactor_steps": proposal.get("post_reactor_steps") or [],
            "mixing_order_reasoning": proposal.get("mixing_order_reasoning"),
            "chemistry_notes": proposal.get("chemistry_notes"),
            "safety_flags": proposal.get("safety_flags") or [],
            "inventory_selection": proposal.get("inventory_selection") or {},
            "confidence": proposal.get("confidence") or result.get("confidence"),
            "reported_disposition": result.get("recommended_disposition") or result.get("reported_disposition"),
            "disposition_rationale": result.get("disposition_rationale"),
        }
    native = result.get("raw_proposal") if isinstance(result.get("raw_proposal"), dict) else proposal
    parameters = {field: native.get(field, proposal.get(field)) for field in OUTCOME_PARAMETER_FIELDS}
    return {
        "parameters": parameters,
        "streams": native.get("streams") or proposal.get("streams") or [],
        "stages": native.get("stage_parameters") or proposal.get("stage_parameters") or [],
        "process_graph": native.get("process_graph") or {},
        "pre_reactor_steps": native.get("pre_reactor_steps") or [],
        "post_reactor_steps": native.get("post_reactor_steps") or [],
        "mixing_order_reasoning": native.get("mixing_order_reasoning"),
        "chemistry_notes": native.get("chemistry_notes"),
        "safety_flags": native.get("safety_flags") or [],
        "inventory_selection": native.get("inventory_selection") or {},
        "confidence": native.get("confidence"),
        "reported_disposition": result.get("reported_disposition"),
        "disposition_rationale": result.get("disposition_rationale"),
    }


def _compact_process_graph(graph: Any) -> dict[str, Any]:
    if not isinstance(graph, dict):
        return {}
    topology = graph.get("topology") if isinstance(graph.get("topology"), dict) else graph
    operations = []
    for operation in topology.get("unit_operations") or []:
        if not isinstance(operation, dict):
            continue
        parameters = operation.get("parameters") or {}
        keep = {
            key: parameters.get(key)
            for key in (
                "flow_rate_mL_min",
                "gas_flow_sccm",
                "gas_flow_actual_mL_min",
                "temperature_C",
                "pressure_bar",
                "volume_mL",
                "residence_time_min",
                "inventory_equipment_id",
                "instrument_name",
            )
            if parameters.get(key) is not None
        }
        operations.append({
            "operation_id": operation.get("op_id") or operation.get("operation_id"),
            "type": operation.get("op_type") or operation.get("type"),
            "label": operation.get("label"),
            "parameters": keep,
        })
    connections = []
    for connection in topology.get("connections") or []:
        if isinstance(connection, dict):
            connections.append({
                "from": connection.get("from") or connection.get("source"),
                "to": connection.get("to") or connection.get("target"),
                "stream": connection.get("stream") or connection.get("label"),
            })
    return {"unit_operations": operations, "connections": connections}


def build_outcome_view(result: dict[str, Any]) -> dict[str, Any]:
    view = _native_proposal(result)
    view["process_graph"] = _compact_process_graph(view.get("process_graph"))
    return blind_value(view, max_string=2200, max_list=40)


def _compact_calculations(calculations: Any) -> dict[str, Any]:
    if not isinstance(calculations, dict):
        return {}
    scalar_keys = (
        "residence_time_min",
        "residence_time_inlet_min",
        "residence_time_in_channel_min",
        "residence_time_basis",
        "flow_rate_mL_min",
        "liquid_flow_rate_mL_min",
        "reactor_volume_mL",
        "temperature_C",
        "BPR_bar",
        "gas_flow_sccm",
        "gas_flow_actual_mL_min",
        "gas_pressure_abs_bar",
        "gas_equiv_supplied",
        "o2_equiv_supplied",
        "consistent",
        "consistency_notes",
        "calculation_mode",
    )
    compact = {key: calculations.get(key) for key in scalar_keys if calculations.get(key) is not None}
    steps = []
    for step in calculations.get("steps") or []:
        if not isinstance(step, dict):
            continue
        values = step.get("values") or {}
        selected_values = {
            key: item
            for key, item in values.items()
            if isinstance(item, (int, float, bool, str, type(None)))
        }
        steps.append({
            "name": step.get("name"),
            "status": step.get("status"),
            "summary": step.get("summary"),
            "values": selected_values,
            "equations": (step.get("equations") or [])[:5],
            "warnings": step.get("warnings") or [],
            "assumptions": step.get("assumptions") or [],
        })
    compact["steps"] = steps
    compact["stage_calculations"] = calculations.get("stage_calculations") or []
    return compact


def _compact_allocation(allocation: Any) -> dict[str, Any]:
    if not isinstance(allocation, dict):
        return {}
    assignments = []
    for item in allocation.get("assignments") or []:
        if not isinstance(item, dict):
            continue
        assignments.append({
            "operation_id": item.get("operation_id"),
            "role": item.get("role"),
            "category": item.get("category"),
            "equipment_item_ids": item.get("equipment_item_ids") or [],
            "settings": item.get("settings") or {},
            "capability_checks": item.get("capability_checks") or {},
        })
    return {
        "status": allocation.get("status"),
        "strict_assignment": allocation.get("strict_assignment"),
        "inventory_fingerprint_present": bool(allocation.get("inventory_sha256")),
        "checks": allocation.get("checks") or {},
        "assignments": assignments,
        "unresolved_requirements": allocation.get("unresolved_requirements") or [],
        "warnings": allocation.get("warnings") or [],
    }


def _compact_analogies(result: dict[str, Any]) -> list[dict[str, Any]]:
    analogies = result.get("_analogies") or result.get("analogies") or []
    output = []
    for item in analogies:
        if not isinstance(item, dict):
            continue
        output.append({
            "source_id": item.get("record_id") or item.get("source_pdf") or (item.get("metadata") or {}).get("source_pdf"),
            "similarity": item.get("similarity") or item.get("score"),
            "relevance": item.get("relevance") or item.get("rationale"),
        })
    return output[:5]


def _compact_deliberation(result: dict[str, Any]) -> dict[str, Any]:
    rounds = result.get("council_rounds") or []
    messages = result.get("council_messages") or []
    log = result.get("deliberation_log") or []
    return {
        "round_count": len(rounds) if isinstance(rounds, list) else 0,
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "decision_log": log[:20] if isinstance(log, list) else log,
        "final_validation": result.get("final_validation") or {},
        "design_disposition": result.get("design_disposition") or {},
    }


def build_assurance_view(result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    proposal = result.get("proposal") or {}
    calculations = result.get("design_calculations") or (result.get("frozen_context") or {}).get("calculations")
    audit_path = run_dir / "snapshots" / "stage3_5_final_audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else None
    view = {
        "field_reasoning": proposal.get("reasoning_per_field") or {},
        "calculation_trace": _compact_calculations(calculations),
        "inventory_trace": _compact_allocation(result.get("inventory_allocation")),
        "source_evidence": _compact_analogies(result),
        "deliberation_trace": _compact_deliberation(result),
        "safety_review": result.get("safety_report") or (result.get("design_realization") or {}).get("safety_contract") or {},
        "final_audit": audit or {},
        "confidence": proposal.get("confidence") or result.get("confidence"),
        "engine_validated": proposal.get("engine_validated"),
        "reported_disposition": result.get("recommended_disposition") or result.get("reported_disposition"),
        "literature_claims": proposal.get("literature_analogies") or [],
    }
    return blind_value(view, max_string=1800, max_list=25)


def blind_candidate_id(run_directory: str, seed: int = 20260814) -> str:
    digest = hashlib.sha256(f"{seed}:{run_directory}".encode()).hexdigest().upper()
    return f"C-{digest[:6]}"


def criterion_ids(rubric: dict[str, Any], track: str) -> list[str]:
    return [item["criterion_id"] for item in rubric["tracks"][track]["criteria"]]


def absolute_response_schema(rubric: dict[str, Any], track: str) -> dict[str, Any]:
    ids = criterion_ids(rubric, track)
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "track": {"type": "string", "enum": [track]},
            "criterion_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string", "enum": ids},
                        "score": {"type": "integer"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "required_correction": {"type": ["string", "null"]},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["criterion_id", "score", "evidence", "required_correction", "confidence"],
                    "additionalProperties": False,
                },
            },
            "overall_comment": {"type": "string"},
        },
        "required": ["candidate_id", "track", "criterion_scores", "overall_comment"],
        "additionalProperties": False,
    }


def pairwise_response_schema(rubric: dict[str, Any], track: str) -> dict[str, Any]:
    ids = criterion_ids(rubric, track)
    return {
        "type": "object",
        "properties": {
            "pair_id": {"type": "string"},
            "track": {"type": "string", "enum": [track]},
            "criterion_preferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string", "enum": ids},
                        "preference": {"type": "string", "enum": ["A", "B", "TIE"]},
                        "evidence": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["criterion_id", "preference", "evidence", "confidence"],
                    "additionalProperties": False,
                },
            },
            "overall_preference": {"type": "string", "enum": ["A", "B", "TIE"]},
            "overall_reason": {"type": "string"},
        },
        "required": ["pair_id", "track", "criterion_preferences", "overall_preference", "overall_reason"],
        "additionalProperties": False,
    }


def validate_absolute_response(value: dict[str, Any], rubric: dict[str, Any], track: str, candidate_id: str) -> list[str]:
    errors = []
    if value.get("candidate_id") != candidate_id:
        errors.append("candidate_id mismatch")
    if value.get("track") != track:
        errors.append("track mismatch")
    expected = set(criterion_ids(rubric, track))
    rows = value.get("criterion_scores")
    if not isinstance(rows, list):
        return errors + ["criterion_scores is not a list"]
    observed = [row.get("criterion_id") for row in rows if isinstance(row, dict)]
    if set(observed) != expected or len(observed) != len(expected):
        errors.append(f"criterion IDs mismatch: {observed}")
    for row in rows:
        if not isinstance(row, dict):
            errors.append("non-object score row")
            continue
        score = row.get("score")
        if not isinstance(score, int) or not 0 <= score <= 4:
            errors.append(f"invalid score for {row.get('criterion_id')}: {score!r}")
        if not row.get("evidence"):
            errors.append(f"missing evidence for {row.get('criterion_id')}")
    return errors


def validate_pairwise_response(value: dict[str, Any], rubric: dict[str, Any], track: str, pair_id: str) -> list[str]:
    errors = []
    if value.get("pair_id") != pair_id:
        errors.append("pair_id mismatch")
    if value.get("track") != track:
        errors.append("track mismatch")
    expected = set(criterion_ids(rubric, track))
    rows = value.get("criterion_preferences")
    if not isinstance(rows, list):
        return errors + ["criterion_preferences is not a list"]
    observed = [row.get("criterion_id") for row in rows if isinstance(row, dict)]
    if set(observed) != expected or len(observed) != len(expected):
        errors.append(f"criterion IDs mismatch: {observed}")
    for row in rows:
        if not isinstance(row, dict) or row.get("preference") not in {"A", "B", "TIE"}:
            errors.append(f"invalid preference row: {row!r}")
    if value.get("overall_preference") not in {"A", "B", "TIE"}:
        errors.append("invalid overall preference")
    return errors


def rubric_text(rubric: dict[str, Any], track: str) -> str:
    criteria = rubric["tracks"][track]["criteria"]
    scale = rubric["score_scale"]
    lines = [
        f"TRACK: {track.upper()}",
        rubric["tracks"][track]["description"],
        "SCORE ANCHORS:",
        *[f"{score}: {description}" for score, description in scale.items()],
        "CRITERIA:",
    ]
    for item in criteria:
        lines.append(f"{item['criterion_id']} | {item['name']} | {item['question']}")
    return "\n".join(lines)


def absolute_prompt(rubric: dict[str, Any], track: str, case_packet: dict[str, Any], candidate_packet: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are an independent blinded evaluator of batch-to-flow chemistry designs. "
        "You are not part of the design pipeline. Judge only the supplied evidence. "
        "Do not infer the generator, architecture, or missing records. Do not reward verbosity, "
        "formatting, or stylistic familiarity. Recalculate simple arithmetic when needed. "
        "Use integer scores and the frozen anchors exactly. Return only the requested JSON."
    )
    evidence = candidate_packet["outcome_view"] if track == "outcome" else candidate_packet["assurance_view"]
    user = (
        rubric_text(rubric, track)
        + "\n\nCASE CONTEXT:\n"
        + json.dumps(case_packet, ensure_ascii=False, separators=(",", ":"))
        + f"\n\nBLINDED CANDIDATE {candidate_packet['candidate_id']} {track.upper()} EVIDENCE:\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        + "\n\nScore every criterion once. Cite candidate-specific evidence. Use null for required_correction only when no correction is needed."
    )
    return system, user


def pairwise_prompt(
    rubric: dict[str, Any],
    track: str,
    case_packet: dict[str, Any],
    pair_id: str,
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are an independent blinded evaluator comparing two batch-to-flow chemistry designs. "
        "You are not part of either design process. Judge only supplied evidence, use TIE when "
        "differences are not meaningful, and ignore formatting, length, or stylistic familiarity. "
        "Return only the requested JSON."
    )
    key = "outcome_view" if track == "outcome" else "assurance_view"
    user = (
        rubric_text(rubric, track)
        + "\n\nCASE CONTEXT:\n"
        + json.dumps(case_packet, ensure_ascii=False, separators=(",", ":"))
        + f"\n\nPAIR ID: {pair_id}\nCANDIDATE A:\n"
        + json.dumps(candidate_a[key], ensure_ascii=False, separators=(",", ":"))
        + "\n\nCANDIDATE B:\n"
        + json.dumps(candidate_b[key], ensure_ascii=False, separators=(",", ":"))
        + "\n\nFor every criterion choose A, B, or TIE and cite the decisive evidence. Then provide an overall preference."
    )
    return system, user


def weighted_track_score(scores: dict[str, float], rubric: dict[str, Any], track: str) -> float:
    criteria = rubric["tracks"][track]["criteria"]
    weighted = sum(float(scores[item["criterion_id"]]) * float(item["weight"]) for item in criteria)
    total_weight = sum(float(item["weight"]) for item in criteria)
    return 100.0 * weighted / (4.0 * total_weight)
