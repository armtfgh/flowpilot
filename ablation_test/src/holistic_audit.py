from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


OMITTED_ARTIFACT_KEYS = {
    "png_path",
    "svg_path",
    "diagram_artifacts",
    "diagram_render_manifest",
    "pid_description",
}

# Generator provenance is joined after judging from the confidential key. It is
# not scientific evidence and must not reveal the candidate's architecture.
WITHHELD_IDENTITY_KEYS = {"variant"}

GENERATOR_PATTERNS = (
    (re.compile(r"flow\s*pilot", re.I), "the audited system"),
    (re.compile(r"flow\s*agent", re.I), "the audited system"),
    (re.compile(r"\bqwen[^\s,;:\]\[}\{\"]*", re.I), "a generator model"),
    (re.compile(r"\bgpt[-\w.]*", re.I), "a generator model"),
    (re.compile(r"\bopenai\b", re.I), "a model provider"),
    (re.compile(r"\bclaude[-\w.]*", re.I), "a generator model"),
    (re.compile(r"\banthropic\b", re.I), "a model provider"),
    (re.compile(r"\bone[- ]shot\b", re.I), "single-pass generation"),
)

DOMAIN_SECTIONS = {
    "chemistry_protocol": (
        "proposal",
        "raw_proposal",
        "pre_council_proposal",
        "chemistry_plan",
        "final_design",
        "process_topology",
        "unit_operations",
        "explanation",
    ),
    "numerical_engineering": (
        "proposal",
        "raw_proposal",
        "pre_council_proposal",
        "final_design",
        "design_calculations",
        "process_topology",
        "unit_operations",
        "inventory_allocation",
        "multistage_inventory_plan",
    ),
    "process_inventory": (
        "proposal",
        "raw_proposal",
        "final_design",
        "process_requirements_topology",
        "process_topology",
        "unit_operations",
        "instrument_manifest",
        "inventory_allocation",
        "inventory_enforcement",
        "design_realization",
        "multistage_inventory_plan",
    ),
    "safety_operations": (
        "proposal",
        "raw_proposal",
        "final_design",
        "chemistry_plan",
        "process_topology",
        "unit_operations",
        "safety_report",
        "design_realization",
        "design_disposition",
        "recommended_disposition",
        "reported_disposition",
        "disposition_rationale",
    ),
    "evidence_uncertainty": (
        "proposal",
        "raw_proposal",
        "pre_council_proposal",
        "chemistry_plan",
        "_analogies",
        "design_calculations",
        "safety_report",
        "design_disposition",
        "confidence",
        "recommended_disposition",
        "reported_disposition",
        "disposition_rationale",
        "design_space",
        "council_messages",
        "deliberation_log",
    ),
    "system_consistency": (
        "proposal",
        "raw_proposal",
        "pre_council_proposal",
        "final_design",
        "council_rounds",
        "council_messages",
        "design_realization",
        "final_validation",
        "design_disposition",
        "production_pipeline_complete",
        "pipeline_runtime",
        "output_contract",
        "response_envelope",
        "schema_valid",
        "confidence",
        "recommended_disposition",
        "reported_disposition",
        "disposition_rationale",
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_text(value: str) -> str:
    output = value
    for pattern, replacement in GENERATOR_PATTERNS:
        output = pattern.sub(replacement, output)
    return output


def sanitize(value: Any) -> Any:
    """Remove generator labels without truncating or dropping nested evidence."""
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, dict):
        return {_sanitize_text(str(key)): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def criteria_for_domain(rubric: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    return list(rubric["domains"][domain]["criteria"])


def all_criterion_ids(rubric: dict[str, Any]) -> list[str]:
    return [
        item["criterion_id"]
        for domain in rubric["domains"].values()
        for item in domain["criteria"]
    ]


def result_coverage(result: dict[str, Any]) -> dict[str, Any]:
    assignments: dict[str, list[str]] = {}
    for key in result:
        if key in OMITTED_ARTIFACT_KEYS:
            assignments[key] = ["artifact_only_omitted"]
            continue
        if key in WITHHELD_IDENTITY_KEYS:
            assignments[key] = ["identity_metadata_withheld"]
            continue
        assignments[key] = [domain for domain, keys in DOMAIN_SECTIONS.items() if key in keys]
    uncovered = sorted(key for key, domains in assignments.items() if not domains)
    return {
        "source_result_keys": sorted(result),
        "domain_assignments": assignments,
        "artifact_only_omissions": sorted(key for key in result if key in OMITTED_ARTIFACT_KEYS),
        "identity_metadata_withheld": sorted(key for key in result if key in WITHHELD_IDENTITY_KEYS),
        "uncovered_meaningful_keys": uncovered,
        "coverage_complete": not uncovered,
    }


def build_domain_record(
    result: dict[str, Any],
    *,
    candidate_id: str,
    case_label: str,
    domain: str,
) -> dict[str, Any]:
    if domain not in DOMAIN_SECTIONS:
        raise KeyError(domain)
    coverage = result_coverage(result)
    sections: dict[str, Any] = {}
    for key in DOMAIN_SECTIONS[domain]:
        sections[key] = result.get(key)
    packet = {
        "schema_version": "newgen_holistic_candidate_domain_v2.0",
        "candidate_id": candidate_id,
        "case_label": case_label,
        "audit_domain": domain,
        "record_policy": {
            "scope": "All source-result sections assigned to this audit domain are supplied without truncation.",
            "missing_section_meaning": "A null section means the generator did not provide that record. Do not infer its content.",
            "intermediate_values": "Intermediate or rejected values must be checked for clear status and propagation; they are errors only when presented as current, unresolved, or contradictory.",
        },
        "record_index": {
            "section_presence": {key: result.get(key) not in (None, {}, [], "") for key in DOMAIN_SECTIONS[domain]},
            "section_hashes": {
                key: json_sha256(result.get(key)) if result.get(key) not in (None, {}, [], "") else None
                for key in DOMAIN_SECTIONS[domain]
            },
            "full_result_coverage": coverage,
        },
        "record_sections": sections,
    }
    sanitized = sanitize(packet)
    encoded = json.dumps(sanitized, ensure_ascii=False)
    if "[truncated]" in encoded or "[depth-limited]" in encoded:
        raise ValueError("Holistic packet contains a truncation marker")
    return sanitized


def build_case_context(
    *,
    case_label: str,
    public_input: dict[str, Any],
    inventory: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    return sanitize({
        "schema_version": "newgen_holistic_case_context_v2.0",
        "case_label": case_label,
        "protocol": public_input.get("protocol"),
        "objective": public_input.get("objective"),
        "hard_constraints": public_input.get("hard_constraints") or [],
        "authoritative_inventory": inventory,
        "held_out_source_fact_check": {
            "batch_reference": oracle.get("batch_reference") or {},
            "published_flow_reference": oracle.get("published_flow_reference") or {},
            "interpretation": oracle.get("interpretation"),
        },
        "source_policy": (
            "Use the source to fact-check claims and arithmetic. The published flow design is not the only valid design, "
            "so a candidate differs from it only when the difference violates physics, protocol facts, inventory, safety, "
            "or lacks a defensible stated basis."
        ),
    })


def response_schema(rubric: dict[str, Any], domain: str) -> dict[str, Any]:
    ids = [item["criterion_id"] for item in criteria_for_domain(rubric, domain)]
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "domain": {"type": "string", "enum": [domain]},
            "criterion_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string", "enum": ids},
                        "status": {"type": "string", "enum": ["PASS", "ERROR", "NOT_ASSESSABLE"]},
                        "severity": {"type": "string", "enum": ["CRITICAL", "MAJOR", "MINOR", "NONE"]},
                        "error_title": {"type": "string", "maxLength": 160},
                        "evidence_paths": {
                            "type": "array", "maxItems": 8,
                            "items": {"type": "string", "maxLength": 240}
                        },
                        "observed_values": {
                            "type": "array", "maxItems": 8,
                            "items": {"type": "string", "maxLength": 240}
                        },
                        "expected_or_correct": {"type": "string", "maxLength": 600},
                        "explanation": {"type": "string", "maxLength": 900},
                        "required_correction": {"type": "string", "maxLength": 600},
                        "source_basis": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "PROTOCOL", "INVENTORY", "HELD_OUT_SOURCE", "CALCULATION",
                                    "PHYSICAL_LAW", "RECORD_CONSISTENCY", "SAFETY_PRACTICE", "NONE"
                                ]
                            }
                        },
                        "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}
                    },
                    "required": [
                        "criterion_id", "status", "severity", "error_title", "evidence_paths",
                        "observed_values", "expected_or_correct", "explanation", "required_correction",
                        "source_basis", "confidence"
                    ],
                    "additionalProperties": False
                }
            },
            "record_sections_reviewed": {"type": "array", "items": {"type": "string"}},
            "cross_domain_notes": {
                "type": "array", "maxItems": 8,
                "items": {"type": "string", "maxLength": 400}
            },
            "domain_summary": {"type": "string", "maxLength": 900}
        },
        "required": [
            "candidate_id", "domain", "criterion_findings", "record_sections_reviewed",
            "cross_domain_notes", "domain_summary"
        ],
        "additionalProperties": False
    }


def validate_response(
    value: dict[str, Any],
    rubric: dict[str, Any],
    domain: str,
    candidate_id: str,
) -> list[str]:
    errors: list[str] = []
    if value.get("candidate_id") != candidate_id:
        errors.append("candidate_id mismatch")
    if value.get("domain") != domain:
        errors.append("domain mismatch")
    expected = [item["criterion_id"] for item in criteria_for_domain(rubric, domain)]
    findings = value.get("criterion_findings")
    if not isinstance(findings, list):
        return errors + ["criterion_findings must be a list"]
    observed = [item.get("criterion_id") for item in findings if isinstance(item, dict)]
    if len(observed) != len(expected) or set(observed) != set(expected):
        errors.append(f"criterion set mismatch: {observed}")
    for item in findings:
        if not isinstance(item, dict):
            errors.append("finding is not an object")
            continue
        required_keys = {
            "criterion_id", "status", "severity", "error_title", "evidence_paths",
            "observed_values", "expected_or_correct", "explanation", "required_correction",
            "source_basis", "confidence",
        }
        missing_keys = sorted(required_keys - set(item))
        if missing_keys:
            errors.append(f"finding missing required keys: {missing_keys}")
        cid = item.get("criterion_id")
        status = item.get("status")
        severity = item.get("severity")
        if status == "ERROR":
            if severity not in {"CRITICAL", "MAJOR", "MINOR"}:
                errors.append(f"{cid}: ERROR requires non-NONE severity")
            for key in ("error_title", "evidence_paths", "observed_values", "expected_or_correct", "explanation", "required_correction", "source_basis"):
                if not item.get(key):
                    errors.append(f"{cid}: ERROR missing {key}")
            conclusion = f"{item.get('explanation', '')} {item.get('expected_or_correct', '')}".lower()
            if any(phrase in conclusion for phrase in ("no error found", "criterion is met", "is internally consistent")):
                errors.append(f"{cid}: ERROR contradicts its own conclusion")
        elif status in {"PASS", "NOT_ASSESSABLE"}:
            if severity != "NONE":
                errors.append(f"{cid}: {status} requires severity NONE")
            if not item.get("explanation"):
                errors.append(f"{cid}: {status} requires explanation")
        else:
            errors.append(f"{cid}: invalid status {status!r}")
    if not value.get("record_sections_reviewed"):
        errors.append("record_sections_reviewed is empty")
    for key in ("cross_domain_notes", "domain_summary"):
        if key not in value:
            errors.append(f"response missing required key: {key}")
    return errors


def audit_prompt(
    rubric: dict[str, Any],
    domain: str,
    case_context: dict[str, Any],
    candidate_record: dict[str, Any],
) -> tuple[str, str]:
    criteria = criteria_for_domain(rubric, domain)
    system = (
        "You are an independent forensic auditor of a batch-to-flow chemistry design record. "
        "Do not score quality, rank systems, or guess the generator or architecture. Audit every supplied section, "
        "including intermediate proposals, calculations, deliberation, validation, inventory assignments, safety records, "
        "and final outputs relevant to this domain. Recalculate arithmetic. Compare repeated values across paths. "
        "An earlier value is not an error when it is clearly rejected and correctly propagated; an unresolved or falsely "
        "validated contradiction is an error. Missing architecture-specific internal records are NOT_ASSESSABLE, not PASS "
        "and not ERROR. Do not reward verbosity or penalize brevity. Every ERROR must cite exact JSON-style paths, observed "
        "values, the correct expectation, source basis, and a concrete correction. Decide the final status before filling "
        "the JSON. Never leave status=ERROR after concluding that no error exists, and never place chain-of-thought or a "
        "self-debate in any field. Use at most two concise sentences in explanation and one sentence in each other text "
        "field. Return only concise audit conclusions in the required JSON."
    )
    criterion_text = "\n".join(
        f"{item['criterion_id']} | {item['name']} | {item['question']}"
        for item in criteria
    )
    user = (
        f"AUDIT DOMAIN: {domain}\n"
        f"DOMAIN PURPOSE: {rubric['domains'][domain]['description']}\n\n"
        f"FIXED CRITERIA:\n{criterion_text}\n\n"
        "STATUS RULES:\n"
        "PASS = supplied evidence supports the criterion and no error is found.\n"
        "ERROR = a specific contradiction, omission required for execution, invalid calculation, unsafe condition, or unsupported claim is found.\n"
        "NOT_ASSESSABLE = the relevant record is genuinely absent or insufficient; state exactly what evidence is missing.\n"
        "CRITICAL = could change chemistry identity, cause unsafe execution, or make the proposed run physically impossible.\n"
        "MAJOR = could materially change conditions, conversion, reproducibility, inventory feasibility, or disposition.\n"
        "MINOR = localized clarity or documentation defect unlikely to change the experiment.\n\n"
        "CASE CONTEXT:\n"
        + json.dumps(case_context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nCOMPLETE DOMAIN RECORD:\n"
        + json.dumps(candidate_record, ensure_ascii=False, separators=(",", ":"))
        + "\n\nAudit every fixed criterion exactly once and list every record section actually reviewed."
    )
    return system, user
