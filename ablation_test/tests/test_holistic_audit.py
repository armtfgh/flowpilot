import json
from pathlib import Path

from ablation_test.src.holistic_audit import (
    all_criterion_ids,
    audit_prompt,
    build_domain_record,
    response_schema,
    stable_seed,
    validate_response,
)
from ablation_test.scripts.run_newgen_holistic_audit_v2 import _schema_for_judge


ROOT = Path(__file__).resolve().parents[2]
RUBRIC = json.loads(
    (ROOT / "ablation_test" / "benchmarks" / "newgen_holistic_error_audit_v2.json").read_text(encoding="utf-8")
)


def _valid_response(domain: str, candidate_id: str) -> dict:
    findings = []
    for item in RUBRIC["domains"][domain]["criteria"]:
        findings.append({
            "criterion_id": item["criterion_id"],
            "status": "PASS",
            "severity": "NONE",
            "error_title": "",
            "evidence_paths": [],
            "observed_values": [],
            "expected_or_correct": "",
            "explanation": "The supplied evidence closes this criterion.",
            "required_correction": "",
            "source_basis": ["NONE"],
            "confidence": "HIGH",
        })
    return {
        "candidate_id": candidate_id,
        "domain": domain,
        "criterion_findings": findings,
        "record_sections_reviewed": ["record_sections.proposal"],
        "cross_domain_notes": [],
        "domain_summary": "No errors found.",
    }


def test_rubric_has_29_unique_fixed_criteria_and_no_score():
    ids = all_criterion_ids(RUBRIC)
    assert len(ids) == 29
    assert len(ids) == len(set(ids))
    assert RUBRIC["composite_score"] is False
    assert "score" not in json.dumps(response_schema(RUBRIC, "chemistry_protocol")).lower()


def test_stable_seed_depends_on_ids_not_loop_positions():
    first = stable_seed("holistic-v2", "claude", "numerical_engineering", "H-123", "audit-pass-1")
    second = stable_seed("holistic-v2", "claude", "numerical_engineering", "H-123", "audit-pass-1")
    other = stable_seed("holistic-v2", "claude", "numerical_engineering", "H-124", "audit-pass-1")
    assert first == second
    assert first != other
    retry = stable_seed("holistic-v2", "claude", "numerical_engineering", "H-123", "audit-pass-1", "attempt_2")
    retry_again = stable_seed("holistic-v2", "claude", "numerical_engineering", "H-123", "audit-pass-1", "attempt_2")
    assert retry == retry_again


def test_domain_record_is_complete_untruncated_and_sanitized():
    result = {
        "proposal": {"note": "FlowPilot generated this with GPT-4o"},
        "raw_proposal": {},
        "pre_council_proposal": {},
        "chemistry_plan": {},
        "final_design": {},
        "process_topology": {},
        "unit_operations": [],
        "explanation": "ok",
    }
    record = build_domain_record(result, candidate_id="H-123", case_label="Case", domain="chemistry_protocol")
    text = json.dumps(record)
    assert "FlowPilot" not in text
    assert "GPT-4o" not in text
    assert "[truncated]" not in text
    assert "[depth-limited]" not in text
    assert record["record_index"]["full_result_coverage"]["coverage_complete"]


def test_generator_variant_is_withheld_from_packets():
    result = {"variant": "general_one_shot", "proposal": {"temperature_C": 40}}
    record = build_domain_record(
        result, candidate_id="H-123", case_label="Case", domain="system_consistency"
    )
    text = json.dumps(record).lower()
    assert "general_one_shot" not in text
    assert record["record_index"]["full_result_coverage"]["identity_metadata_withheld"] == ["variant"]


def test_validation_rejects_error_without_actionable_evidence():
    value = _valid_response("chemistry_protocol", "H-123")
    value["criterion_findings"][0].update({"status": "ERROR", "severity": "MAJOR"})
    errors = validate_response(value, RUBRIC, "chemistry_protocol", "H-123")
    assert any("missing error_title" in error for error in errors)
    assert any("missing required_correction" in error for error in errors)


def test_validation_rejects_error_that_concludes_no_error():
    value = _valid_response("chemistry_protocol", "H-123")
    value["criterion_findings"][0].update({
        "status": "ERROR", "severity": "MAJOR", "error_title": "Contradiction",
        "evidence_paths": ["$.proposal.x"], "observed_values": ["x=1"],
        "expected_or_correct": "x=2", "explanation": "No error found after review.",
        "required_correction": "Set x to 2.", "source_basis": ["CALCULATION"],
    })
    errors = validate_response(value, RUBRIC, "chemistry_protocol", "H-123")
    assert any("contradicts its own conclusion" in error for error in errors)


def test_validation_rejects_missing_required_finding_key():
    value = _valid_response("chemistry_protocol", "H-123")
    del value["criterion_findings"][0]["confidence"]
    errors = validate_response(value, RUBRIC, "chemistry_protocol", "H-123")
    assert any("missing required keys" in error for error in errors)


def test_validation_accepts_complete_fixed_criterion_set():
    value = _valid_response("process_inventory", "H-123")
    assert validate_response(value, RUBRIC, "process_inventory", "H-123") == []


def test_prompt_requires_full_record_audit_not_final_only():
    domain = "system_consistency"
    record = build_domain_record(
        {key: {} for key in (
            "proposal", "raw_proposal", "pre_council_proposal", "final_design", "design_calculations",
            "design_space", "council_rounds", "council_messages", "deliberation_log", "design_realization",
            "inventory_allocation", "instrument_manifest", "process_topology", "unit_operations", "safety_report",
            "final_validation", "design_disposition", "production_pipeline_complete", "pipeline_runtime",
            "output_contract", "response_envelope", "schema_valid", "variant", "confidence",
            "recommended_disposition", "reported_disposition", "disposition_rationale"
        )},
        candidate_id="H-123",
        case_label="Case",
        domain=domain,
    )
    system, user = audit_prompt(RUBRIC, domain, {"protocol": "test"}, record)
    assert "including intermediate proposals" in system
    assert "Audit every supplied section" in system
    assert "COMPLETE DOMAIN RECORD" in user


def test_claude_schema_removes_only_unsupported_length_keywords():
    schema = response_schema(RUBRIC, "chemistry_protocol")
    cleaned = _schema_for_judge(schema, "claude")
    encoded = json.dumps(cleaned)
    assert "maxItems" not in encoded
    assert "maxLength" not in encoded
    assert cleaned["required"] == schema["required"]
