from __future__ import annotations

from ablation_test.src.newgen_2_outcome import (
    build_outcome_packet, normalize_response, packet_gate, response_schema,
    validate_response, write_json,
)


def _rubric():
    return {"criteria": [{"criterion_id": f"UO-{i:02d}"} for i in range(1, 15)]}


def _packet(gas=False, multistage=False):
    return build_outcome_packet(
        candidate="N2-TEST", case_context={"case_label": "case", "protocol": "A to B", "objective": "screen", "hard_constraints": [], "authoritative_inventory": {}, "held_out_source_fact_check": {}, "source_policy": "fact check"},
        result={"proposal": {"flow_rate_mL_min": 0.1, "streams": [], "stage_parameters": []}, "reported_disposition": "SCREEN", "variant": "qwen one-shot"},
        deterministic_verification={"criteria": []}, has_gas=gas, is_multistage=multistage,
    )


def _response(packet, gas=False, multistage=False):
    rows = []
    for i in range(1, 15):
        cid = f"UO-{i:02d}"
        na = (cid == "UO-08" and not gas) or (cid == "UO-09" and not multistage)
        rows.append({
            "criterion_id": cid, "applicability": "NOT_APPLICABLE" if na else "APPLICABLE",
            "score": None if na else 4, "critical_error": False, "severity": "NONE",
            "evidence_paths": [], "observed_values": [], "expected_or_correct": "",
            "rationale": "not relevant" if na else "complete", "required_correction": "",
            "confidence": "HIGH",
        })
    return {"candidate_id": packet["candidate_id"], "criterion_scores": rows, "overall_executability": "EXECUTABLE", "cross_criterion_notes": []}


def test_packet_is_common_and_identity_blinded():
    packet = _packet()
    assert not packet_gate(packet)
    assert "qwen" not in str(packet).lower()
    assert set(packet["delivered_final_outcome"]) >= {"parameters", "streams", "stages", "topology", "chemistry_and_operations", "safety_controls"}


def test_blocked_pipeline_uses_final_contract_normalizer():
    packet = build_outcome_packet(
        candidate="N2-BLOCKED",
        case_context={
            "case_label": "case", "protocol": "A to B", "objective": "screen",
            "hard_constraints": [], "authoritative_inventory": {},
            "held_out_source_fact_check": {}, "source_policy": "fact check",
        },
        result={
            "variant": "full",
            "proposal": {"flow_rate_mL_min": 9.9},
            "final_design": {
                "status": "blocked",
                "parameters": {"flow_rate_mL_min": 0.2},
                "streams": [], "stages": [],
            },
            "process_topology": {"status": "requirements_only"},
            "recommended_disposition": "BLOCK",
        },
        deterministic_verification={"checks": []},
        has_gas=False,
        is_multistage=False,
    )
    assert packet["delivered_final_outcome"]["parameters"]["flow_rate_mL_min"] == 0.2
    assert packet["delivered_final_outcome"]["topology"]["status"] == "requirements_only"


def test_response_contract_accepts_only_strict_na_rules():
    packet = _packet()
    response = _response(packet)
    assert not validate_response(response, _rubric(), packet)
    response["criterion_scores"][0]["applicability"] = "NOT_APPLICABLE"
    response["criterion_scores"][0]["score"] = None
    assert any("NOT_APPLICABLE is not allowed" in item for item in validate_response(response, _rubric(), packet))


def test_normalize_response_canonicalizes_only_na_fields():
    packet = _packet()
    response = _response(packet)
    na_row = response["criterion_scores"][7]
    na_row.update({"score": 4, "critical_error": True, "severity": "CRITICAL"})
    applicable_row = response["criterion_scores"][0]
    normalize_response(response)
    assert na_row["score"] is None
    assert na_row["critical_error"] is False
    assert na_row["severity"] == "NONE"
    assert applicable_row["score"] == 4


def test_low_score_requires_evidence_and_correction():
    packet = _packet(gas=True, multistage=True)
    response = _response(packet, gas=True, multistage=True)
    response["criterion_scores"][5].update({"score": 1, "severity": "MAJOR"})
    errors = validate_response(response, _rubric(), packet)
    assert any("requires evidence paths" in item for item in errors)
    assert any("requires required_correction" in item for item in errors)


def test_schema_has_equal_fixed_criterion_rows():
    schema = response_schema()
    rows = schema["properties"]["criterion_scores"]
    assert rows["minItems"] == rows["maxItems"] == 14
    assert rows["items"]["properties"]["score"]["maximum"] == 4


def test_report_aggregation_requires_and_uses_selected_judges(tmp_path):
    from ablation_test.scripts.build_newgen_2_0_report import aggregate

    root, output = tmp_path / "campaign", tmp_path / "report"
    packet = _packet()
    write_json(root / "frozen/outcome_rubric.json", _rubric())
    write_json(root / "frozen/campaign_manifest.json", {"selected_judges": ["qwen", "openai"]})
    write_json(root / "frozen/candidate_key_confidential.json", {"candidates": [{
        "candidate_id": "N2-TEST", "case": "case", "generator_model": "model",
        "generator_family": "qwen", "architecture": "One-shot", "run_directory": "/tmp/run",
    }]})
    write_json(root / "packets/N2-TEST.json", packet)
    response = _response(packet)
    for judge in ("qwen", "openai"):
        write_json(root / f"judgments/{judge}/N2-TEST/status.json", {"status": "valid"})
        write_json(root / f"judgments/{judge}/N2-TEST/parsed_response.json", response)
    summary = aggregate(root, output)
    assert summary["judgment_count"] == 2
    assert summary["selected_judges"] == ["qwen", "openai"]
    assert summary["architecture_summary"][0]["mean_score_0_1"] == 1.0
    assert (output / "tables/criterion_judgments.csv").is_file()
