from copy import deepcopy
import pytest
from ablation_test.src.newgen_2_outcome import build_outcome_packet, validate_response
from ablation_test.src.newgen_2_outcome import evidence_path_catalog, response_schema
from ablation_test.src.outcome_verification import delivered_arithmetic, resolves_json_path
from ablation_test.tests.test_newgen_2_outcome import _packet, _response, _rubric


def test_air_arithmetic_is_identical_across_architectures_and_keeps_inputs():
    outcome = {"parameters": {"flow_rate_mL_min": .20544, "concentration_M": .1,
        "temperature_C": 21, "pressure_absolute_bar": 4.01325, "reactor_volume_mL": 1},
        "streams": [{"phase": "gas", "contents": ["air"], "gas_flow_sccm": 2.192605,
            "gas_flow_actual_mL_min": .59614, "molar_equiv": 1}]}
    original = deepcopy(outcome)
    report = delivered_arithmetic(outcome)
    gas = report["gas_streams"][0]
    assert gas["calculated_active_equivalents"] == pytest.approx(1, rel=1e-4)
    assert gas["equivalents_match_active_species_within_rounding"]
    assert gas["channel_flow_matches_within_rounding"]
    assert outcome == original
    assert "unassessable_reason" in delivered_arithmetic(outcome, multistage=True)


def test_shared_parameter_boundary_retains_actual_block_and_issue():
    proposal = {"temperature_C": 40, "inventory_constraints": {"flow_recalculation": {"temperature_C": 100}}}
    args = dict(candidate="test", case_context={}, deterministic_verification={}, has_gas=False, is_multistage=False)
    os = build_outcome_packet(**args, result={"proposal": proposal})
    fp = build_outcome_packet(**args, result={"variant": "full", "proposal": proposal,
        "final_design": {"status": "blocked", "consistency": {"issues": [{"code": "REAL-DEFECT"}]}},
        "reported_disposition": "BLOCK", "process_topology": {"unit_operations": []}})
    assert os["delivered_final_outcome"]["parameters"] == fp["delivered_final_outcome"]["parameters"]
    assert fp["delivered_final_outcome"]["delivery_issues"][0]["code"] == "REAL-DEFECT"
    assert fp["delivered_final_outcome"]["reported_disposition"] == "BLOCK"
    assert fp["delivered_final_outcome"]["topology"]["unit_operations"] == []


def test_nonexistent_evidence_path_and_full_marks_for_missing_delivery_are_rejected():
    packet = _packet()
    response = _response(packet)
    response["criterion_scores"][0]["evidence_paths"] = ["$.invented.field"]
    packet["delivered_final_outcome"].update(delivery_status="blocked", topology={"unit_operations": []})
    errors = validate_response(response, _rubric(), packet)
    assert any("does not resolve" in e for e in errors)
    assert any("UO-05" in e and "empty" in e for e in errors)
    assert any("UO-13" in e and "no executable" in e for e in errors)
    assert response["criterion_scores"][0]["score"] == 4  # validation never rewrites a score


@pytest.mark.parametrize("path", ["$.a[0].b", "a[0]['b']", 'a[0]["b"]'])
def test_evidence_path_resolution(path):
    assert resolves_json_path({"a": [{"b": None}]}, path)
    assert not resolves_json_path({"a": []}, path)


def test_citation_catalog_constrains_paths_without_changing_evidence_or_scores():
    packet = _packet()
    original = deepcopy(packet)
    catalog = evidence_path_catalog(packet)
    assert catalog and all(resolves_json_path(packet, path) for path in catalog)
    assert "$.delivered_final_outcome.chemistry_and_operations" in catalog
    assert packet == original
    schema = response_schema(packet)
    fields = schema["properties"]["criterion_scores"]["items"]["properties"]
    assert fields["evidence_paths"]["items"]["enum"] == catalog
    assert fields["score"] == response_schema()["properties"]["criterion_scores"]["items"]["properties"]["score"]
