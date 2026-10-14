from types import SimpleNamespace
import io
import json
import pytest

from ablation_test.scripts.evaluate_astra_delivered_outputs import delivered_result, delivered_verification, exact_local_context, local_judge_output_budget


def test_keeps_raw_one_shot_values_without_mutating_saved_result():
    source = {"variant": "general_one_shot", "proposal": {"residence_time_min": 1.43},
              "raw_proposal": {"residence_time_min": 30, "residence_time_basis": "empty bed"}}
    result = delivered_result(source)
    assert result["proposal"]["residence_time_min"] == 30
    assert source["proposal"]["residence_time_min"] == 1.43
    result["proposal"]["residence_time_min"] = 15
    assert source["raw_proposal"]["residence_time_min"] == 30


def test_pipeline_is_not_replaced_by_an_intermediate_proposal():
    result = {"variant": "full", "proposal": {"x": 1}, "raw_proposal": {"x": 2},
              "final_design": {"parameters": {"x": 3}}}
    assert delivered_result(result) == result


def test_gas_time_bases_are_not_falsely_scored_as_liquid_closure():
    case = SimpleNamespace(expected_features={"gas_required": True}, inventory={"reactors": [{"volume_mL": 3}]}, excluded_record_ids=[])
    result = {"proposal": {"flow_rate_mL_min": .1, "residence_time_min": 1.43, "reactor_volume_mL": 3}}
    closure = delivered_verification(result, {}, case)["top_level_volume_flow_time_closure"]
    assert closure["assessable"] is False
    assert closure["passes_10_percent"] is None
    assert closure["arithmetic_only"]["calculated_Q_tau_mL"] == .143


def test_liquid_only_calculation_remains_deterministic():
    case = SimpleNamespace(expected_features={}, inventory={"reactors": [{"volume_mL": 3}]}, excluded_record_ids=[])
    result = {"proposal": {"flow_rate_mL_min": .1, "residence_time_min": 30, "reactor_volume_mL": 3}}
    assert delivered_verification(result, {}, case)["top_level_volume_flow_time_closure"]["passes_10_percent"] is True


def test_judge_context_uses_real_token_count_without_trimming(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps({"count": 10000, "max_model_len": 32768}).encode()))
    messages = [{"role": "user", "content": "x" * 60000}]
    system, retained = exact_local_context("S", messages, max_tokens=12000, audit_path=tmp_path / "audit.jsonl")
    assert system == "S" and retained == messages
    assert len(retained[0]["content"]) == 60000


def test_oversized_judge_packet_fails_instead_of_silent_evidence_loss(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps({"count": 24000, "max_model_len": 32768}).encode()))
    with pytest.raises(RuntimeError, match="evidence was not truncated"):
        exact_local_context("S", [{"role": "user", "content": "U"}], max_tokens=12000, audit_path=tmp_path / "audit.jsonl")


@pytest.mark.parametrize("count, fits", [(19468, True), (20256, True), (20257, False)])
def test_exact_context_boundary_keeps_full_output_budget(monkeypatch, tmp_path, count, fits):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps({"count": count, "max_model_len": 32768}).encode()))
    messages = [{"role": "user", "content": "unaltered evidence"}]
    if fits:
        assert exact_local_context("S", messages, max_tokens=12000, audit_path=tmp_path / "audit.jsonl") == ("S", messages)
    else:
        with pytest.raises(RuntimeError, match="evidence was not truncated"):
            exact_local_context("S", messages, max_tokens=12000, audit_path=tmp_path / "audit.jsonl")
    audit = json.loads((tmp_path / "audit.jsonl").read_text())
    assert audit["output_allowance"] == 12000
    assert audit["safety_margin"] == 512
    assert audit["truncated"] is False


@pytest.mark.parametrize("count, expected", [(10000, 12000), (21513, 10743), (24756, 7500), (26234, 6022), (28160, 4096)])
def test_output_reservation_adapts_without_removing_evidence(monkeypatch, tmp_path, count, expected):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps({"count": count, "max_model_len": 32768}).encode()))
    messages = [{"role": "user", "content": "complete original evidence"}]
    assert local_judge_output_budget("S", messages, audit_path=tmp_path / "audit.jsonl") == expected
    assert messages == [{"role": "user", "content": "complete original evidence"}]
    assert exact_local_context("S", messages, max_tokens=expected, audit_path=tmp_path / "audit.jsonl") == ("S", messages)


def test_output_reservation_rejects_insufficient_room(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps({"count": 28161, "max_model_len": 32768}).encode()))
    with pytest.raises(RuntimeError, match="below the 4096-token minimum; evidence was not truncated"):
        local_judge_output_budget("S", [{"role": "user", "content": "U"}], audit_path=tmp_path / "audit.jsonl")
