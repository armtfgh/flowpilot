import json
from pathlib import Path

from ablation_test.src.cases import ROOT, load_cases_from_path
from ablation_test.src.stage1_oracle import (
    score_scenario,
    validate_oracle_witness,
)


STUDY_DIR = ROOT / "studies" / "fair_architecture_benchmark_v1_20260730"
STUDY_V11_DIR = ROOT / "studies" / "fair_architecture_benchmark_v1_1_20260730"


def _cases():
    return load_cases_from_path(STUDY_DIR / "protocol_scenarios.json")


def test_stage1_has_five_matched_feasible_infeasible_pairs():
    cases = _cases()
    assert len(cases) == 10
    pairs = {}
    for case in cases:
        pairs.setdefault(case.pair_id, set()).add(case.scenario_kind)
    assert len(pairs) == 5
    assert all(kinds == {"feasible", "infeasible"} for kinds in pairs.values())


def test_all_oracle_witnesses_pass():
    outcomes = [validate_oracle_witness(case) for case in _cases()]
    assert all(outcome["passed"] for outcome in outcomes)


def test_oracle_rejects_known_feasible_constraint_violation():
    case = next(case for case in _cases() if case.case_id == "photo_oxidation_feasible")
    result = {
        "reported_disposition": "SCREEN",
        "proposal": {
            "wavelength_nm": 525,
            "flow_rate_mL_min": 0.005,
            "reactor_volume_mL": 20,
        },
    }
    score = score_scenario(case, result)
    assert score["disposition_correct"]
    assert not score["critical_engineering_pass"]
    assert score["critical_violation_count"] == 3


def test_oracle_is_condition_label_independent():
    case = next(case for case in _cases() if case.case_id == "suzuki_feasible")
    result = {
        "reported_disposition": "SCREEN",
        "proposal": {
            "temperature_C": 50,
            "flow_rate_mL_min": 0.5,
            "reactor_volume_mL": 5,
        },
    }
    first = score_scenario(case, {**result, "condition_id": "qwen"})
    second = score_scenario(case, {**result, "condition_id": "claude"})
    assert first == second


def test_smoke_plan_is_exactly_eight_cells():
    config = json.loads(
        (STUDY_DIR / "benchmark_config.json").read_text(encoding="utf-8")
    )
    assert len(config["smoke_scenario_ids"]) == 2
    assert len(config["conditions"]) == 4
    assert config["stage_1_gate"]["planned_cells"] == 8


def test_canonical_design_input_does_not_expose_hidden_oracle():
    case = _cases()[0]
    public = case.public_payload()
    text = json.dumps(public, ensure_ascii=False).lower()
    assert "expected_disposition" not in text
    assert "oracle_witness" not in text
    assert "oracle_constraints" not in text


def test_v11_photochemical_pair_has_identical_gas_hardware():
    cases = load_cases_from_path(STUDY_V11_DIR / "protocol_scenarios.json")
    feasible = next(case for case in cases if case.case_id == "photo_oxidation_feasible")
    infeasible = next(
        case for case in cases if case.case_id == "photo_oxidation_infeasible"
    )
    assert feasible.inventory["gas_hardware"]
    assert feasible.inventory["gas_hardware"] == infeasible.inventory["gas_hardware"]
    assert feasible.inventory["light_sources"][0]["wavelength_nm"] == 420
    assert infeasible.inventory["light_sources"][0]["wavelength_nm"] == 525
