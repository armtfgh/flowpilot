import json

from ablation_test.src.cases import ROOT, load_cases_from_path


STUDY_DIR = ROOT / "studies" / "fair_architecture_benchmark_v1_1_20260730"


def test_stage2_matrix_is_120_cells():
    config = json.loads((STUDY_DIR / "stage2_config.json").read_text())
    assert len(config["scenario_ids"]) == 10
    assert len(config["conditions"]) == 4
    assert config["repeats"] == 3
    assert len(config["scenario_ids"]) * len(config["conditions"]) * 3 == 120
    assert config["completion_gate"]["planned_cells"] == 120


def test_stage2_scenarios_exist_and_are_unique():
    config = json.loads((STUDY_DIR / "stage2_config.json").read_text())
    cases = load_cases_from_path(STUDY_DIR / config["scenario_file"])
    ids = [case.scenario_id for case in cases]
    assert len(ids) == len(set(ids)) == 10
    assert set(config["scenario_ids"]) == set(ids)


def test_each_pair_has_one_feasible_and_one_infeasible_scenario():
    cases = load_cases_from_path(STUDY_DIR / "protocol_scenarios.json")
    pairs = {}
    for case in cases:
        pairs.setdefault(case.pair_id, set()).add(
            (case.scenario_kind, case.expected_disposition)
        )
    assert len(pairs) == 5
    assert all(
        values == {("feasible", "SCREEN"), ("infeasible", "BLOCK")}
        for values in pairs.values()
    )
