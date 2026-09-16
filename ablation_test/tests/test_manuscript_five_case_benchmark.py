from pathlib import Path

from ablation_test.scripts.run_manuscript_five_case_benchmark import (
    campaign_cells,
    initialize,
    load_frozen_cases,
)
from ablation_test.src.newgen_2_outcome import read_json


def test_suite_has_five_cases_and_twenty_matched_cells():
    cases = load_frozen_cases()
    cells = campaign_cells(cases)
    assert len(cases) == 5
    assert len(cells) == 20
    assert {row["generator_family"] for row in cells} == {"qwen", "openai"}
    assert {row["architecture"] for row in cells} == {"One-shot", "FlowPilot"}
    for _, case in cases:
        hashes = {row["design_input_sha256"] for row in cells if row["case_id"] == case.case_id}
        assert hashes == {case.design_input_sha256}
        assert case.chemistry_identity_confirmation.get("confirmed") is True


def test_initialize_freezes_plan_without_network(tmp_path: Path):
    cases = load_frozen_cases()
    initialize(tmp_path, cases)
    campaign = read_json(tmp_path / "frozen/campaign_manifest.json")
    plan = read_json(tmp_path / "frozen/execution_plan.json")
    assert campaign["candidate_count"] == 20
    assert campaign["planned_judgments"] == 40
    assert len(plan["cells"]) == 20
    assert (tmp_path / "frozen/SELECTION_POLICY.md").is_file()


def test_held_out_source_is_excluded_for_every_case():
    for _, case in load_frozen_cases():
        assert case.source_record_id
        assert case.source_record_id in case.excluded_record_ids
        assert case.source_record_id not in case.design_input_text
