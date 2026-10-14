import json
from pathlib import Path

import pytest

from ablation_test.scripts import run_astra_repair_pilot as pilot


def test_pilot_stays_at_six_cells_even_after_evaluator_configuration():
    pilot.configure()
    pilot.evaluation.configure()
    cells = pilot.campaign.campaign_cells(pilot.campaign.selected_cases())
    assert len(cells) == 6
    assert {r["repeat_id"] for r in cells} == {"repeat_01"}
    assert {r["model_id"] for r in cells} == {"gpt-6-astra"}
    assert {r["candidate_budget"] for r in cells} == {2}
    assert {r["architecture"] for r in cells} == {"One-shot", "FlowPilot"}


def test_usage_retains_failed_attempt_cost_and_local_judge_has_no_api_charge(tmp_path):
    def save(name, value, jsonl=False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + ("\n" if jsonl else ""))

    event = {"model": "gpt-6-astra", "usage": {"input_tokens": 1000, "output_tokens": 200,
             "input_tokens_details": {"cached_tokens": 100}}}
    save("generation/case/astra_flowpilot/repeat_01/llm_events.jsonl", event, True)
    save("generation/case/astra_flowpilot/repeat_01/attempts/attempt_01/llm_events.jsonl", event, True)
    save("judgments/openai/id/telemetry.json", [{"model": "gpt-4o", "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}])
    save("judgments/claude/id/telemetry.json", [{"model": "claude-sonnet-4-6", "usage": {"input_tokens": 1000, "output_tokens": 100}}])
    save("judgments/qwen/id/telemetry.json", [{"model": "/models/Qwen3.6-27B", "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}])
    result = pilot.usage(tmp_path)
    assert result["generation_usd"] == pytest.approx(.0382)
    assert result["judging_usd"] == pytest.approx(.008)
    assert result["total_usd"] == pytest.approx(.0462)
    assert result["recorded_calls"] == 5


def test_progress_counts_successful_judge_retry_once(tmp_path):
    root = tmp_path / "judgments/qwen/candidate1"
    for suffix, status in [("status.json", "invalid"), ("attempts/attempt_01/status.json", "valid"), ("attempts/attempt_02/status.json", "valid")]:
        path = root / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status}))
    assert pilot.completed_judgments(tmp_path) == 1
