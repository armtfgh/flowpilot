from pathlib import Path

from ablation_test.src.llm_judge import (
    absolute_response_schema,
    blind_value,
    criterion_ids,
    parse_json_object,
    validate_absolute_response,
    validate_pairwise_response,
    weighted_track_score,
)


ROOT = Path(__file__).resolve().parents[2]


def _rubric():
    import json

    return json.loads(
        (
            ROOT
            / "ablation_test"
            / "benchmarks"
            / "newgen_llm_judge_rubric_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_blinding_removes_architecture_and_model_family_terms_from_keys_and_values():
    value = blind_value(
        {
            "flowpilot_mode": "FlowPilot using Qwen and a council-approved result",
            "provider": "OpenAI GPT-5.4 and Claude by Anthropic",
        }
    )
    text = str(value).lower()
    for forbidden in ("flowpilot", "qwen", "council", "openai", "gpt-", "claude", "anthropic"):
        assert forbidden not in text


def test_absolute_response_requires_each_frozen_criterion_once():
    rubric = _rubric()
    ids = criterion_ids(rubric, "outcome")
    response = {
        "candidate_id": "C-ABC123",
        "track": "outcome",
        "criterion_scores": [
            {
                "criterion_id": criterion_id,
                "score": 3,
                "evidence": ["specific evidence"],
                "required_correction": None,
                "confidence": "high",
            }
            for criterion_id in ids
        ],
        "overall_comment": "Mostly correct.",
    }

    assert validate_absolute_response(response, rubric, "outcome", "C-ABC123") == []
    response["criterion_scores"].pop()
    assert validate_absolute_response(response, rubric, "outcome", "C-ABC123")


def test_pairwise_response_maps_only_frozen_preferences():
    rubric = _rubric()
    ids = criterion_ids(rubric, "assurance")
    response = {
        "pair_id": "P-01",
        "track": "assurance",
        "criterion_preferences": [
            {
                "criterion_id": criterion_id,
                "preference": "TIE",
                "evidence": "No meaningful difference.",
                "confidence": "medium",
            }
            for criterion_id in ids
        ],
        "overall_preference": "TIE",
        "overall_reason": "Equivalent evidence.",
    }

    assert validate_pairwise_response(response, rubric, "assurance", "P-01") == []
    response["criterion_preferences"][0]["preference"] = "C"
    assert validate_pairwise_response(response, rubric, "assurance", "P-01")


def test_weighted_track_score_is_calculated_by_code_not_judge():
    rubric = _rubric()
    all_four = {criterion_id: 4 for criterion_id in criterion_ids(rubric, "outcome")}
    all_two = {criterion_id: 2 for criterion_id in criterion_ids(rubric, "outcome")}

    assert weighted_track_score(all_four, rubric, "outcome") == 100.0
    assert weighted_track_score(all_two, rubric, "outcome") == 50.0


def test_parse_json_object_accepts_fenced_json():
    assert parse_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_absolute_schema_is_bound_to_selected_track():
    rubric = _rubric()
    schema = absolute_response_schema(rubric, "outcome")
    assert schema["properties"]["track"]["enum"] == ["outcome"]
    item_schema = schema["properties"]["criterion_scores"]["items"]
    assert set(item_schema["properties"]["criterion_id"]["enum"]) == {
        "O-01", "O-02", "O-03", "O-04", "O-05", "O-06"
    }
