import json
from pathlib import Path

import pandas as pd

from ablation_test.scripts.cross_model_report import matched_results
from ablation_test.src.cases import ROOT


def test_cross_model_conditions_cover_matched_controls():
    config = json.loads(
        (ROOT / "configs" / "cross_model_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    conditions = config["conditions"]
    assert conditions["qwen_one_shot"] == {
        "label": "Qwen 27B one-shot",
        "variant": "general_one_shot",
        "bundle": "qwen",
    }
    assert conditions["qwen_full"]["variant"] == "full"
    assert conditions["qwen_full"]["bundle"] == "qwen"
    assert conditions["gpt4o_one_shot"]["variant"] == "general_one_shot"
    assert conditions["claude_one_shot"]["variant"] == "general_one_shot"
    assert conditions["gpt4o_full"]["variant"] == "full"


def test_matched_results_drops_incomplete_case_repeat():
    conditions = ["qwen_one_shot", "qwen_full", "gpt4o_one_shot"]
    rows = []
    for condition in conditions:
        rows.append(
            {
                "condition_id": condition,
                "case_id": "complete",
                "repeat": 1,
                "status": "completed",
            }
        )
    rows.extend(
        [
            {
                "condition_id": "qwen_one_shot",
                "case_id": "incomplete",
                "repeat": 1,
                "status": "completed",
            },
            {
                "condition_id": "qwen_full",
                "case_id": "incomplete",
                "repeat": 1,
                "status": "failed",
            },
        ]
    )
    matched = matched_results(pd.DataFrame(rows), conditions)
    assert set(matched["case_id"]) == {"complete"}
    assert len(matched) == 3


def test_frozen_scorer_exists():
    assert (ROOT / "src" / "metrics.py").is_file()
