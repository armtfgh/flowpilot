import numpy as np
import pandas as pd

from ablation_test.scripts.build_newgen_llm_judge_report import _icc_2_1, _resolved_call, pairwise_summaries


def test_icc_is_one_for_identical_judges():
    ratings = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 4, 4]], dtype=float)
    assert _icc_2_1(ratings) == 1.0


def test_pairwise_summary_uses_normalized_architecture_and_order_consistency():
    rows = []
    for order, raw in ((1, "A"), (2, "B")):
        rows.append({
            "judge": "qwen",
            "track": "outcome",
            "pair_id": "P-01",
            "order": order,
            "generator_model": "Model X",
            "generator_family": "x",
            "case": "Case 1",
            "criterion_id": "OVERALL",
            "raw_preference": raw,
            "normalized_preference": "FlowPilot",
        })
    summary, consistency, consensus = pairwise_summaries(pd.DataFrame(rows))
    assert int(summary.loc[0, "FlowPilot"]) == 2
    assert bool(consistency.loc[0, "order_consistent"])
    assert consensus.loc[0, "consensus_preference"] == "FlowPilot"


def test_resolved_call_preserves_invalid_original_and_uses_valid_child(tmp_path):
    call = tmp_path / "cell"
    call.mkdir()
    (call / "status.json").write_text('{"status":"invalid"}', encoding="utf-8")
    attempt = call / "attempts" / "attempt_2"
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text('{"status":"valid"}', encoding="utf-8")
    status, response_dir, initial_status, attempt_count = _resolved_call(call / "status.json")
    assert status["status"] == "valid"
    assert response_dir == attempt
    assert initial_status == "invalid"
    assert attempt_count == 2


def test_resolved_call_prefers_explicit_valid_child_over_valid_original(tmp_path):
    call = tmp_path / "cell"
    call.mkdir()
    (call / "status.json").write_text('{"status":"valid","seed":1}', encoding="utf-8")
    attempt = call / "attempts" / "attempt_2"
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text('{"status":"valid","seed":1}', encoding="utf-8")
    status, response_dir, initial_status, attempt_count = _resolved_call(call / "status.json")
    assert status["status"] == "valid"
    assert response_dir == attempt
    assert initial_status == "valid"
    assert attempt_count == 2
