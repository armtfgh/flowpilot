import json

import pytest

from ablation_test.scripts.run_astra_workers import check_billing_failure


def write_error(tmp_path, message):
    path = tmp_path / "generation/case/astra_flowpilot/repeat_01/error.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"type": "RuntimeError", "error": message}))
    return path


@pytest.mark.parametrize("message", [
    "You have no credits remaining.", "insufficient_quota", "You exceeded your current quota.",
])
def test_exhausted_billing_stops_before_scoring(tmp_path, message):
    write_error(tmp_path, message)
    with pytest.raises(RuntimeError, match="No comparative scores"):
        check_billing_failure(tmp_path)
    status = json.loads((tmp_path / "supervisor_progress.json").read_text())
    assert status["phase"] == "interrupted_billing_exhausted"


def test_real_design_failure_is_not_excluded_as_a_billing_issue(tmp_path):
    write_error(tmp_path, "Final graph has an unknown residence-time basis")
    check_billing_failure(tmp_path)
    assert not (tmp_path / "supervisor_progress.json").exists()


def test_successful_retry_does_not_leave_stale_billing_block(tmp_path):
    path = write_error(tmp_path, "no credits remaining")
    path.with_name("run_summary.json").write_text(json.dumps({"status": "completed"}))
    check_billing_failure(tmp_path)
