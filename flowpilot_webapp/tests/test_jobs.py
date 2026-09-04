from __future__ import annotations

import json

from flowpilot_webapp.backend import jobs


def test_completed_job_is_recovered_from_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path)
    target = tmp_path / "recover-me"
    target.mkdir()
    (target / "job.json").write_text(
        json.dumps({
            "job_id": "recover-me",
            "status": "completed",
            "progress": 1,
            "phase": "Design complete",
            "created_at": "2026-09-04T00:00:00+00:00",
            "messages": [],
            "autosave_dir": "outputs/gui_runs/example",
        }),
        encoding="utf-8",
    )
    (target / "result.json").write_text(
        json.dumps({"final_design": {"status": "executable"}}),
        encoding="utf-8",
    )

    manager = jobs.JobManager(max_workers=1)
    recovered = manager.get("recover-me")

    assert recovered is not None
    assert recovered.status == "completed"
    assert recovered.result["final_design"]["status"] == "executable"


def test_inflight_job_is_reported_as_interrupted_after_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path)
    target = tmp_path / "interrupted"
    target.mkdir()
    (target / "job.json").write_text(
        json.dumps({
            "job_id": "interrupted",
            "status": "running",
            "progress": 0.4,
            "phase": "Flow proposal",
            "created_at": "2026-09-04T00:00:00+00:00",
            "messages": [],
        }),
        encoding="utf-8",
    )

    manager = jobs.JobManager(max_workers=1)
    recovered = manager.get("interrupted")

    assert recovered is not None
    assert recovered.status == "failed"
    assert "backend restarted" in recovered.error
