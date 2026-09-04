from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from flowpilot_webapp.backend.app import app
from flowpilot_webapp.backend.app import JOBS
from flora_translate.intake_agent import QUESTION_BANK


client = TestClient(app)


def test_health_reports_pipeline_and_models() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "FlowPilot"
    assert payload["models"]["chemistry"]


def test_model_catalog_exposes_stable_upstream_and_downstream_routes() -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    route_ids = {item["route_id"] for item in payload["models"]}
    assert {"claude-opus-4-6", "claude-sonnet-4-6", "gpt-4o"} <= route_ids
    assert {"qwen3.6-27b", "qwen3.8-27b"} <= route_ids
    assert payload["defaults"]["upstream"] in route_ids
    assert payload["defaults"]["downstream"] in route_ids


def test_inventory_import_normalizes_a_minimal_profile() -> None:
    response = client.post(
        "/api/inventory/import",
        json={
            "name": "Web test laboratory",
            "laboratory": "test",
            "lab_inventory": {
                "pumps": [],
                "reactors": [],
                "tubing": [],
                "light_sources": [],
            },
            "operating_constraints": {},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Web test laboratory"
    assert "validation" in payload


def test_intake_fallback_returns_only_fixed_question_ids() -> None:
    response = client.post(
        "/api/intake/analyze",
        json={
            "raw_protocol": "A substrate was stirred in acetonitrile for 2 h at 40 C.",
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["package"]["raw_protocol"]
    assert {item["question_id"] for item in payload["pending_questions"]} <= set(QUESTION_BANK)


def test_intake_routes_the_selected_upstream_model(monkeypatch) -> None:
    captured = {}

    class FakeAgent:
        def analyze(self, raw_protocol, **kwargs):
            from flora_translate.schemas import DesignInputPackage
            import flora_translate.config as cfg
            from flora_translate.engine import llm_agents

            captured["input_parser"] = cfg.MODEL_INPUT_PARSER
            captured["endpoint"] = llm_agents.get_model_endpoint_overrides().get(
                "/models/Qwen3.6-27B"
            )
            return DesignInputPackage(raw_protocol=raw_protocol)

        def pending_questions(self, package):
            return []

    monkeypatch.setattr("flora_translate.intake_agent.IntakeAgent", FakeAgent)
    monkeypatch.setattr(
        "flora_translate.model_catalog.model_route_availability",
        lambda route: {"available": True, "reason": ""},
    )

    response = client.post(
        "/api/intake/analyze",
        json={
            "raw_protocol": "A batch protocol.",
            "use_llm": True,
            "upstream_model_id": "qwen3.6-27b",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "input_parser": "/models/Qwen3.6-27B",
        "endpoint": "http://10.13.24.169:8000/v1",
    }


def test_plain_language_chemistry_answer_closes_q_chem() -> None:
    initial = client.post(
        "/api/intake/analyze",
        json={
            "raw_protocol": "Compound A was treated with reagent B and afforded compound C.",
            "use_llm": False,
        },
    ).json()
    assert "Q-CHEM-001" in initial["package"]["missing_question_ids"]
    response = client.post(
        "/api/intake/analyze",
        json={
            "raw_protocol": initial["package"]["raw_protocol"],
            "existing_package": initial["package"],
            "answers": [{
                "question_id": "Q-CHEM-001",
                "status": "answered",
                "answer": "Ring closure of precursor A gives cyclic product C.",
            }],
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "Q-CHEM-001" not in payload["package"]["missing_question_ids"]
    assert "Q-CHEM-001" not in {
        item["question_id"] for item in payload["pending_questions"]
    }


def test_design_job_receives_selected_upstream_and_council_models(monkeypatch) -> None:
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_submit(request):
        captured.update(request)
        return SimpleNamespace(
            public=lambda **_: {
                "job_id": "model-route-test",
                "status": "queued",
                "progress": 0,
                "phase": "Queued",
                "messages": [],
            }
        )

    monkeypatch.setattr(JOBS, "submit", fake_submit)
    response = client.post(
        "/api/design/jobs",
        json={
            "batch_input": "A to B.",
            "intake_package": {
                "raw_protocol": "A to B.",
                "objective": "first flow screen",
                "ready_for_design": True,
                "missing_question_ids": [],
            },
            "upstream_model_id": "claude-opus-4-6",
            "downstream_model_id": "gpt-4o",
        },
    )

    assert response.status_code == 202
    runtime = captured["runtime_options"]
    assert runtime["upstream_model"] == "claude-opus-4-6"
    assert runtime["downstream_model"] == "gpt-4o"
    assert runtime["downstream_provider"] == "openai"


def test_design_job_rejects_an_unconfigured_model_before_submission(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        "/api/design/jobs",
        json={
            "batch_input": "A to B.",
            "intake_package": {
                "raw_protocol": "A to B.",
                "objective": "first flow screen",
                "ready_for_design": True,
                "missing_question_ids": [],
            },
            "upstream_model_id": "gpt-4o",
            "downstream_model_id": "gpt-4o",
        },
    )

    assert response.status_code == 400
    assert "OPENAI_API_KEY is not configured" in response.json()["detail"]


def test_saved_run_can_be_reopened_with_its_topology(tmp_path, monkeypatch) -> None:
    run_id = "20260831_120000_webapp"
    directory = tmp_path / "outputs" / "gui_runs" / run_id
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps({"final_design": {"status": "executable"}}),
        encoding="utf-8",
    )
    (directory / "process.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = client.get(f"/api/runs/{run_id}")
    artifact = client.get(f"/api/runs/{run_id}/artifacts/process-svg")

    assert result.status_code == 200
    assert result.json()["final_design"]["status"] == "executable"
    assert artifact.status_code == 200
    assert artifact.headers["content-type"].startswith("image/svg+xml")
