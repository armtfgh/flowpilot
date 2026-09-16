from pathlib import Path

from ablation_test.src.cases import AblationCase
from ablation_test.src.runner import _production_full
from benchmark.recorder import BenchmarkRecorder


def test_full_variant_calls_shared_production_entry_point(monkeypatch, tmp_path):
    case = AblationCase(
        case_id="shared_path",
        title="Shared production path",
        protocol="Run A and B in flow.",
        suite_id="test",
        category="test",
        source_record_id="held_out.pdf",
        source_record_aliases=("held_out",),
        objective="Produce a conservative screen.",
        hard_constraints=("Use only inventory.",),
        inventory={"schema_version": "flowpilot_lab_inventory_v3.0"},
    )
    recorder = BenchmarkRecorder(tmp_path / "run", {"case_id": case.case_id})
    captured = {}

    def fake_translate(batch_input, inventory_path, intake_package, runtime_options):
        captured.update(
            {
                "batch_input": batch_input,
                "inventory_path": inventory_path,
                "intake_package": intake_package,
                "runtime": runtime_options,
            }
        )
        return {
            "proposal": {"residence_time_min": 1.0},
            "final_design": {"status": "blocked"},
            "process_topology": {"unit_operations": []},
            "recommended_disposition": "BLOCK",
        }

    monkeypatch.setattr(
        "ablation_test.src.runner.production_translate",
        fake_translate,
    )

    result = _production_full(
        case,
        recorder,
        candidate_budget=3,
        temperature=0.0,
        seed=20260813,
    )

    assert captured["batch_input"] == case.design_input_text
    assert Path(captured["inventory_path"]).name == "input_inventory.json"
    assert captured["runtime"].candidate_budget == 3
    assert captured["intake_package"] is None
    assert captured["runtime"].retrieval_mode == "lexical"
    assert captured["runtime"].exclude_record_ids == {
        "held_out.pdf",
        "held_out",
    }
    assert captured["runtime"].objective_override == case.objective
    assert captured["runtime"].hard_constraints == case.hard_constraints
    assert result["production_pipeline_complete"] is True
    assert result["variant"] == "full"
