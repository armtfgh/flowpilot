import json
from pathlib import Path

from flora_translate.gui_autosave import autosave_gui_result


def test_autosave_gui_result_writes_jsons_and_diagrams(tmp_path):
    svg = tmp_path / "source.svg"
    png = tmp_path / "source.png"
    svg.write_text("<svg></svg>")
    png.write_bytes(b"png")

    result = {
        "confidence": "MEDIUM",
        "recommended_disposition": "BLOCK",
        "disposition_rationale": "Inventory conflict.",
        "design_disposition": {
            "hard_failures": [
                {
                    "finding_id": "INVENTORY-TEST",
                    "message": "No compatible reactor.",
                }
            ]
        },
        "svg_path": str(svg),
        "png_path": str(png),
        "chemistry_plan": {"reaction_name": "test reaction"},
        "proposal": {
            "residence_time_min": 12.5,
            "flow_rate_mL_min": 0.1,
            "reactor_volume_mL": 1.25,
            "tubing_ID_mm": 1.0,
            "temperature_C": 40,
            "concentration_M": 0.5,
            "BPR_bar": 6,
            "streams": [{"phase": "gas", "gas_flow_sccm": 1.2}],
        },
    }
    intake = {"schema_version": "flowpilot_intake_v1.0", "ready_for_design": True}

    run_dir = autosave_gui_result(
        result,
        intake_package=intake,
        source="test_gui",
        user_input="A to B",
        base_dir=tmp_path / "gui_runs",
    )

    assert (run_dir / "result.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "intake_package.json").exists()
    assert (run_dir / "input.txt").read_text() == "A to B"
    assert (run_dir / "translate_process.svg").read_text() == "<svg></svg>"
    assert (run_dir / "translate_process.png").read_bytes() == b"png"

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["source"] == "test_gui"
    assert summary["residence_time_min"] == 12.5
    assert summary["gas_flow_sccm"] == 1.2
    assert summary["recommended_disposition"] == "BLOCK"
    assert summary["hard_failures"][0]["finding_id"] == "INVENTORY-TEST"


def test_autosave_gui_result_creates_unique_run_dirs(tmp_path):
    result = {"proposal": {}}

    first = autosave_gui_result(result, source="repeat", base_dir=tmp_path)
    second = autosave_gui_result(result, source="repeat", base_dir=tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()
