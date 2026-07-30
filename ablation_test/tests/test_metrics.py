from pathlib import Path

from ablation_test.src.cases import load_cases
from ablation_test.src.metrics import score_run


def test_consistent_liquid_geometry_passes(tmp_path: Path):
    case = load_cases()[0]
    result = {
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 10,
            "flow_rate_mL_min": 0.5,
            "reactor_volume_mL": 5,
            "temperature_C": 80,
            "concentration_M": 0.2,
            "BPR_bar": 5,
            "tubing_ID_mm": 1.0,
            "streams": [],
            "safety_flags": ["Use a back pressure regulator for hot pressurized solvent."],
        },
    }
    metrics = score_run(case, result, tmp_path)
    assert metrics["geometry_consistent_10pct"] is True
    assert metrics["numeric_completeness"] == 1.0


def test_gas_design_requires_both_flow_bases(tmp_path: Path):
    case = next(
        case
        for case in load_cases()
        if case.case_id == "aerobic_oxidation_fmoc_methionine"
    )
    result = {
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 5,
            "flow_rate_mL_min": 0.1,
            "reactor_volume_mL": 1,
            "temperature_C": 25,
            "concentration_M": 0.1,
            "BPR_bar": 5,
            "tubing_ID_mm": 1.0,
            "residence_time_basis": "inlet/STP",
            "streams": [
                {
                    "phase": "gas",
                    "gas_flow_sccm": 0.1,
                    "molar_equiv": 2,
                }
            ],
        },
    }
    metrics = score_run(case, result, tmp_path)
    assert metrics["gas_has_stp_flow"] is True
    assert metrics["gas_has_in_channel_flow"] is False
    assert metrics["gas_bookkeeping_complete"] is False
    assert metrics["deployment_readiness_score_v2"] <= 0.55
    assert "required_gas_bookkeeping_incomplete" in metrics[
        "deployment_gate_reasons_v2"
    ]


def test_schema_invalid_output_is_capped(tmp_path: Path):
    case = load_cases()[0]
    result = {
        "schema_valid": False,
        "proposal": {
            "residence_time_min": 10,
            "flow_rate_mL_min": 0.5,
            "reactor_volume_mL": 5,
            "temperature_C": 80,
            "concentration_M": 0.2,
            "BPR_bar": 5,
            "tubing_ID_mm": 1.0,
            "streams": [],
        },
    }
    metrics = score_run(case, result, tmp_path)
    assert metrics["quality_assurance_score_v2"] > 0.35
    assert metrics["deployment_readiness_score_v2"] == 0.35
    assert "schema_invalid" in metrics["deployment_gate_reasons_v2"]


def test_traceable_review_scores_above_untraced_output(tmp_path: Path):
    case = load_cases()[0]
    proposal = {
        "residence_time_min": 10,
        "flow_rate_mL_min": 0.5,
        "reactor_volume_mL": 5,
        "temperature_C": 80,
        "concentration_M": 0.2,
        "BPR_bar": 5,
        "tubing_ID_mm": 1.0,
        "streams": [],
        "confidence": "moderate",
        "reasoning_per_field": {
            "residence_time_min": "kinetic estimate",
            "flow_rate_mL_min": "volume divided by residence time",
            "reactor_volume_mL": "inventory basis",
            "temperature_C": "batch anchor",
            "BPR_bar": "boiling-point margin",
            "tubing_ID_mm": "inventory basis",
        },
    }
    plain = score_run(
        case,
        {"schema_valid": True, "proposal": proposal},
        tmp_path,
    )

    snapshots = tmp_path / "trace" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "stage3_5_final_audit.json").write_text("{}", encoding="utf-8")
    inventory_path = (
        Path(__file__).resolve().parents[2]
        / "flora_translate"
        / "data"
        / "lab_inventory.json"
    )
    traced = score_run(
        case,
        {
            "schema_valid": True,
            "proposal": proposal,
            "frozen_context": {
                "calculations": {"volume_mL": 5},
                "inventory_path": str(inventory_path),
                "analogies": [
                    {"record_id": f"record-{index}", "summary": "precedent"}
                    for index in range(3)
                ],
            },
            "final_design_candidate": {
                "deliberation_log": ["reviewed"],
                "safety_report": {"status": "reviewed"},
            },
        },
        tmp_path / "trace",
    )
    assert traced["quality_assurance_dimensions_v2"]["decision_assurance"] == 1.0
    assert traced["quality_assurance_dimensions_v2"]["evidence_provenance"] == 1.0
    assert (
        traced["quality_assurance_score_v2"]
        > plain["quality_assurance_score_v2"]
    )


def test_screen_required_design_is_not_deployment_ready(tmp_path: Path):
    case = load_cases()[0]
    result = {
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 10,
            "flow_rate_mL_min": 0.5,
            "reactor_volume_mL": 5,
            "temperature_C": 80,
            "concentration_M": 0.2,
            "BPR_bar": 5,
            "tubing_ID_mm": 1.0,
            "streams": [],
            "confidence": "LOW",
            "safety_flags": ["SCREEN_REQUIRED: no validated candidate"],
        },
    }
    metrics = score_run(case, result, tmp_path)
    assert metrics["deployment_ready_v2"] is False
    assert metrics["deployment_readiness_score_v2"] <= 0.65
    assert "screen_required" in metrics["deployment_gate_reasons_v2"]
