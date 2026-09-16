from pathlib import Path

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.error_audit import audit_result, load_expectations


ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = load_expectations(
    ROOT / "ablation_test" / "benchmarks" / "newgen_error_expectations_v1.json"
)


def _case(name: str):
    path = (
        ROOT
        / "ablation_test"
        / "benchmarks"
        / f"newgen_benchmark_v1_pilot_{name}"
        / "case.json"
    )
    return load_cases_from_path(path)[0]


def _status(audit: dict, criterion_id: str) -> str:
    return next(
        row["status"]
        for row in audit["criteria"]
        if row["criterion_id"] == criterion_id
    )


def test_gas_arithmetic_errors_are_counted_deterministically():
    case = _case("hydrogenolysis")
    result = {
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 30.0,
            "flow_rate_mL_min": 0.1,
            "temperature_C": 60.0,
            "concentration_M": 0.126,
            "BPR_bar": 21.0,
            "reactor_volume_mL": 3.0,
            "tubing_ID_mm": 4.35,
            "residence_time_inlet_min": 30.0,
            "residence_time_in_channel_min": 12.0,
            "reactor_type": "hydrogenolysis in Pd(OH)2/Al2O3 packed bed",
            "streams": [
                {
                    "stream_label": "A",
                    "phase": "liquid",
                    "contents": ["DMAOL", "methanol"],
                    "concentration_M": 0.126,
                    "flow_rate_mL_min": 0.1,
                },
                {
                    "stream_label": "H2",
                    "phase": "gas",
                    "contents": ["hydrogen"],
                    "gas_flow_sccm": 1.0,
                    "gas_flow_actual_mL_min": 0.002,
                    "molar_equiv": 2.0,
                },
            ],
            "pre_reactor_steps": [
                "N2 purge and leak test; pump through gas-liquid mixer, water bath, and packed bed at 21 bar absolute."
            ],
            "post_reactor_steps": [
                "Pass BPR to separator, approved exhaust, and grounded collector."
            ],
            "reasoning_per_field": {"flow": "calculation", "gas": "calculation", "pressure": "fact"},
            "safety_flags": ["hydrogen pressure vent"],
        },
    }

    audit = audit_result(case, result, EXPECTATIONS[case.case_id])

    assert _status(audit, "NG-15") == "FAIL"
    assert _status(audit, "NG-16") == "FAIL"
    assert _status(audit, "NG-17") == "FAIL"


def test_multistage_total_residence_time_uses_sum_of_stage_times():
    case = _case("multistep")
    result = {
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 15.04,
            "flow_rate_mL_min": 1.5,
            "temperature_C": 80.0,
            "concentration_M": 0.667,
            "BPR_bar": 0.0,
            "reactor_volume_mL": 15.04,
            "tubing_ID_mm": 0.5,
            "reactor_type": "two-stage oxidative amidation",
            "streams": [
                {"stream_label": "A", "phase": "liquid", "contents": ["benzyl alcohol", "dioxane"], "concentration_M": 0.667, "flow_rate_mL_min": 0.5, "molar_equiv": 1.0},
                {"stream_label": "B", "phase": "liquid", "contents": ["H2O2", "NaBr", "H2SO4"], "concentration_M": 1.334, "flow_rate_mL_min": 0.5, "molar_equiv": 2.0},
                {"stream_label": "C", "phase": "liquid", "contents": ["morpholine", "TBHP"], "solvent": "dioxane", "concentration_M": 0.667, "flow_rate_mL_min": 0.5, "molar_equiv": 1.0},
            ],
            "stage_parameters": [
                {"reactor_id": "reactor_ms_196", "mixer_id": "mixer_ms_1", "reactor_volume_mL": 1.96, "temperature_C": 70.0, "cumulative_flow_mL_min": 1.0, "residence_time_min": 1.96},
                {"reactor_id": "reactor_ms_1308", "mixer_id": "mixer_ms_2", "reactor_volume_mL": 13.08, "temperature_C": 80.0, "cumulative_flow_mL_min": 1.5, "residence_time_min": 8.72},
            ],
            "pre_reactor_steps": ["Pumps A and B to mixer_ms_1 and reactor_ms_196; add C at mixer_ms_2 before reactor_ms_1308."],
            "post_reactor_steps": ["Cool and collect; flush until peroxide test is negative."],
            "reasoning_per_field": {"stage1": "V/Q", "stage2": "V/Q", "total": "sum"},
            "safety_flags": ["peroxide acid temperature control and flush"],
        },
    }

    audit = audit_result(case, result, EXPECTATIONS[case.case_id])

    assert _status(audit, "NG-13") == "FAIL"
    row = next(row for row in audit["criteria"] if row["criterion_id"] == "NG-13")
    assert "sum stages=10.68" in row["observed"]


def test_gas_checks_are_not_applicable_to_gas_free_case():
    case = _case("cuaac")
    result = {
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 3.0,
            "flow_rate_mL_min": 0.1,
            "temperature_C": 150.0,
            "concentration_M": 0.25,
            "BPR_bar": 20.0,
            "reactor_volume_mL": 0.3,
            "tubing_ID_mm": 4.0,
            "reactor_type": "Cu/C packed-bed CuAAC to triazole",
            "streams": [{"phase": "liquid", "contents": ["benzyl azide", "phenylacetylene", "acetone", "Cu/C", "1.1 equivalent"], "concentration_M": 0.25, "flow_rate_mL_min": 0.1}],
            "pre_reactor_steps": ["Pump through heated packed bed at BPR 20 bar."],
            "post_reactor_steps": ["Vented collection with azide pressure controls."],
            "reasoning_per_field": {"flow": "V/tau", "temperature": "protocol", "pressure": "inventory"},
        },
    }

    audit = audit_result(case, result, EXPECTATIONS[case.case_id])

    assert _status(audit, "NG-15") == "NOT_APPLICABLE"
    assert _status(audit, "NG-16") == "NOT_APPLICABLE"
    assert _status(audit, "NG-17") == "NOT_APPLICABLE"
