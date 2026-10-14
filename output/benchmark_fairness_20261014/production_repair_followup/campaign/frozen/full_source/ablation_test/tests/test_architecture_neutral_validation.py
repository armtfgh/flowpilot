from ablation_test.src.architecture_neutral_validation import assess_candidate
from ablation_test.src.cases import AblationCase


def _case() -> AblationCase:
    return AblationCase(
        case_id="neutral",
        title="Neutral validator",
        protocol="Thermal reaction",
        suite_id="test",
        category="thermal",
        inventory={
            "pumps": [
                {
                    "equipment_id": "pump_1",
                    "quantity": 1,
                    "service_status": "available",
                    "min_flow_rate_mL_min": 0.01,
                    "max_flow_rate_mL_min": 1.0,
                }
            ],
            "reactors": [
                {
                    "equipment_id": "reactor_1",
                    "service_status": "available",
                    "volume_mL": 10.0,
                    "min_temperature_C": 20,
                    "max_temperature_C": 100,
                }
            ],
            "BPR_available": [0, 3],
        },
    )


def _result(variant: str) -> dict:
    return {
        "variant": variant,
        "schema_valid": True,
        "proposal": {
            "residence_time_min": 100,
            "flow_rate_mL_min": 0.1,
            "temperature_C": 60,
            "concentration_M": 0.1,
            "BPR_bar": 0,
            "reactor_volume_mL": 10,
            "streams": [
                {
                    "stream_label": "A",
                    "phase": "liquid",
                    "flow_rate_mL_min": 0.1,
                    "concentration_M": 0.1,
                }
            ],
        },
    }


def test_identical_published_designs_receive_identical_assessment():
    one_shot = assess_candidate(_case(), _result("general_one_shot"))
    flowpilot = assess_candidate(_case(), _result("full"))

    assert one_shot == flowpilot
    assert one_shot["status"] == "executable"


def test_extra_liquid_feed_fails_pump_quantity_for_any_architecture():
    result = _result("general_one_shot")
    result["proposal"]["streams"] = [
        {"stream_label": "A", "phase": "liquid", "flow_rate_mL_min": 0.05},
        {"stream_label": "B", "phase": "liquid", "flow_rate_mL_min": 0.05},
    ]

    assessment = assess_candidate(_case(), result)

    assert assessment["status"] == "not_executable"
    assert "AN-05-LIQUID-PUMP-FEASIBILITY" in assessment["failed_check_ids"]
