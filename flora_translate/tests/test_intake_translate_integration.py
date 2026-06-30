import pytest

from flora_translate.intake_agent import historical_text_from_package
from flora_translate.main import _inventory_from_intake_or_path, translate
from flora_translate.schemas import DesignInputPackage


def test_translate_rejects_non_ready_intake_package_before_model_calls():
    package = DesignInputPackage(
        raw_protocol="A to B.",
        missing_question_ids=["Q-OBJ-001"],
        ready_for_design=False,
    )

    with pytest.raises(ValueError, match="Q-OBJ-001"):
        translate("A to B.", intake_package=package)


def test_intake_inventory_payload_converts_to_lab_inventory():
    package = DesignInputPackage(
        raw_protocol="A to B.",
        objective="screen",
        inventory_constraints={
            "pumps": [],
            "tubing": [],
            "BPR_available": [0, 6],
            "light_sources": [
                {
                    "name": "450 nm LED",
                    "wavelength_nm": 450,
                    "power_W": 10,
                    "compatible_reactor": "coil",
                }
            ],
            "reactors": [
                {
                    "type": "coil",
                    "material": "FEP",
                    "volume_mL": 10,
                    "ID_mm": 1.0,
                    "name": "10 mL coil",
                    "system": "test",
                }
            ],
        },
        ready_for_design=True,
    )

    inventory = _inventory_from_intake_or_path(package, "flora_translate/data/lab_inventory_thq.json")

    assert inventory.reactors[0].volume_mL == 10
    assert inventory.reactors[0].ID_mm == 1.0
    assert inventory.BPR_available == [0, 6]


def test_historical_text_from_package_preserves_measured_feedback_only():
    package = DesignInputPackage(
        raw_protocol="A to B.",
        objective="screen",
        historical_data=(
            "Entry 1 KRICT: t in-channel = 37.68 min, product = 12%. "
            "Entry 2 KRICT: t in-channel = 34.42 min, product = 10%."
        ),
        hypotheses=["reaction may need longer residence time"],
        ready_for_design=True,
    )

    text = historical_text_from_package(package)

    assert "KRICT" in text
    assert "product = 12%" in text
    assert "longer residence" not in text
