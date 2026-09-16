from ablation_test.scripts.run_qwen_outcome_benchmark import (
    _close,
    _gas_reagent_fraction,
    build_weight_sensitivity_rows,
    pair_feasible_rows,
)


def test_relative_closure_handles_valid_and_invalid_values() -> None:
    assert _close(10.04, 10.0, 0.01)
    assert not _close(11.0, 10.0, 0.01)
    assert not _close(None, 10.0, 0.01)


def test_gas_reagent_fraction_distinguishes_air_from_pure_gas() -> None:
    assert _gas_reagent_fraction("air") == 0.21
    assert _gas_reagent_fraction("hydrogen") == 1.0


def test_pairing_prioritizes_executability_before_quality() -> None:
    rows = []
    for architecture, executable, quality, failures in (
        ("one_shot", False, 99.0, 2),
        ("flowpilot", True, 70.0, 0),
    ):
        rows.append(
            {
                "architecture": architecture,
                "scenario_id": "suzuki_feasible",
                "scenario_kind": "feasible",
                "repeat": 1,
                "executable": executable,
                "quality_score": quality,
                "critical_failure_count": failures,
            }
        )
    for repeat in (2, 3):
        for architecture in ("one_shot", "flowpilot"):
            rows.append(
                {
                    "architecture": architecture,
                    "scenario_id": "suzuki_feasible",
                    "scenario_kind": "feasible",
                    "repeat": repeat,
                    "executable": True,
                    "quality_score": 80.0,
                    "critical_failure_count": 0,
                }
            )

    paired = pair_feasible_rows(rows, tie_points=5.0)

    assert paired[0]["winner"] == "flowpilot"
    assert paired[0]["decision_reason"] == "executable_design_gate"
    assert paired[1]["winner"] == "tie"


def test_weight_sensitivity_cannot_override_executability_gate() -> None:
    rows = []
    for repeat in (1, 2, 3):
        for architecture, executable, failures in (
            ("one_shot", False, 1),
            ("flowpilot", True, 0),
        ):
            rows.append(
                {
                    "architecture": architecture,
                    "scenario_id": "suzuki_feasible",
                    "scenario_kind": "feasible",
                    "repeat": repeat,
                    "executable": executable,
                    "critical_failure_count": failures,
                    **{
                        f"{dimension}_score": (
                            100.0 if architecture == "one_shot" else 60.0
                        )
                        for dimension in (
                            "inventory_compliance",
                            "numerical_closure",
                            "chemistry_fidelity",
                            "process_completeness",
                        )
                    },
                }
            )

    sensitivity = build_weight_sensitivity_rows(rows, tie_points=5.0)

    assert all(row["flowpilot_wins"] == 3 for row in sensitivity)
    assert all(row["one_shot_wins"] == 0 for row in sensitivity)
