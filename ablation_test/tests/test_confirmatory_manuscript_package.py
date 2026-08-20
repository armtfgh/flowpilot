from ablation_test.scripts.build_confirmatory_manuscript_package import (
    _failure_type,
    build_input_integrity_summary,
)


def test_failure_type_keeps_overlapping_failure_categories() -> None:
    row = {
        "disposition_correct": False,
        "expected_disposition": "SCREEN",
        "critical_engineering_pass": False,
    }

    assert _failure_type(row) == (
        "false_block_feasible|critical_engineering_failure"
    )


def test_correct_block_with_compliant_output_is_not_a_failure() -> None:
    row = {
        "disposition_correct": True,
        "expected_disposition": "BLOCK",
        "critical_engineering_pass": True,
    }

    assert _failure_type(row) == "none"


def test_input_integrity_summary_exposes_condition_mismatch() -> None:
    audit = {
        "passed_for_recorded_cells": False,
        "represented_conditions": ["full", "one_shot"],
        "missing_conditions": ["control"],
        "records": [
            {
                "public_artifact_identical": True,
                "inventory_artifact_identical": True,
                "input_public_sha256": "public-a",
                "input_inventory_sha256": "inventory-a",
            },
            {
                "public_artifact_identical": False,
                "inventory_artifact_identical": True,
                "input_public_sha256": "public-b",
                "input_inventory_sha256": "inventory-a",
            },
        ],
    }

    summary = build_input_integrity_summary(audit)

    assert summary["passed"] is False
    assert summary["all_public_inputs_identical"] is False
    assert summary["all_inventory_inputs_identical"] is True
    assert summary["unique_public_hashes"] == 2
    assert summary["unique_inventory_hashes"] == 1
