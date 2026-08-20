from ablation_test.scripts.build_evidence_first_validation_package import (
    build_before_after_rows,
    summarize_cells,
)


def _cell(*, fallback: bool, ready: bool, joint: bool = True) -> dict:
    return {
        "scenario_kind": "feasible",
        "engine_fallback": fallback,
        "final_validation_status": "ready" if ready else "unresolved",
        "joint_success": joint,
    }


def test_summary_keeps_primary_endpoint_separate_from_council_path() -> None:
    rows = [
        _cell(fallback=True, ready=True),
        _cell(fallback=False, ready=True),
        _cell(fallback=False, ready=False, joint=False),
    ]

    summary = summarize_cells(rows)

    assert summary["joint_success_n"] == 2
    assert summary["feasible_ready_n"] == 2
    assert summary["feasible_fallback_n"] == 1
    assert summary["feasible_clean_council_n"] == 2


def test_before_after_reports_fallback_reduction_without_rescoring() -> None:
    rows = [
        {
            "run_version": "baseline",
            "scenario_id": scenario,
            "scenario_kind": "feasible",
            "joint_success_n": 3,
            "fallback_n": 3,
            "ready_n": 3,
        }
        for scenario in (
            "suzuki_feasible",
            "suzuki_infeasible",
            "photo_oxidation_feasible",
            "photo_oxidation_infeasible",
            "hydrogenolysis_feasible",
            "hydrogenolysis_infeasible",
            "dinitration_feasible",
            "dinitration_infeasible",
            "multistep_feasible",
            "multistep_infeasible",
        )
    ]
    rows += [
        {
            "run_version": "evidence_first",
            "scenario_id": row["scenario_id"],
            "scenario_kind": row["scenario_kind"],
            "joint_success_n": 3,
            "fallback_n": 0,
            "ready_n": 3,
        }
        for row in rows
    ]

    comparison = build_before_after_rows(rows)

    assert len(comparison) == 10
    assert all(row["fallback_reduction_n"] == 3 for row in comparison)
    assert all(
        row["baseline_joint_success_n"] == row["evidence_first_joint_success_n"]
        for row in comparison
    )
