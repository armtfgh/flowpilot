"""Run a reproducible FLORA closed-loop refinement case study.

This script does not call LLM APIs. It starts from a representative FLORA
batch-to-flow design, feeds back three experimental observations, and writes
the resulting campaign package to outputs/closed_loop_case_study/summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate.experiment_loop import (
    ActualConditions,
    ExperimentalOutcomes,
    ExperimentResult,
    refine_from_experiment,
    summarize_campaign,
)


def _initial_design() -> dict:
    return {
        "design_version": 1,
        "confidence": "MEDIUM",
        "explanation": "Case-study starting design for photoredox batch-to-flow translation.",
        "proposal": {
            "residence_time_min": 10.0,
            "flow_rate_mL_min": 0.2,
            "reactor_volume_mL": 2.0,
            "temperature_C": 40.0,
            "concentration_M": 0.1,
            "BPR_bar": 5.0,
            "reactor_type": "photoreactor coil",
            "tubing_material": "FEP",
            "tubing_ID_mm": 0.8,
            "light_setup": "450 nm LED array",
            "wavelength_nm": 450.0,
            "deoxygenation_method": "N2 sparging before pumping",
            "pre_reactor_steps": ["degas stream A with N2"],
            "post_reactor_steps": ["collect under low light"],
            "chemistry_notes": "Photoredox reaction; avoid oxygen inhibition and long over-irradiation.",
            "safety_flags": [],
            "streams": [
                {
                    "stream_label": "A",
                    "pump_role": "substrate + photocatalyst",
                    "contents": ["substrate", "photocatalyst"],
                    "solvent": "MeCN",
                    "concentration_M": 0.1,
                    "flow_rate_mL_min": 0.1,
                    "phase": "liquid",
                    "reasoning": "Main reaction stream.",
                },
                {
                    "stream_label": "B",
                    "pump_role": "base",
                    "contents": ["base"],
                    "solvent": "MeCN",
                    "concentration_M": 0.1,
                    "flow_rate_mL_min": 0.1,
                    "phase": "liquid",
                    "reasoning": "Separate base stream to reduce pre-mixing risk.",
                },
            ],
            "reasoning_per_field": {
                "residence_time_min": "Initial tau from FLORA design-space candidate.",
                "temperature_C": "Mild heating to accelerate photoredox turnover.",
            },
        },
    }


def _make_experiment(result: dict, run_id: str, outcomes: ExperimentalOutcomes) -> ExperimentResult:
    proposal = result["proposal"]
    return ExperimentResult(
        run_id=run_id,
        design_version=result["design_version"],
        actual_conditions=ActualConditions(
            residence_time_min=proposal["residence_time_min"],
            flow_rate_mL_min=proposal["flow_rate_mL_min"],
            reactor_volume_mL=proposal["reactor_volume_mL"],
            temperature_C=proposal["temperature_C"],
            concentration_M=proposal["concentration_M"],
            tubing_ID_mm=proposal["tubing_ID_mm"],
            BPR_bar=proposal["BPR_bar"],
            wavelength_nm=proposal.get("wavelength_nm"),
        ),
        outcomes=outcomes,
    )


def run_case_study() -> dict:
    result = _initial_design()
    cycles = []

    outcomes_by_run = [
        ExperimentalOutcomes(
            yield_pct=28.0,
            conversion_pct=45.0,
            selectivity_pct=92.0,
            pressure_bar=5.5,
            pressure_drift_bar=0.3,
            notes="Clean but incomplete conversion.",
        ),
        ExperimentalOutcomes(
            yield_pct=52.0,
            conversion_pct=88.0,
            selectivity_pct=64.0,
            pressure_bar=9.0,
            pressure_drift_bar=3.5,
            precipitation_observed=True,
            impurity_notes="New late-eluting impurity at extended residence time.",
        ),
        ExperimentalOutcomes(
            yield_pct=84.0,
            conversion_pct=94.0,
            selectivity_pct=91.0,
            pressure_bar=6.0,
            pressure_drift_bar=0.4,
            notes="Stable pressure after dilution and inline filtration.",
        ),
    ]

    for idx, outcomes in enumerate(outcomes_by_run, 1):
        experiment = _make_experiment(result, f"run_{idx:02d}", outcomes)
        closed_loop = refine_from_experiment(
            result,
            experiment,
            campaign_history=[cycle.model_dump() for cycle in cycles],
        )
        cycles.append(closed_loop)
        result = closed_loop.refined_result

    package = {
        "case_study": "photoredox_closed_loop_refinement",
        "summary": summarize_campaign(cycles),
        "final_design": result,
        "cycles": [cycle.model_dump() for cycle in cycles],
    }

    output_dir = Path("outputs/closed_loop_case_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(package, indent=2, default=str))
    package["output_path"] = str(output_path)
    return package


def main() -> int:
    package = run_case_study()
    summary = package["summary"]
    final_proposal = package["final_design"]["proposal"]
    print(json.dumps({
        "output_path": package["output_path"],
        "n_cycles": summary["n_cycles"],
        "best_score": summary["best_score"],
        "final_version": package["final_design"]["design_version"],
        "final_status": package["cycles"][-1]["decision"]["status"],
        "final_next_experiment": package["cycles"][-1]["decision"]["next_experiment"],
        "final_key_parameters": {
            "residence_time_min": final_proposal["residence_time_min"],
            "flow_rate_mL_min": final_proposal["flow_rate_mL_min"],
            "temperature_C": final_proposal["temperature_C"],
            "concentration_M": final_proposal["concentration_M"],
            "BPR_bar": final_proposal["BPR_bar"],
            "tubing_ID_mm": final_proposal["tubing_ID_mm"],
        },
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
