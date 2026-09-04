"""THQ closed-loop revision using KHU/KRICT experimental feedback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate.experiment_loop import (  # noqa: E402
    ActualConditions,
    ExperimentalOutcomes,
    ExperimentResult,
    calibrate_experimental_campaign,
    refine_from_experimental_campaign,
    refine_from_experiment,
)


BASELINE_PATH = Path("outputs/thq_variations/v1_baseline_result.json")
OUTPUT_DIR = Path("outputs/thq_experimental_revision")


OBSERVATIONS = [
    {
        "entry": 1,
        "note": "KHU Vapourtec",
        "temperature_C": 60.0,
        "concentration_M": 0.25,
        "pressure_bar": 3.0,
        "substrate_flow_mL_min": 0.014,
        "o2_in_channel_mL_min": 0.059,
        "o2_inlet_stp_mL_min": 0.1530,
        "o2_equiv": 1.95,
        "t_inlet_min": 59.91,
        "t_in_channel_min": 137.08,
        "reactor_volume_mL": 10.00,
        "starting_material_pct": None,
        "product_pct": 52.0,
        "isolated_yield_pct": 47.0,
        "tubing_ID_mm": 1.016,
    },
    {
        "entry": 2,
        "note": "KHU manual",
        "temperature_C": 40.0,
        "concentration_M": 0.25,
        "pressure_bar": 3.0,
        "substrate_flow_mL_min": 0.014,
        "o2_in_channel_mL_min": 0.059,
        "o2_inlet_stp_mL_min": 0.1530,
        "o2_equiv": 1.95,
        "t_inlet_min": 59.91,
        "t_in_channel_min": 137.08,
        "reactor_volume_mL": 10.00,
        "starting_material_pct": None,
        "product_pct": 52.0,
        "isolated_yield_pct": None,
        "tubing_ID_mm": 1.016,
    },
    {
        "entry": 3,
        "note": "KRICT 2",
        "temperature_C": 40.0,
        "concentration_M": 0.50,
        "pressure_bar": 6.0,
        "substrate_flow_mL_min": 0.0509,
        "o2_in_channel_mL_min": 0.234,
        "o2_inlet_stp_mL_min": 1.2113,
        "o2_equiv": 2.12,
        "t_inlet_min": 8.51,
        "t_in_channel_min": 37.68,
        "reactor_volume_mL": 10.74,
        "starting_material_pct": 83.0,
        "product_pct": 12.0,
        "isolated_yield_pct": None,
        "tubing_ID_mm": 0.75,
    },
    {
        "entry": 4,
        "note": "KRICT 6",
        "temperature_C": 40.0,
        "concentration_M": 0.50,
        "pressure_bar": 6.0,
        "substrate_flow_mL_min": 0.0656,
        "o2_in_channel_mL_min": 0.320,
        "o2_inlet_stp_mL_min": 1.6565,
        "o2_equiv": 2.25,
        "t_inlet_min": 7.71,
        "t_in_channel_min": 34.42,
        "reactor_volume_mL": 13.27,
        "starting_material_pct": 85.0,
        "product_pct": 10.0,
        "isolated_yield_pct": None,
        "tubing_ID_mm": 1.00,
    },
]


def _experiment_from_observation(obs: dict, design_version: int) -> ExperimentResult:
    return ExperimentResult(
        run_id=f"entry_{obs['entry']:02d}_{obs['note'].lower().replace(' ', '_')}",
        design_version=design_version,
        actual_conditions=ActualConditions(
            residence_time_inlet_min=obs["t_inlet_min"],
            residence_time_in_channel_min=obs["t_in_channel_min"],
            residence_time_basis="in_channel",
            flow_rate_mL_min=obs["substrate_flow_mL_min"],
            substrate_flow_mL_min=obs["substrate_flow_mL_min"],
            gas_flow_in_channel_mL_min=obs["o2_in_channel_mL_min"],
            gas_flow_stp_mL_min=obs["o2_inlet_stp_mL_min"],
            gas_equiv_inlet=obs["o2_equiv"],
            temperature_C=obs["temperature_C"],
            concentration_M=obs["concentration_M"],
            tubing_ID_mm=obs["tubing_ID_mm"],
            reactor_volume_mL=obs["reactor_volume_mL"],
            BPR_bar=obs["pressure_bar"],
        ),
        outcomes=ExperimentalOutcomes(
            product_pct=obs["product_pct"],
            yield_pct=obs["isolated_yield_pct"] or obs["product_pct"],
            starting_material_pct=obs["starting_material_pct"],
            pressure_bar=obs["pressure_bar"],
            notes=obs["note"],
        ),
        free_text_observations=obs["note"],
    )


def run_revision() -> dict:
    baseline = json.loads(BASELINE_PATH.read_text())
    experiments = [
        _experiment_from_observation(obs, design_version=1)
        for obs in OBSERVATIONS
    ]
    failed_krict = experiments[-1]
    single_run_closed_loop = refine_from_experiment(
        baseline,
        failed_krict,
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
        target_selectivity_pct=85.0,
    )
    campaign_closed_loop = refine_from_experimental_campaign(
        baseline,
        experiments,
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
        target_selectivity_pct=85.0,
    )
    calibration = calibrate_experimental_campaign(
        experiments,
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
    )

    package = {
        "baseline_path": str(BASELINE_PATH),
        "observations": OBSERVATIONS,
        "single_run_closed_loop_from_krict6": single_run_closed_loop.model_dump(),
        "campaign_closed_loop": campaign_closed_loop.model_dump(),
        "campaign_calibration": calibration.model_dump(),
        "evidence_based_revised_design": campaign_closed_loop.refined_result,
        "recommendation": {
            "primary_next_design": calibration.recommended_conditions,
            "design_ladder": calibration.design_ladder,
            "calibration_notes": calibration.notes,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(package, indent=2, default=str))
    package["output_path"] = str(output_path)
    return package


def main() -> int:
    package = run_revision()
    print(json.dumps({
        "output_path": package["output_path"],
        "primary_next_design": package["recommendation"]["primary_next_design"],
        "design_ladder": package["recommendation"]["design_ladder"],
        "campaign_failure_modes": package["campaign_closed_loop"]["decision"]["failure_modes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
