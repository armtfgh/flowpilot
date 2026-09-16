"""Run THQ KRICT-only benchmark with forced THQ reactor inventory.

Configuration:
- upstream: Claude defaults from flora_translate.config
- downstream council: OpenAI GPT-4o
- inventory: flora_translate/data/lab_inventory_thq.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate import config as cfg  # noqa: E402
from flora_translate.main import translate  # noqa: E402


OUT_DIR = Path("outputs/thq_inventory_claude_gpt4o_20260626")
INVENTORY_PATH = Path("flora_translate/data/lab_inventory_thq.json")

PROMPT = """Use evidence-calibrated closed-loop refinement, not first-pass flow intensification.

Batch reaction:
6-methyl-1,2,3,4-tetrahydroquinoline to 6-methylquinoline in DMSO, 0.50 M, O2, blue LED 450 nm, 40 C, 15 h batch, 75% isolated yield.

Initial flow designs around 34-38 min in-channel gave poor product formation. Use only the experimental feedback below as higher-priority evidence than the original model prediction.

Do not assume any external experimental results. Do not use any KHU results, collaborator data, or hidden benchmark data. Only use the KRICT experiments listed below.

Experimental results from KRICT:

Entry 1, KRICT 2:
T = 40 C
c = 0.50 M
P = 6 bar
substrate flow = 0.0509 mL/min
O2 in-channel = 0.234 mL/min
O2 inlet/STP = 1.2113 mL/min
O2 equiv inlet = 2.12
t inlet = 8.51 min
t in-channel = 37.68 min
reactor volume = 10.74 mL
product = 12%
tubing ID = 0.75 mm

Entry 2, KRICT 6:
T = 40 C
c = 0.50 M
P = 6 bar
substrate flow = 0.0656 mL/min
O2 in-channel = 0.320 mL/min
O2 inlet/STP = 1.6565 mL/min
O2 equiv inlet = 2.25
t inlet = 7.71 min
t in-channel = 34.42 min
reactor volume = 13.27 mL
product = 10%
tubing ID = 1.00 mm

Task:
1. Treat these KRICT measurements as the only experimental evidence.
2. Do not use any KHU data, collaborator data, or hidden benchmark data.
3. The 34-38 min in-channel designs gave only 10-12% product, so do not recommend another design in this same short residence-time window as the main next experiment.
4. Use in-channel residence time as the kinetic calibration basis.
5. Preserve the gas/liquid ratio when scaling flows unless there is a clear reason to change it.
6. Identify the best KRICT result so far.
7. Recommend the next longer residence-time experiment to improve product formation substantially, for example toward 25-30% product first.
8. Optionally provide a rough long-range estimate for approaching the original 75% batch yield, but clearly mark this estimate as highly uncertain because only two low-product KRICT data points are available.
9. Report both t inlet and t in-channel.
10. Clearly state uncertainty and assumptions.
"""


def _setup_logging() -> logging.FileHandler:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"
    log_path.write_text("")
    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _gas_stream(proposal: dict) -> dict:
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase") or "").lower() == "gas" or stream.get("gas_flow_sccm") is not None:
            return stream
    return {}


def _summary(result: dict) -> dict:
    proposal = result.get("proposal") or {}
    calc = result.get("design_calculations") or {}
    gas = _gas_stream(proposal)
    calibration = proposal.get("evidence_calibration") or {}
    return {
        "models": {
            "upstream_input_parser": cfg.MODEL_INPUT_PARSER,
            "upstream_chemistry": cfg.MODEL_CHEMISTRY_AGENT,
            "upstream_translation": cfg.MODEL_TRANSLATION,
            "downstream_provider": cfg.ENGINE_PROVIDER,
            "downstream_model": cfg.ENGINE_MODEL_OPENAI,
        },
        "inventory_path": str(INVENTORY_PATH),
        "residence_time_min": proposal.get("residence_time_min"),
        "residence_time_in_channel_min": proposal.get("residence_time_in_channel_min"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_basis": proposal.get("residence_time_basis"),
        "substrate_flow_mL_min": proposal.get("flow_rate_mL_min"),
        "gas_flow_in_channel_mL_min": gas.get("gas_flow_actual_mL_min"),
        "gas_flow_inlet_STP_mL_min": gas.get("gas_flow_sccm"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "tubing_ID_mm": proposal.get("tubing_ID_mm"),
        "tubing_material": proposal.get("tubing_material"),
        "temperature_C": proposal.get("temperature_C"),
        "concentration_M": proposal.get("concentration_M"),
        "BPR_bar": proposal.get("BPR_bar"),
        "wavelength_nm": proposal.get("wavelength_nm"),
        "light_setup": proposal.get("light_setup"),
        "inventory_selection": proposal.get("inventory_selection"),
        "inventory_constraints": proposal.get("inventory_constraints"),
        "gas_holdup": calc.get("gas_holdup"),
        "pressure_drop_bar": calc.get("pressure_drop_bar"),
        "two_phase_pressure_drop_bar": calc.get("two_phase_pressure_drop_bar"),
        "heat_transfer": proposal.get("heat_transfer_metrics"),
        "calibration": {
            "best_run_id": calibration.get("best_run_id"),
            "best_response_pct": calibration.get("best_response_pct"),
            "best_tau_in_channel_min": calibration.get("best_tau_in_channel_min"),
            "recommended_tau_in_channel_min": calibration.get("recommended_tau_in_channel_min"),
            "target_tau_in_channel_min": calibration.get("target_tau_in_channel_min"),
            "design_ladder": calibration.get("design_ladder"),
        },
        "confidence": result.get("confidence"),
        "safety_flags": proposal.get("safety_flags"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    cfg.ENGINE_PROVIDER = "openai"
    cfg.ENGINE_MODEL_OPENAI = "gpt-4o"

    (OUT_DIR / "prompt.txt").write_text(PROMPT)
    handler = _setup_logging()
    try:
        result = translate(PROMPT, inventory_path=str(INVENTORY_PATH))
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    safe = json.loads(json.dumps(result, default=str))
    (OUT_DIR / "result.json").write_text(json.dumps(safe, indent=2))
    summary = _summary(safe)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
