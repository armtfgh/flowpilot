"""Run THQ benchmark through the standardized FlowPilot intake package.

Configuration:
- upstream: Claude defaults from flora_translate.config
- downstream council: OpenAI GPT-4o
- intake: KRICT-only feedback, THQ inventory, longer-residence hypothesis
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
from flora_translate.intake_agent import IntakeAgent, intake_context_block  # noqa: E402
from flora_translate.main import translate  # noqa: E402
from flora_translate.schemas import IntakeAnswer  # noqa: E402


OUT_DIR = Path("outputs/thq_intake_claude_gpt4o_20260629")
INVENTORY_PATH = Path("flora_translate/data/lab_inventory_thq.json")

PROTOCOL = """A flame-dried screw-cap reaction tube (13 x 100 mm, 8 mL) equipped with a magnetic stir bar was charged with 6-methyl-1,2,3,4-tetrahydroquinoline (1a, 0.20 mmol, 1.0 equiv, 29.4 mg, MW = 147.22 g/mol) and dimethyl sulfoxide (DMSO, 0.40 mL, 0.50 M) as solvent. No photocatalyst, transition metal, or any additive was used. Oxygen gas (O2, 1 atm, 0.4 mmol, 2.0 equiv, 12.8 mg, MW = 32 g/mol) was bubbled through the reaction mixture for 10 minutes using an oxygen-filled balloon to saturate the solution prior to irradiation. The sealed reaction tube was placed between two MR16 blue LED lamps (lambda = 450 +/- 15 nm, 5 W x 2) and irradiated from both sides simultaneously with no external cooling applied, allowing the reaction temperature to rise naturally to 40 C driven by the heat output of the LEDs. The reaction mixture was stirred continuously at 40 C under blue LED irradiation for 15 hours. This protocol afforded 6-methylquinoline (2a, MW = 143.19 g/mol) in 75% isolated yield (21.5 mg)."""

KRICT_HISTORY = """Use only these KRICT experimental results as measured feedback. Do not use collaborator or hidden benchmark data.

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

Entry 3, KRICT 2-1:
T = 40 C
c = 0.50 M
P = 6 bar
substrate flow = 0.01047 mL/min
O2 in-channel = 0.0489 mL/min
O2 inlet/STP = 0.2492 mL/min
O2 equiv inlet = 2.0
t inlet = 38.6 min
t in-channel = 168.43 min
reactor volume = 10.00 mL
product = 27%
tubing ID = 1.016 mm
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


def _build_intake_package():
    inventory = json.loads(INVENTORY_PATH.read_text())
    answers = [
        IntakeAnswer(
            question_id="Q-OBJ-001",
            answer=(
                "Use evidence-calibrated closed-loop refinement for the next "
                "longer KRICT-only screening design. Avoid another main design "
                "near the failed short screen. Use inlet/STP apparent residence "
                "time as the primary design and calibration basis because O2 "
                "equivalents are controlled by the MFC/STP inlet flow. Report "
                "both inlet and pressure-corrected in-channel residence times."
            ),
        ),
        IntakeAnswer(question_id="Q-HIST-001", answer=KRICT_HISTORY),
        IntakeAnswer(question_id="Q-INV-001", answer=inventory),
        IntakeAnswer(
            question_id="Q-CONSTR-001",
            answer=(
                "Use only the provided inventory. Operate within 0-8 bar, "
                "0.1-0.5 M, and the temperature limits of the selected reactor. "
                "Preserve KRICT O2 inlet/STP-to-liquid ratio when scaling flows "
                "unless there is a clear reason to change it."
            ),
        ),
        IntakeAnswer(
            question_id="Q-HYP-001",
            answer=(
                "The KRICT short screens had only 7.7-8.5 min inlet/STP apparent "
                "residence time and low product. The next KRICT run at 38.6 min "
                "inlet/STP apparent residence time improved product to 27%, so "
                "the next design should not shorten residence time below that "
                "measured anchor unless there is direct evidence for overreaction. "
                "Oxygen delivery and photon flux should be treated as possible "
                "secondary limitations."
            ),
        ),
        IntakeAnswer(
            question_id="Q-PREF-001",
            answer=(
                "Return one inventory-constrained next experiment plus uncertainty, "
                "calibration basis, and a rough long-range estimate only if clearly "
                "marked uncertain."
            ),
        ),
    ]
    package = IntakeAgent().analyze(PROTOCOL, answers=answers, use_llm=False)
    if not package.ready_for_design:
        raise RuntimeError(f"Intake package is not ready: {package.missing_question_ids}")
    serialized = json.dumps(package.model_dump(), default=str)
    if "KHU" in serialized:
        raise RuntimeError("KHU data leaked into standardized intake package")
    return package


def _gas_stream(proposal: dict) -> dict:
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase") or "").lower() == "gas" or stream.get("gas_flow_sccm") is not None:
            return stream
    return {}


def _summary(result: dict, package: dict) -> dict:
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
        "intake_ready": package.get("ready_for_design"),
        "missing_question_ids": package.get("missing_question_ids"),
        "inventory_from_intake": True,
        "contains_khu_in_intake": "KHU" in json.dumps(package, default=str),
        "residence_time_min": proposal.get("residence_time_min"),
        "residence_time_in_channel_min": proposal.get("residence_time_in_channel_min"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_basis": proposal.get("residence_time_basis"),
        "substrate_flow_mL_min": proposal.get("flow_rate_mL_min"),
        "gas_flow_in_channel_mL_min": gas.get("gas_flow_actual_mL_min"),
        "gas_flow_inlet_STP_mL_min": gas.get("gas_flow_sccm"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "tubing_ID_mm": proposal.get("tubing_ID_mm"),
        "temperature_C": proposal.get("temperature_C"),
        "concentration_M": proposal.get("concentration_M"),
        "BPR_bar": proposal.get("BPR_bar"),
        "wavelength_nm": proposal.get("wavelength_nm"),
        "light_setup": proposal.get("light_setup"),
        "inventory_selection": proposal.get("inventory_selection"),
        "inventory_constraints": proposal.get("inventory_constraints"),
        "gas_holdup": calc.get("gas_holdup"),
        "pressure_drop_bar": calc.get("pressure_drop_bar"),
        "calibration": {
            "primary_residence_time_basis": calibration.get("primary_residence_time_basis"),
            "best_run_id": calibration.get("best_run_id"),
            "best_response_pct": calibration.get("best_response_pct"),
            "best_tau_min": calibration.get("best_tau_min"),
            "recommended_tau_min": calibration.get("recommended_tau_min"),
            "target_tau_min": calibration.get("target_tau_min"),
            "best_tau_inlet_min": calibration.get("best_tau_inlet_min"),
            "best_tau_in_channel_min": calibration.get("best_tau_in_channel_min"),
            "recommended_tau_inlet_min": calibration.get("recommended_tau_inlet_min"),
            "recommended_tau_in_channel_min": calibration.get("recommended_tau_in_channel_min"),
            "target_tau_inlet_min": calibration.get("target_tau_inlet_min"),
            "target_tau_in_channel_min": calibration.get("target_tau_in_channel_min"),
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

    package = _build_intake_package()
    (OUT_DIR / "prompt.txt").write_text(PROTOCOL)
    (OUT_DIR / "intake_package.json").write_text(
        json.dumps(package.model_dump(), indent=2, default=str)
    )
    (OUT_DIR / "intake_context.txt").write_text(intake_context_block(package))

    handler = _setup_logging()
    try:
        result = translate(
            PROTOCOL,
            inventory_path=str(INVENTORY_PATH),
            intake_package=package,
        )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    safe = json.loads(json.dumps(result, default=str))
    result_text = json.dumps(safe, default=str)
    if "KHU" in result_text:
        raise RuntimeError("KHU string appeared in result; benchmark must remain KRICT-only")
    (OUT_DIR / "result.json").write_text(json.dumps(safe, indent=2))
    summary = _summary(safe, package.model_dump())
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
