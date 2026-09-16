"""Run case_study1 with inline degassing explicitly unavailable.

The chemistry still requires oxygen exclusion in Stage 1. This run therefore
allows offline preparation of pre-degassed feeds in sealed inert reservoirs,
but forbids an inline membrane/vacuum degasser in the process design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flora_translate.config as cfg
from benchmark.run_model_matrix_benchmark import council_bundle
from benchmark.run_multiphase_upgrade_validation import PROMPT_PHOTOREDOX_AEROBIC
from flora_translate.engine.llm_agents import (
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)
from flora_translate.intake_agent import get_question
from flora_translate.main import translate
from flora_translate.schemas import DesignInputPackage, IntakeAnswer, LabInventory


QUESTION_IDS = (
    "Q-BATCH-001",
    "Q-OBJ-001",
    "Q-HIST-001",
    "Q-INV-001",
    "Q-CONSTR-001",
    "Q-HYP-001",
    "Q-PREF-001",
)

OPERATING_LIMITS = {
    "inline_degasser_available": False,
    "forbidden_equipment": [
        "inline membrane degasser",
        "inline vacuum degasser",
        "dedicated inline degassing unit",
    ],
    "required_stage_1_atmosphere": "Strictly oxygen-free/inert.",
    "allowed_oxygen_exclusion_method": (
        "Prepare the EtOH/pH 9 buffer and Stage 1 feed solutions offline using "
        "pre-degassed solvent or argon sparging, transfer them to sealed inerted "
        "feed reservoirs, and maintain an argon headspace during pumping."
    ),
    "stage_2_requirement": (
        "Introduce air only after Stage 1 through the available gas-feed hardware."
    ),
    "design_rule": (
        "Do not include a degasser unit operation in the final process topology."
    ),
}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _build_package(inventory: dict[str, Any]) -> DesignInputPackage:
    objective = (
        "Reproduce the same two-stage photoredox Giese addition followed by "
        "aerobic sulfoxidation, while using no inline degasser and preserving "
        "the strict Stage 1 oxygen-exclusion requirement."
    )
    hypothesis = (
        "Offline pre-degassing of the solvent/feed followed by pumping from "
        "sealed argon-blanketed reservoirs can replace an inline degasser "
        "without changing the two-stage reaction sequence."
    )
    return DesignInputPackage(
        raw_protocol=PROMPT_PHOTOREDOX_AEROBIC,
        objective=objective,
        historical_data=None,
        inventory_constraints=inventory,
        hypotheses=[hypothesis],
        operating_limits=OPERATING_LIMITS,
        output_preferences=(
            "Return one best inventory-constrained design with separate Stage 1 "
            "and Stage 2 residence times, stream flow rates, and full topology."
        ),
        question_log=[get_question(question_id) for question_id in QUESTION_IDS],
        answers=[
            IntakeAnswer(
                question_id="Q-BATCH-001",
                answer=PROMPT_PHOTOREDOX_AEROBIC,
            ),
            IntakeAnswer(
                question_id="Q-OBJ-001",
                answer=objective,
            ),
            IntakeAnswer(
                question_id="Q-HIST-001",
                status="unavailable",
            ),
            IntakeAnswer(
                question_id="Q-INV-001",
                answer=inventory,
            ),
            IntakeAnswer(
                question_id="Q-CONSTR-001",
                answer=OPERATING_LIMITS,
            ),
            IntakeAnswer(
                question_id="Q-HYP-001",
                answer=hypothesis,
            ),
            IntakeAnswer(
                question_id="Q-PREF-001",
                answer=(
                    "One best design with stage-specific calculations and "
                    "an explicit no-inline-degasser statement."
                ),
            ),
        ],
        missing_question_ids=[],
        ready_for_design=True,
    )


def _case_study_inventory() -> dict[str, Any]:
    """Return the process hardware shown in case_study1.png, minus degassing."""

    base = LabInventory.from_json(str(cfg.LAB_INVENTORY_PATH)).model_dump()
    return {
        "pumps": base["pumps"],
        "tubing": [
            {
                "material": "FEP",
                "ID_mm": 1.0,
                "max_pressure_bar": 25,
                "max_temperature_C": 180,
                "transparent": True,
            }
        ],
        # The original figure used 2 bar. A 3 bar setting is also included
        # because FlowPilot's deterministic gas-liquid safety floor is 3 bar.
        "BPR_available": [2.0, 3.0],
        "light_sources": [
            {
                "name": "Case-study blue LED module",
                "wavelength_nm": 452,
                "power_W": 10,
                "compatible_reactor": "coil",
            }
        ],
        "gas_hardware": [
            {
                "name": "Air mass-flow controller",
                "type": "MFC",
                "gas": "air",
                "min_flow_sccm": 0.01,
                "max_flow_sccm": 10,
                "max_pressure_bar": 10,
                "service_status": "available",
                "notes": "Air is introduced only between Stage 1 and Stage 2.",
            },
            {
                "name": "Air-liquid T-mixer",
                "type": "gas-liquid mixer",
                "gas": "air",
                "max_pressure_bar": 10,
                "service_status": "available",
                "notes": "Interstage air injection upstream of Stage 2.",
            },
        ],
        "reactors": [
            {
                "name": "Case-study two-stage photoreactor train",
                "system": "Manual photoflow setup",
                "type": "coil",
                "material": "FEP",
                "volume_mL": 8.5,
                "ID_mm": 1.0,
                "light_source": "Two 10 W blue LED modules",
                "wavelength_nm": 452,
                "irradiation": "One illuminated coil per reaction stage",
                "allowed_temperatures_C": [25],
                "min_concentration_M": 0.1,
                "max_concentration_M": 0.1,
                "min_pressure_bar": 0,
                "max_pressure_bar": 5,
                "configuration": "two-stage serial",
                "component_volumes_mL": [4.25, 4.25],
                "notes": (
                    "Interstage air-addition port is available. No inline "
                    "degasser or membrane deoxygenation unit is installed."
                ),
            }
        ],
    }


def _constraint_audit(result: dict[str, Any]) -> dict[str, Any]:
    proposal = result.get("proposal") or {}
    chemistry = result.get("chemistry_plan") or {}
    topology = result.get("process_topology") or {}
    operations = topology.get("unit_operations") or []
    streams = proposal.get("streams") or []

    degasser_operations = [
        operation
        for operation in operations
        if str(operation.get("op_type") or "").lower()
        in {"deoxygenation_unit", "degas", "degasser"}
    ]
    preparation_text = " ".join(
        [
            str(proposal.get("deoxygenation_method") or ""),
            *[str(step) for step in proposal.get("pre_reactor_steps") or []],
        ]
    ).lower()
    offline_deoxygenation_present = any(
        token in preparation_text
        for token in (
            "pre-degass",
            "predegass",
            "argon sparg",
            "ar sparg",
            "inert reservoir",
            "argon-blanket",
            "argon blanket",
        )
    )
    stage_2_air_streams = [
        stream
        for stream in streams
        if "air" in " ".join(str(item) for item in stream.get("contents") or []).lower()
        and (
            str(stream.get("phase") or "").lower() == "gas"
            or stream.get("gas_flow_sccm") is not None
        )
    ]
    oxygen_exclusion_recognized = bool(
        chemistry.get("deoxygenation_required")
        or chemistry.get("oxygen_sensitive")
        or "oxygen" in str(chemistry.get("deoxygenation_reasoning") or "").lower()
    )
    checks = {
        "no_inline_degasser_unit": not degasser_operations,
        "offline_feed_deoxygenation_present": offline_deoxygenation_present,
        "stage_1_oxygen_exclusion_recognized": oxygen_exclusion_recognized,
        "stage_2_air_feed_present": bool(stage_2_air_streams),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "degasser_operations": degasser_operations,
        "deoxygenation_method": proposal.get("deoxygenation_method"),
        "pre_reactor_steps": proposal.get("pre_reactor_steps") or [],
        "stage_2_air_streams": stage_2_air_streams,
    }


def _summary(result: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    proposal = result.get("proposal") or {}
    topology = result.get("process_topology") or {}
    return {
        "status": "completed",
        "constraint_audit_passed": audit["passed"],
        "recommended_disposition": result.get("recommended_disposition"),
        "confidence": result.get("confidence"),
        "upstream_models": {
            "input_parser": cfg.MODEL_INPUT_PARSER,
            "chemistry_agent": cfg.MODEL_CHEMISTRY_AGENT,
            "translation": cfg.MODEL_TRANSLATION,
            "output_formatter": cfg.MODEL_OUTPUT_FORMATTER,
        },
        "downstream": {
            "provider": cfg.ENGINE_PROVIDER,
            "model": cfg.ENGINE_MODEL_OPENAI,
        },
        "residence_time_min": proposal.get("residence_time_min"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_in_channel_min": proposal.get("residence_time_in_channel_min"),
        "stage_parameters": proposal.get("stage_parameters") or [],
        "liquid_flow_rate_mL_min": proposal.get("flow_rate_mL_min"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "temperature_C": proposal.get("temperature_C"),
        "BPR_bar": proposal.get("BPR_bar"),
        "wavelength_nm": proposal.get("wavelength_nm"),
        "deoxygenation_method": proposal.get("deoxygenation_method"),
        "topology_operations": [
            {
                "op_type": operation.get("op_type"),
                "label": operation.get("label"),
            }
            for operation in topology.get("unit_operations") or []
        ],
    }


def _write_checksums(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(run_dir)}")
    (run_dir / "checksums.sha256").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"case_study1_no_inline_degas_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    inventory = _case_study_inventory()
    LabInventory.model_validate(inventory)
    package = _build_package(inventory)
    (run_dir / "input_protocol.txt").write_text(PROMPT_PHOTOREDOX_AEROBIC)
    (run_dir / "inventory.json").write_text(json.dumps(inventory, indent=2))
    (run_dir / "operating_limits.json").write_text(
        json.dumps(OPERATING_LIMITS, indent=2)
    )
    (run_dir / "intake_package.json").write_text(
        json.dumps(package.model_dump(), indent=2, ensure_ascii=False)
    )

    log_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.getLogger().addHandler(log_handler)

    event_count = 0

    def observe(event: dict[str, Any]) -> None:
        nonlocal event_count
        event_count += 1
        with (run_dir / "llm_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(event), ensure_ascii=False) + "\n")

    set_llm_observer(observe)
    set_llm_runtime_overrides(
        temperature=args.temperature,
        capture_content=True,
    )
    try:
        with council_bundle("gpt4o"):
            result = translate(
                PROMPT_PHOTOREDOX_AEROBIC,
                intake_package=package,
            )
        result = _json_safe(result)
        audit = _constraint_audit(result)
        summary = _summary(result, audit)
        summary["llm_event_count"] = event_count

        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        (run_dir / "constraint_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False)
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        for key, filename in (
            ("svg_path", "process_flow.svg"),
            ("png_path", "process_flow.png"),
        ):
            source = Path(str(result.get(key) or ""))
            if source.is_file():
                shutil.copy2(source, run_dir / filename)
        _write_checksums(run_dir)
        print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
        if not audit["passed"]:
            raise SystemExit(2)
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


if __name__ == "__main__":
    main()
