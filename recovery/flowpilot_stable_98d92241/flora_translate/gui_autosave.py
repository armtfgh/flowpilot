"""Autosave helpers for GUI-generated FlowPilot designs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


GUI_RUNS_DIR = Path("outputs/gui_runs")


def autosave_gui_result(
    result: dict[str, Any],
    *,
    intake_package: Any = None,
    source: str = "gui",
    user_input: str | None = None,
    base_dir: Path = GUI_RUNS_DIR,
) -> Path:
    """Write one completed GUI result to a timestamped run folder."""

    run_dir = _unique_run_dir(base_dir, source)
    run_dir.mkdir(parents=True, exist_ok=False)

    safe_result = _json_safe(result)
    (run_dir / "result.json").write_text(json.dumps(safe_result, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(_summary(safe_result, source), indent=2))

    package_payload = intake_package if intake_package is not None else safe_result.get("intake_package")
    if package_payload is not None:
        if hasattr(package_payload, "model_dump"):
            package_payload = package_payload.model_dump()
        (run_dir / "intake_package.json").write_text(
            json.dumps(_json_safe(package_payload), indent=2)
        )

    if user_input:
        (run_dir / "input.txt").write_text(str(user_input))

    _copy_if_exists(safe_result.get("svg_path"), run_dir / "translate_process.svg")
    _copy_if_exists(safe_result.get("png_path"), run_dir / "translate_process.png")

    return run_dir


def _unique_run_dir(base_dir: Path, source: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source)
    candidate = base_dir / f"{stamp}_{safe_source}"
    counter = 2
    while candidate.exists():
        candidate = base_dir / f"{stamp}_{safe_source}_{counter}"
        counter += 1
    return candidate


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _summary(result: dict[str, Any], source: str) -> dict[str, Any]:
    proposal = result.get("proposal") or {}
    gas = {}
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase") or "").lower() == "gas" or stream.get("gas_flow_sccm") is not None:
            gas = stream
            break
    return {
        "source": source,
        "confidence": result.get("confidence"),
        "reaction": (result.get("chemistry_plan") or {}).get("reaction_name"),
        "residence_time_min": proposal.get("residence_time_min"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_in_channel_min": proposal.get("residence_time_in_channel_min"),
        "flow_rate_mL_min": proposal.get("flow_rate_mL_min"),
        "gas_flow_sccm": gas.get("gas_flow_sccm"),
        "gas_flow_actual_mL_min": gas.get("gas_flow_actual_mL_min"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "tubing_ID_mm": proposal.get("tubing_ID_mm"),
        "temperature_C": proposal.get("temperature_C"),
        "concentration_M": proposal.get("concentration_M"),
        "BPR_bar": proposal.get("BPR_bar"),
        "wavelength_nm": proposal.get("wavelength_nm"),
        "inventory_selection": proposal.get("inventory_selection"),
        "safety_flags": proposal.get("safety_flags"),
    }


def _copy_if_exists(src: Any, dst: Path) -> None:
    if not src:
        return
    path = Path(str(src))
    if path.exists() and path.is_file():
        shutil.copy2(path, dst)
