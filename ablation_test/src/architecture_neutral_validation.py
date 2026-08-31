"""Architecture-neutral executability checks for benchmark candidates."""

from __future__ import annotations

import math
import re
from typing import Any

from .cases import AblationCase


AVAILABLE = {"available", "ready", "in_service", "in service"}


def assess_candidate(case: AblationCase, result: dict[str, Any]) -> dict[str, Any]:
    """Assess only the design a system publishes, not its internal architecture."""

    proposal = _published_proposal(result)
    streams = [item for item in proposal.get("streams") or [] if isinstance(item, dict)]
    inventory = case.inventory or {}
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, observed: Any, expected: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    mandatory = {
        key: _number(proposal.get(key))
        for key in (
            "residence_time_min",
            "flow_rate_mL_min",
            "temperature_C",
            "concentration_M",
            "BPR_bar",
            "reactor_volume_mL",
        )
    }
    add(
        "AN-01-SCHEMA-AND-MANDATORY-NUMERICS",
        bool(result.get("schema_valid", bool(proposal)))
        and all(value is not None for value in mandatory.values()),
        mandatory,
        "Schema-valid output with all mandatory numeric run parameters.",
    )
    add(
        "AN-02-PHYSICAL-NUMERIC-DOMAIN",
        all(
            mandatory[key] is not None and mandatory[key] > 0
            for key in (
                "residence_time_min",
                "flow_rate_mL_min",
                "temperature_C",
                "concentration_M",
                "reactor_volume_mL",
            )
        )
        and mandatory["BPR_bar"] is not None
        and mandatory["BPR_bar"] >= 0,
        mandatory,
        "Positive finite process values and nonnegative pressure.",
    )

    liquid = [stream for stream in streams if _phase(stream) != "gas"]
    liquid_flows = [_number(stream.get("flow_rate_mL_min")) for stream in liquid]
    top_flow = mandatory["flow_rate_mL_min"]
    flow_assessable = bool(liquid) and all(value is not None for value in liquid_flows)
    add(
        "AN-03-LIQUID-FLOW-CLOSURE",
        flow_assessable and _close(sum(liquid_flows), top_flow),
        {"stream_sum_mL_min": sum(value or 0 for value in liquid_flows), "top_level_mL_min": top_flow},
        "Sum of physical liquid-feed flows equals the published total liquid flow within 5%.",
    )

    expected_volume = (
        mandatory["flow_rate_mL_min"] * mandatory["residence_time_min"]
        if mandatory["flow_rate_mL_min"] is not None
        and mandatory["residence_time_min"] is not None
        else None
    )
    add(
        "AN-04-VOLUME-FLOW-TIME-CLOSURE",
        _close(mandatory["reactor_volume_mL"], expected_volume),
        {"reported_volume_mL": mandatory["reactor_volume_mL"], "Q_tau_mL": expected_volume},
        "Published reactor volume equals Q_liquid x residence time within 5%.",
    )

    pumps = _available(inventory.get("pumps") or [])
    pump_slots = [pump for pump in pumps for _ in range(int(pump.get("quantity") or 1))]
    pump_range_ok = flow_assessable and len(liquid) <= len(pump_slots) and _flows_match_devices(
        [value for value in liquid_flows if value is not None], pump_slots
    )
    add(
        "AN-05-LIQUID-PUMP-FEASIBILITY",
        pump_range_ok,
        {"liquid_feeds": len(liquid), "available_pump_slots": len(pump_slots), "flows_mL_min": liquid_flows},
        "Each physical liquid feed has one available pump operating within its declared range.",
    )

    reactors = _available(inventory.get("reactors") or [])
    reactor_matches = [
        item
        for item in reactors
        if _close(_number(item.get("volume_mL")), mandatory["reactor_volume_mL"], rel=0.001)
        and _within(mandatory["temperature_C"], item.get("min_temperature_C"), item.get("max_temperature_C"))
        and _allowed(mandatory["temperature_C"], item.get("allowed_temperatures_C"))
    ]
    add(
        "AN-06-REACTOR-CAPABILITY",
        bool(reactor_matches),
        {
            "reported_volume_mL": mandatory["reactor_volume_mL"],
            "reported_temperature_C": mandatory["temperature_C"],
            "matching_equipment_ids": [item.get("equipment_id") for item in reactor_matches],
        },
        "At least one available reactor supports the published volume and temperature.",
    )

    pressure = mandatory["BPR_bar"]
    pressure_values = [_number(item) for item in inventory.get("BPR_available") or []]
    for controller in _available(inventory.get("pressure_controllers") or []):
        pressure_values.extend(_number(item) for item in controller.get("setpoints_bar") or [])
    pressure_values = [value for value in pressure_values if value is not None]
    pressure_ok = pressure == 0 or any(_close(pressure, value, rel=0.001) for value in pressure_values)
    add(
        "AN-07-PRESSURE-CONTROL",
        pressure_ok,
        {"reported_BPR_bar": pressure, "available_setpoints_bar": pressure_values},
        "Zero pressure or an explicitly available BPR setpoint.",
    )

    gas_streams = [stream for stream in streams if _phase(stream) == "gas"]
    gas_hardware = _available(inventory.get("gas_hardware") or [])
    gas_ok = True
    gas_details: list[dict[str, Any]] = []
    for stream in gas_streams:
        identity = _gas_identity(stream)
        q = _number(stream.get("gas_flow_sccm"))
        matches = [
            item
            for item in gas_hardware
            if "mfc" in str(item.get("type") or "").lower()
            and _same_gas(identity, str(item.get("gas") or ""))
            and _within(q, item.get("min_flow_sccm"), item.get("max_flow_sccm"))
        ]
        gas_ok = gas_ok and q is not None and q > 0 and bool(matches)
        gas_details.append(
            {"gas": identity, "flow_sccm": q, "matching_equipment_ids": [item.get("equipment_id") for item in matches]}
        )
    add(
        "AN-08-GAS-FEED-CAPABILITY",
        gas_ok,
        gas_details,
        "Every published gas stream has a compatible available MFC in range; no gas is also valid.",
    )

    wavelength = _number(proposal.get("wavelength_nm"))
    light_text = str(proposal.get("light_setup") or "").lower()
    light_required = wavelength is not None or bool(re.search(r"\b(?:led|lamp|photoreactor)\b", light_text))
    lights = _available(inventory.get("light_sources") or [])
    light_ok = not light_required or any(
        wavelength is None or _close(_number(item.get("wavelength_nm")), wavelength, rel=0.02)
        for item in lights
    )
    add(
        "AN-09-LIGHT-SOURCE-CAPABILITY",
        light_ok,
        {"required": light_required, "wavelength_nm": wavelength, "available": [item.get("wavelength_nm") for item in lights]},
        "No optical source requested, or an available source matches within 2%.",
    )

    failures = [item for item in checks if not item["passed"]]
    return {
        "schema_version": "flowpilot_architecture_neutral_executability_v1.0",
        "status": "executable" if not failures else "not_executable",
        "passed_checks": len(checks) - len(failures),
        "total_checks": len(checks),
        "pass_fraction": round((len(checks) - len(failures)) / len(checks), 6),
        "failed_check_ids": [item["check_id"] for item in failures],
        "checks": checks,
    }


def _published_proposal(result: dict[str, Any]) -> dict[str, Any]:
    proposal = dict(result.get("proposal") or {})
    final = result.get("final_design")
    if isinstance(final, dict) and final.get("status") == "executable":
        proposal.update(final.get("parameters") or {})
        if final.get("streams"):
            proposal["streams"] = final["streams"]
    return proposal


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _close(a: float | None, b: float | None, *, rel: float = 0.05) -> bool:
    return a is not None and b is not None and abs(a - b) <= rel * max(abs(b), 1e-12)


def _available(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if str(item.get("service_status") or "available").lower() in AVAILABLE]


def _within(value: float | None, minimum: Any, maximum: Any) -> bool:
    if value is None:
        return False
    low, high = _number(minimum), _number(maximum)
    return (low is None or value >= low) and (high is None or value <= high)


def _allowed(value: float | None, allowed: Any) -> bool:
    values = [_number(item) for item in (allowed or [])]
    values = [item for item in values if item is not None]
    return not values or any(_close(value, item, rel=0.001) for item in values)


def _flows_match_devices(flows: list[float], devices: list[dict[str, Any]]) -> bool:
    if not flows:
        return False
    used: set[int] = set()

    def assign(index: int) -> bool:
        if index == len(flows):
            return True
        for device_index, device in enumerate(devices):
            if device_index in used:
                continue
            if not _within(flows[index], device.get("min_flow_rate_mL_min"), device.get("max_flow_rate_mL_min")):
                continue
            used.add(device_index)
            if assign(index + 1):
                return True
            used.remove(device_index)
        return False

    return assign(0)


def _phase(stream: dict[str, Any]) -> str:
    phase = str(stream.get("phase") or "").lower()
    return "gas" if phase == "gas" or stream.get("gas_flow_sccm") not in (None, 0, 0.0, "") else "liquid"


def _gas_identity(stream: dict[str, Any]) -> str:
    text = " ".join(str(item) for item in stream.get("contents") or []).lower().replace("₂", "2")
    for identity, pattern in (
        ("air", r"\bair\b"),
        ("O2", r"\b(?:o2|oxygen)\b"),
        ("H2", r"\b(?:h2|hydrogen)\b"),
        ("CO2", r"\b(?:co2|carbon dioxide)\b"),
    ):
        if re.search(pattern, text):
            return identity
    return text.strip()


def _same_gas(left: str, right: str) -> bool:
    aliases = {
        "oxygen": "o2", "o2": "o2", "hydrogen": "h2", "h2": "h2",
        "carbon dioxide": "co2", "co2": "co2", "air": "air",
    }
    return aliases.get(left.lower(), left.lower()) == aliases.get(right.lower(), right.lower())
