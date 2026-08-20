from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ablation_test.src.paths import resolve_artifact_path


ARCHITECTURES = ("one_shot", "flowpilot")
ARCH_LABELS = {
    "one_shot": "Qwen 27B one-shot",
    "flowpilot": "Qwen 27B + FlowPilot",
}
ARCH_COLORS = {
    "one_shot": "#4C78A8",
    "flowpilot": "#2A9D8F",
}
DIMENSION_ORDER = (
    "inventory_compliance",
    "numerical_closure",
    "chemistry_fidelity",
    "process_completeness",
)
DIMENSION_LABELS = {
    "inventory_compliance": "Inventory compliance",
    "numerical_closure": "Numerical closure",
    "chemistry_fidelity": "Chemistry fidelity",
    "process_completeness": "Process completeness",
}
FAMILY_ORDER = (
    "suzuki",
    "photo_oxidation",
    "hydrogenolysis",
    "dinitration",
    "multistep",
)
FAMILY_LABELS = {
    "suzuki": "Suzuki",
    "photo_oxidation": "Photo",
    "hydrogenolysis": "H2",
    "dinitration": "Nitration",
    "multistep": "Multistep",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> bool:
    number = _as_float(value)
    return number is not None and number > 0


def _close(observed: Any, expected: Any, tolerance: float) -> bool:
    left = _as_float(observed)
    right = _as_float(expected)
    if left is None or right is None:
        return False
    scale = max(abs(right), 1e-9)
    return abs(left - right) / scale <= tolerance


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return str(value or "")


def _family(scenario_id: str) -> str:
    return scenario_id.removesuffix("_feasible").removesuffix("_infeasible")


def _streams(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in proposal.get("streams") or [] if isinstance(item, dict)]


def _gas_stream(proposal: dict[str, Any]) -> dict[str, Any] | None:
    for stream in _streams(proposal):
        if str(stream.get("phase") or "").lower() == "gas":
            return stream
        if _positive(stream.get("gas_flow_sccm")):
            return stream
    return None


def _liquid_streams(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for stream in _streams(proposal):
        if str(stream.get("phase") or "").lower() == "gas":
            continue
        if _positive(stream.get("gas_flow_sccm")):
            continue
        output.append(stream)
    return output


def _gas_identity(stream: dict[str, Any] | None) -> str:
    if not stream:
        return ""
    return _flatten(stream.get("contents")).lower()


def _gas_reagent_fraction(identity: str) -> float:
    return 0.21 if "air" in identity else 1.0


def _check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    category: str,
    passed: bool,
    critical: bool,
    observed: Any,
    requirement: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "passed": bool(passed),
            "critical": bool(critical),
            "observed": json.dumps(observed, ensure_ascii=True)
            if isinstance(observed, (dict, list))
            else str(observed),
            "requirement": requirement,
        }
    )


def _topology_present(
    feature: str,
    proposal: dict[str, Any],
    synonyms: dict[str, list[str]],
) -> bool:
    text = _flatten(proposal).lower().replace("_", " ")
    liquid_count = len(_liquid_streams(proposal))
    gas = _gas_stream(proposal)
    identity = _gas_identity(gas)
    if feature in {"feed_pumps", "liquid_pump"}:
        return liquid_count >= 1
    if feature == "separate_feed_pumps":
        return liquid_count >= 2
    if feature in {"mixer", "gas_liquid_mixer", "micromixer"}:
        mixer = str(proposal.get("mixer_type") or "").lower()
        if feature == "micromixer":
            return "micro" in mixer or "micromixer" in text
        return bool(mixer.strip())
    if feature == "oxygen_mass_flow_controller":
        return gas is not None and ("air" in identity or "oxygen" in identity or "o2" in identity)
    if feature == "hydrogen_mass_flow_controller":
        return gas is not None and ("hydrogen" in identity or "h2" in identity)
    if feature == "back_pressure_regulator":
        return _positive(proposal.get("BPR_bar"))
    if feature == "photoreactor":
        return _positive(proposal.get("wavelength_nm")) and (
            "photo" in text or "led" in text
        )
    if feature == "heated_reactor":
        return _positive(proposal.get("temperature_C"))
    if feature == "temperature_controlled_microreactor":
        return "micro" in text and _positive(proposal.get("temperature_C"))
    if feature == "packed_bed":
        return "packed bed" in text or "packed-bed" in text
    return any(term.lower().replace("_", " ") in text for term in synonyms.get(feature, []))


def _inventory_checks(
    proposal: dict[str, Any],
    scenario: dict[str, Any],
    checks: list[dict[str, Any]],
    tolerance: float,
) -> None:
    inventory = scenario["inventory"]
    volume = _as_float(proposal.get("reactor_volume_mL"))
    diameter = _as_float(proposal.get("tubing_ID_mm"))
    temperature = _as_float(proposal.get("temperature_C"))
    flow = _as_float(proposal.get("flow_rate_mL_min"))
    bpr = _as_float(proposal.get("BPR_bar"))
    wavelength = _as_float(proposal.get("wavelength_nm"))

    reactors = inventory.get("reactors") or []
    matching = [
        reactor
        for reactor in reactors
        if _close(volume, reactor.get("volume_mL"), tolerance)
        and _close(diameter, reactor.get("ID_mm"), tolerance)
    ]
    _check(
        checks,
        check_id="INV-REACTOR",
        category="inventory_compliance",
        passed=bool(matching),
        critical=True,
        observed={"volume_mL": volume, "ID_mm": diameter},
        requirement="Volume and ID must match one listed reactor.",
    )
    reactor_temperature_ok = bool(matching) and any(
        temperature is not None
        and temperature <= float(reactor.get("max_temperature_C") or math.inf)
        for reactor in matching
    )
    tubing_temperature_ok = any(
        temperature is not None
        and temperature <= float(item.get("max_temperature_C") or math.inf)
        and _close(diameter, item.get("ID_mm"), tolerance)
        for item in inventory.get("tubing") or []
    )
    _check(
        checks,
        check_id="INV-TEMPERATURE-RATING",
        category="inventory_compliance",
        passed=reactor_temperature_ok and tubing_temperature_ok,
        critical=True,
        observed=temperature,
        requirement="Temperature must be within listed reactor and tubing ratings.",
    )

    pumps = inventory.get("pumps") or []
    pump_ok = any(
        flow is not None
        and float(item.get("min_flow_rate_mL_min") or 0.0) <= flow
        <= float(item.get("max_flow_rate_mL_min") or math.inf)
        for item in pumps
    )
    _check(
        checks,
        check_id="INV-PUMP-FLOW",
        category="inventory_compliance",
        passed=pump_ok,
        critical=True,
        observed=flow,
        requirement="Liquid flow must be within a listed pump range.",
    )

    available_bprs = [float(item) for item in inventory.get("BPR_available") or []]
    bpr_ok = bpr is not None and any(
        _close(bpr, item, tolerance) for item in available_bprs
    )
    _check(
        checks,
        check_id="INV-BPR",
        category="inventory_compliance",
        passed=bpr_ok,
        critical=True,
        observed=bpr,
        requirement=f"BPR must be one of the listed settings: {available_bprs}.",
    )
    pressure_ratings = [
        float(item.get("max_pressure_bar") or math.inf)
        for item in pumps + (inventory.get("tubing") or [])
    ]
    _check(
        checks,
        check_id="INV-PRESSURE-RATING",
        category="inventory_compliance",
        passed=bpr is not None
        and bool(pressure_ratings)
        and bpr <= min(pressure_ratings),
        critical=True,
        observed=bpr,
        requirement="BPR setting must not exceed the limiting pump or tubing rating.",
    )

    light_sources = inventory.get("light_sources") or []
    if light_sources:
        light_ok = any(
            _close(wavelength, item.get("wavelength_nm"), tolerance)
            for item in light_sources
        )
        _check(
            checks,
            check_id="INV-LIGHT",
            category="inventory_compliance",
            passed=light_ok,
            critical=True,
            observed=wavelength,
            requirement="Wavelength must match a listed light source.",
        )

    proposal_text = _flatten(proposal).lower()
    unlisted_phrases = (
        "not in inventory, assumed",
        "not in inventory; assumed",
        "assumed standard lab",
        "assuming standard lab",
    )
    found = [phrase for phrase in unlisted_phrases if phrase in proposal_text]
    _check(
        checks,
        check_id="INV-NO-ASSUMED-EQUIPMENT",
        category="inventory_compliance",
        passed=not found,
        critical=True,
        observed=found,
        requirement="The outcome must not explicitly assume unlisted equipment.",
    )


def _numerical_checks(
    proposal: dict[str, Any],
    scenario: dict[str, Any],
    checks: list[dict[str, Any]],
    tolerance: float,
    gas_tolerance: float,
) -> None:
    volume = _as_float(proposal.get("reactor_volume_mL"))
    tau = _as_float(proposal.get("residence_time_min"))
    liquid_flow = _as_float(proposal.get("flow_rate_mL_min"))
    temperature = _as_float(proposal.get("temperature_C"))
    bpr = _as_float(proposal.get("BPR_bar"))
    basis = str(proposal.get("residence_time_basis") or "").lower()
    gas = _gas_stream(proposal)

    expected_volume = None
    if volume is not None and tau is not None and liquid_flow is not None:
        expected_volume = tau * liquid_flow
        if gas and "inlet" in basis:
            expected_volume = tau * (
                liquid_flow + float(_as_float(gas.get("gas_flow_sccm")) or 0.0)
            )
        elif gas and ("channel" in basis or "pressure" in basis):
            expected_volume = tau * (
                liquid_flow
                + float(_as_float(gas.get("gas_flow_actual_mL_min")) or 0.0)
            )
    _check(
        checks,
        check_id="NUM-V-Q-TAU",
        category="numerical_closure",
        passed=_close(volume, expected_volume, tolerance),
        critical=True,
        observed={"V": volume, "Q": liquid_flow, "tau": tau, "basis": basis},
        requirement="Reactor volume must close against flow and stated residence-time basis.",
    )

    liquid_streams = _liquid_streams(proposal)
    stream_flows = [
        _as_float(stream.get("flow_rate_mL_min")) for stream in liquid_streams
    ]
    stream_flows = [item for item in stream_flows if item is not None and item > 0]
    if stream_flows:
        _check(
            checks,
            check_id="NUM-LIQUID-STREAM-SUM",
            category="numerical_closure",
            passed=_close(sum(stream_flows), liquid_flow, tolerance),
            critical=False,
            observed={"sum": sum(stream_flows), "reported": liquid_flow},
            requirement="Individual liquid-stream flows should sum to total liquid flow.",
        )

    gas_expected = "gas" in str(scenario.get("expected_features", {}).get("phase_regime", ""))
    if not gas_expected:
        return

    identity = _gas_identity(gas)
    stp_flow = _as_float((gas or {}).get("gas_flow_sccm"))
    actual_flow = _as_float((gas or {}).get("gas_flow_actual_mL_min"))
    declared_equiv = _as_float((gas or {}).get("molar_equiv"))
    concentration = _as_float(proposal.get("concentration_M"))
    pressure_abs = (bpr or 0.0) + 1.01325
    actual_expected = None
    if stp_flow is not None and temperature is not None and pressure_abs > 0:
        actual_expected = stp_flow * (temperature + 273.15) / 273.15 / pressure_abs
    _check(
        checks,
        check_id="NUM-GAS-STP",
        category="numerical_closure",
        passed=stp_flow is not None and stp_flow > 0,
        critical=True,
        observed=stp_flow,
        requirement="Gas flow at inlet/STP must be explicit and positive.",
    )
    _check(
        checks,
        check_id="NUM-GAS-PRESSURE-CORRECTION",
        category="numerical_closure",
        passed=_close(actual_flow, actual_expected, gas_tolerance),
        critical=True,
        observed={"reported": actual_flow, "expected": actual_expected},
        requirement="In-channel gas flow must satisfy ideal-gas pressure correction.",
    )

    calculated_equiv = None
    if (
        stp_flow is not None
        and concentration is not None
        and concentration > 0
        and liquid_flow is not None
        and liquid_flow > 0
    ):
        gas_mmol_min = stp_flow / 22.414
        substrate_mmol_min = concentration * liquid_flow
        calculated_equiv = (
            gas_mmol_min * _gas_reagent_fraction(identity) / substrate_mmol_min
        )
    _check(
        checks,
        check_id="NUM-GAS-EQUIV",
        category="numerical_closure",
        passed=declared_equiv is not None
        and declared_equiv > 0
        and _close(declared_equiv, calculated_equiv, gas_tolerance),
        critical=True,
        observed={"declared": declared_equiv, "calculated": calculated_equiv},
        requirement="Declared gas equivalents must close from STP flow and substrate molar flow.",
    )

    inlet_tau = _as_float(proposal.get("residence_time_inlet_min"))
    channel_tau = _as_float(proposal.get("residence_time_in_channel_min"))
    expected_inlet = None
    expected_channel = None
    if volume is not None and liquid_flow is not None and stp_flow is not None:
        expected_inlet = volume / max(liquid_flow + stp_flow, 1e-9)
    if volume is not None and liquid_flow is not None and actual_flow is not None:
        expected_channel = volume / max(liquid_flow + actual_flow, 1e-9)
    _check(
        checks,
        check_id="NUM-GAS-INLET-TAU",
        category="numerical_closure",
        passed=_close(inlet_tau, expected_inlet, gas_tolerance),
        critical=True,
        observed={"reported": inlet_tau, "expected": expected_inlet},
        requirement="Inlet/STP residence time must close from total inlet volumetric flow.",
    )
    _check(
        checks,
        check_id="NUM-GAS-CHANNEL-TAU",
        category="numerical_closure",
        passed=_close(channel_tau, expected_channel, gas_tolerance),
        critical=True,
        observed={"reported": channel_tau, "expected": expected_channel},
        requirement="In-channel residence time must close from pressure-corrected total flow.",
    )


def _chemistry_and_process_checks(
    proposal: dict[str, Any],
    scenario: dict[str, Any],
    checks: list[dict[str, Any]],
    spec: dict[str, Any],
) -> None:
    tolerance = spec["thresholds"]["relative_closure_tolerance"]
    for constraint in scenario.get("oracle_constraints") or []:
        field = constraint["field"]
        value = _as_float(proposal.get(field))
        passed = value is not None
        if constraint.get("allowed"):
            passed = passed and any(
                _close(value, item, tolerance) for item in constraint["allowed"]
            )
        if constraint.get("min") is not None:
            passed = passed and value >= float(constraint["min"])
        if constraint.get("max") is not None:
            passed = passed and value <= float(constraint["max"])
        _check(
            checks,
            check_id=f"CHEM-{constraint['constraint_id']}",
            category="chemistry_fidelity",
            passed=passed,
            critical=True,
            observed=value,
            requirement=json.dumps(constraint, sort_keys=True),
        )

    gas = _gas_stream(proposal)
    identity = _gas_identity(gas)
    if scenario["scenario_id"].startswith("photo_oxidation"):
        gas_ok = "air" in identity or "oxygen" in identity or "o2" in identity
        _check(
            checks,
            check_id="CHEM-GAS-IDENTITY",
            category="chemistry_fidelity",
            passed=gas_ok,
            critical=True,
            observed=identity,
            requirement="Aerobic photo-oxidation requires air or oxygen.",
        )
    if scenario["scenario_id"].startswith("hydrogenolysis"):
        gas_ok = "hydrogen" in identity or "h2" in identity
        _check(
            checks,
            check_id="CHEM-GAS-IDENTITY",
            category="chemistry_fidelity",
            passed=gas_ok,
            critical=True,
            observed=identity,
            requirement="Hydrogenolysis requires hydrogen.",
        )

    synonyms = spec["topology_synonyms"]
    hard_topology = {
        "gas_liquid_mixer",
        "oxygen_mass_flow_controller",
        "hydrogen_mass_flow_controller",
        "packed_bed",
        "micromixer",
        "immediate_quench",
        "interstage_addition",
        "stage_1_oxidation",
        "stage_2_amidation",
    }
    for feature in scenario.get("expected_features", {}).get("topology") or []:
        present = _topology_present(feature, proposal, synonyms)
        _check(
            checks,
            check_id=f"PROC-TOPOLOGY-{feature.upper()}",
            category="process_completeness",
            passed=present,
            critical=feature in hard_topology,
            observed=present,
            requirement=f"Required topology feature: {feature}.",
        )

    liquid_streams = _liquid_streams(proposal)
    _check(
        checks,
        check_id="PROC-LIQUID-FEED",
        category="process_completeness",
        passed=bool(liquid_streams)
        and all(_flatten(item.get("contents")).strip() for item in liquid_streams),
        critical=False,
        observed=len(liquid_streams),
        requirement="At least one liquid feed with explicit contents is required.",
    )
    _check(
        checks,
        check_id="PROC-PRE-REACTOR",
        category="process_completeness",
        passed=bool(_flatten(proposal.get("pre_reactor_steps")).strip()),
        critical=False,
        observed=bool(_flatten(proposal.get("pre_reactor_steps")).strip()),
        requirement="Pre-reactor preparation should be explicit.",
    )
    _check(
        checks,
        check_id="PROC-POST-REACTOR",
        category="process_completeness",
        passed=bool(_flatten(proposal.get("post_reactor_steps")).strip()),
        critical=False,
        observed=bool(_flatten(proposal.get("post_reactor_steps")).strip()),
        requirement="Collection, quench, or downstream handling should be explicit.",
    )
    _check(
        checks,
        check_id="PROC-SAFETY",
        category="process_completeness",
        passed=bool(proposal.get("safety_flags"))
        or "safety" in _flatten(proposal).lower()
        or "hazard" in _flatten(proposal).lower(),
        critical=False,
        observed=bool(proposal.get("safety_flags")),
        requirement="At least one explicit safety or hazard statement is required.",
    )


def evaluate_cell(
    *,
    raw: dict[str, str],
    architecture: str,
    scenario: dict[str, Any],
    result: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposal = result.get("proposal") or {}
    reported = str(raw.get("reported_disposition") or result.get("reported_disposition") or "")
    expected = scenario["expected_disposition"]
    disposition_correct = reported == expected
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        check_id="CORE-DISPOSITION",
        category="chemistry_fidelity",
        passed=disposition_correct,
        critical=True,
        observed=reported,
        requirement=f"Expected disposition is {expected}.",
    )

    if scenario["scenario_kind"] == "infeasible":
        rationale = (
            str(result.get("disposition_rationale") or "")
            + " "
            + _flatten(result.get("design_disposition") or {})
        ).lower()
        conflict_terms = {
            "suzuki_infeasible": ("temperature", "40", "50"),
            "photo_oxidation_infeasible": ("525", "wavelength", "light"),
            "hydrogenolysis_infeasible": ("hydrogen", "prohibit", "unavailable"),
            "dinitration_infeasible": ("prohibit", "nitration", "safety"),
            "multistep_infeasible": ("interstage", "second reactor", "unavailable"),
        }[scenario["scenario_id"]]
        identifies_conflict = any(term in rationale for term in conflict_terms)
        _check(
            checks,
            check_id="BLOCK-RATIONALE",
            category="process_completeness",
            passed=identifies_conflict,
            critical=False,
            observed=rationale[:300],
            requirement="Block rationale should identify the controlled inventory conflict.",
        )
        critical_failures = [item for item in checks if item["critical"] and not item["passed"]]
        row = {
            "architecture": architecture,
            "architecture_label": ARCH_LABELS[architecture],
            "scenario_id": scenario["scenario_id"],
            "family": _family(scenario["scenario_id"]),
            "scenario_kind": "infeasible",
            "repeat": int(raw["repeat"]),
            "seed": int(raw["seed"]),
            "expected_disposition": expected,
            "reported_disposition": reported,
            "disposition_correct": disposition_correct,
            "executable": False,
            "critical_failure_count": len(critical_failures),
            "critical_failure_ids": "|".join(item["check_id"] for item in critical_failures),
            "quality_score": "",
            **{f"{key}_score": "" for key in DIMENSION_ORDER},
            "design_input_sha256": raw["design_input_sha256"],
            "run_dir": raw["run_dir"],
        }
        return row, checks

    proposal_present = isinstance(proposal, dict) and bool(proposal)
    _check(
        checks,
        check_id="CORE-PROPOSAL",
        category="process_completeness",
        passed=proposal_present,
        critical=True,
        observed=proposal_present,
        requirement="Feasible cases require a structured design proposal.",
    )
    for field in spec["mandatory_core_fields"]:
        value = proposal.get(field)
        _check(
            checks,
            check_id=f"CORE-FIELD-{field.upper()}",
            category="process_completeness",
            passed=_positive(value),
            critical=True,
            observed=value,
            requirement=f"Mandatory positive field: {field}.",
        )

    _inventory_checks(
        proposal,
        scenario,
        checks,
        spec["thresholds"]["relative_closure_tolerance"],
    )
    _numerical_checks(
        proposal,
        scenario,
        checks,
        spec["thresholds"]["relative_closure_tolerance"],
        spec["thresholds"]["gas_relative_tolerance"],
    )
    _chemistry_and_process_checks(proposal, scenario, checks, spec)

    dimension_scores = {}
    for category in DIMENSION_ORDER:
        category_checks = [item for item in checks if item["category"] == category]
        dimension_scores[category] = (
            100 * sum(item["passed"] for item in category_checks) / len(category_checks)
            if category_checks
            else 0.0
        )
    quality = sum(
        dimension_scores[key] * float(spec["quality_dimensions"][key])
        for key in DIMENSION_ORDER
    )
    critical_failures = [item for item in checks if item["critical"] and not item["passed"]]
    executable = disposition_correct and proposal_present and not critical_failures
    proposal_values = {
        "temperature_C": proposal.get("temperature_C"),
        "residence_time_min": proposal.get("residence_time_min"),
        "flow_rate_mL_min": proposal.get("flow_rate_mL_min"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "tubing_ID_mm": proposal.get("tubing_ID_mm"),
        "BPR_bar": proposal.get("BPR_bar"),
        "wavelength_nm": proposal.get("wavelength_nm"),
        "gas_flow_sccm": (_gas_stream(proposal) or {}).get("gas_flow_sccm"),
        "gas_flow_actual_mL_min": (_gas_stream(proposal) or {}).get(
            "gas_flow_actual_mL_min"
        ),
        "gas_equiv": (_gas_stream(proposal) or {}).get("molar_equiv"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_in_channel_min": proposal.get(
            "residence_time_in_channel_min"
        ),
    }
    row = {
        "architecture": architecture,
        "architecture_label": ARCH_LABELS[architecture],
        "scenario_id": scenario["scenario_id"],
        "family": _family(scenario["scenario_id"]),
        "scenario_kind": "feasible",
        "repeat": int(raw["repeat"]),
        "seed": int(raw["seed"]),
        "expected_disposition": expected,
        "reported_disposition": reported,
        "disposition_correct": disposition_correct,
        "executable": executable,
        "critical_failure_count": len(critical_failures),
        "critical_failure_ids": "|".join(item["check_id"] for item in critical_failures),
        "quality_score": round(quality, 3),
        **{
            f"{key}_score": round(value, 3)
            for key, value in dimension_scores.items()
        },
        **proposal_values,
        "design_input_sha256": raw["design_input_sha256"],
        "run_dir": raw["run_dir"],
    }
    return row, checks


def pair_feasible_rows(
    rows: list[dict[str, Any]], tie_points: float
) -> list[dict[str, Any]]:
    lookup = {
        (row["architecture"], row["scenario_id"], row["repeat"]): row
        for row in rows
        if row["scenario_kind"] == "feasible"
    }
    output = []
    for scenario in sorted({row["scenario_id"] for row in rows if row["scenario_kind"] == "feasible"}):
        for repeat in (1, 2, 3):
            one = lookup[("one_shot", scenario, repeat)]
            flow = lookup[("flowpilot", scenario, repeat)]
            if flow["executable"] != one["executable"]:
                winner = "flowpilot" if flow["executable"] else "one_shot"
                reason = "executable_design_gate"
            elif flow["executable"] and one["executable"]:
                delta = float(flow["quality_score"]) - float(one["quality_score"])
                winner = (
                    "flowpilot"
                    if delta > tie_points
                    else "one_shot"
                    if delta < -tie_points
                    else "tie"
                )
                reason = "quality_score" if winner != "tie" else "quality_tie"
            else:
                failure_delta = (
                    int(one["critical_failure_count"])
                    - int(flow["critical_failure_count"])
                )
                winner = (
                    "flowpilot"
                    if failure_delta > 0
                    else "one_shot"
                    if failure_delta < 0
                    else "tie"
                )
                reason = "fewer_critical_failures" if winner != "tie" else "invalid_tie"
            output.append(
                {
                    "scenario_id": scenario,
                    "family": _family(scenario),
                    "repeat": repeat,
                    "one_shot_executable": one["executable"],
                    "flowpilot_executable": flow["executable"],
                    "one_shot_quality_score": one["quality_score"],
                    "flowpilot_quality_score": flow["quality_score"],
                    "quality_difference_flowpilot_minus_one_shot": round(
                        float(flow["quality_score"]) - float(one["quality_score"]), 3
                    ),
                    "one_shot_critical_failures": one["critical_failure_count"],
                    "flowpilot_critical_failures": flow["critical_failure_count"],
                    "winner": winner,
                    "decision_reason": reason,
                }
            )
    return output


def _cluster_bootstrap(
    pairs: list[dict[str, Any]],
    value_fn,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["family"]].append(row)
    families = sorted(grouped)
    observed = mean(value_fn(row) for row in pairs)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        sampled_rows = [row for family in sampled_families for row in grouped[str(family)]]
        samples.append(mean(value_fn(row) for row in sampled_rows))
    low, high = np.percentile(samples, [2.5, 97.5])
    return observed, float(low), float(high)


def build_reproducibility_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "temperature_C",
        "residence_time_min",
        "flow_rate_mL_min",
        "reactor_volume_mL",
        "tubing_ID_mm",
        "BPR_bar",
        "wavelength_nm",
        "gas_flow_sccm",
        "gas_flow_actual_mL_min",
        "gas_equiv",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scenario_kind"] == "feasible":
            grouped[(row["architecture"], row["family"])].append(row)
    output = []
    for architecture in ARCHITECTURES:
        for family in FAMILY_ORDER:
            group = grouped[(architecture, family)]
            cvs = []
            agreements = []
            for field in numeric_fields:
                values = [_as_float(row.get(field)) for row in group]
                values = [item for item in values if item is not None]
                if len(values) != len(group) or not values:
                    continue
                center = abs(mean(values))
                if center > 1e-9:
                    cvs.append(pstdev(values) / center)
                agreements.append(
                    max(values) - min(values) <= max(1e-6, 0.01 * center)
                )
            output.append(
                {
                    "architecture": architecture,
                    "family": family,
                    "repeat_count": len(group),
                    "executable_repeat_rate": sum(row["executable"] for row in group)
                    / len(group),
                    "median_numeric_cv": median(cvs) if cvs else 0.0,
                    "parameter_exact_agreement_rate": (
                        sum(agreements) / len(agreements) if agreements else 0.0
                    ),
                    "quality_score_sd": pstdev(
                        float(row["quality_score"]) for row in group
                    ),
                }
            )
    return output


def build_weight_sensitivity_rows(
    rows: list[dict[str, Any]], tie_points: float
) -> list[dict[str, Any]]:
    schemes = {
        "registered": {
            "inventory_compliance": 0.30,
            "numerical_closure": 0.30,
            "chemistry_fidelity": 0.20,
            "process_completeness": 0.20,
        },
        "equal_weights": {key: 0.25 for key in DIMENSION_ORDER},
        "inventory_heavy": {
            "inventory_compliance": 0.40,
            "numerical_closure": 0.20,
            "chemistry_fidelity": 0.20,
            "process_completeness": 0.20,
        },
        "numerical_heavy": {
            "inventory_compliance": 0.20,
            "numerical_closure": 0.40,
            "chemistry_fidelity": 0.20,
            "process_completeness": 0.20,
        },
        "chemistry_heavy": {
            "inventory_compliance": 0.20,
            "numerical_closure": 0.20,
            "chemistry_fidelity": 0.40,
            "process_completeness": 0.20,
        },
        "process_heavy": {
            "inventory_compliance": 0.20,
            "numerical_closure": 0.20,
            "chemistry_fidelity": 0.20,
            "process_completeness": 0.40,
        },
    }
    lookup = {
        (row["architecture"], row["scenario_id"], row["repeat"]): row
        for row in rows
        if row["scenario_kind"] == "feasible"
    }
    output = []
    for scheme, weights in schemes.items():
        counts: Counter[str] = Counter()
        differences = []
        for scenario in sorted(
            {row["scenario_id"] for row in rows if row["scenario_kind"] == "feasible"}
        ):
            for repeat in (1, 2, 3):
                one = lookup[("one_shot", scenario, repeat)]
                flow = lookup[("flowpilot", scenario, repeat)]
                one_score = sum(
                    float(one[f"{key}_score"]) * weight
                    for key, weight in weights.items()
                )
                flow_score = sum(
                    float(flow[f"{key}_score"]) * weight
                    for key, weight in weights.items()
                )
                differences.append(flow_score - one_score)
                if flow["executable"] != one["executable"]:
                    winner = (
                        "flowpilot"
                        if flow["executable"]
                        else "one_shot"
                    )
                elif flow["executable"] and one["executable"]:
                    delta = flow_score - one_score
                    winner = (
                        "flowpilot"
                        if delta > tie_points
                        else "one_shot"
                        if delta < -tie_points
                        else "tie"
                    )
                else:
                    failure_delta = (
                        int(one["critical_failure_count"])
                        - int(flow["critical_failure_count"])
                    )
                    winner = (
                        "flowpilot"
                        if failure_delta > 0
                        else "one_shot"
                        if failure_delta < 0
                        else "tie"
                    )
                counts[winner] += 1
        output.append(
            {
                "weight_scheme": scheme,
                **{f"weight_{key}": value for key, value in weights.items()},
                "flowpilot_wins": counts["flowpilot"],
                "ties": counts["tie"],
                "one_shot_wins": counts["one_shot"],
                "mean_quality_difference": mean(differences),
            }
        )
    return output


def _build_study_figure(figures: Path, model_label: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.45", edgecolor="#333333", linewidth=1.2)
    ax.text(
        0.5,
        0.84,
        f"Same {model_label} + identical frozen input",
        ha="center",
        va="center",
        fontsize=16,
        bbox={**box, "facecolor": "#F4F4F4"},
    )
    ax.annotate("", xy=(0.27, 0.62), xytext=(0.46, 0.78), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.73, 0.62), xytext=(0.54, 0.78), arrowprops={"arrowstyle": "->"})
    ax.text(
        0.25,
        0.52,
        "One-shot translation",
        ha="center",
        fontsize=14,
        bbox={**box, "facecolor": "#DCE8F5"},
    )
    ax.text(
        0.75,
        0.52,
        "Full FlowPilot\nupstream → engineering → council → validator",
        ha="center",
        fontsize=14,
        bbox={**box, "facecolor": "#D7EFEA"},
    )
    ax.annotate("", xy=(0.43, 0.26), xytext=(0.25, 0.42), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.57, 0.26), xytext=(0.75, 0.42), arrowprops={"arrowstyle": "->"})
    ax.text(
        0.5,
        0.18,
        "Paired deterministic outcome evaluation\n"
        "5 chemistry families × 2 inventory states × 3 repeats = 30 pairs",
        ha="center",
        va="center",
        fontsize=14,
        bbox={**box, "facecolor": "#FFF2CC"},
    )
    _save(fig, figures, "01_controlled_benchmark_design")


def _build_executable_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    categories = ("Feasible executable designs", "Infeasible correctly blocked")
    values: dict[str, list[float]] = {}
    labels: dict[str, list[str]] = {}
    for architecture in ARCHITECTURES:
        feasible = [
            row
            for row in rows
            if row["architecture"] == architecture and row["scenario_kind"] == "feasible"
        ]
        infeasible = [
            row
            for row in rows
            if row["architecture"] == architecture and row["scenario_kind"] == "infeasible"
        ]
        feasible_n = sum(row["executable"] for row in feasible)
        blocked_n = sum(row["disposition_correct"] for row in infeasible)
        values[architecture] = [
            100 * feasible_n / len(feasible),
            100 * blocked_n / len(infeasible),
        ]
        labels[architecture] = [
            f"{feasible_n}/{len(feasible)}",
            f"{blocked_n}/{len(infeasible)}",
        ]
    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for offset, architecture in ((-width / 2, "one_shot"), (width / 2, "flowpilot")):
        bars = ax.bar(
            x + offset,
            values[architecture],
            width,
            color=ARCH_COLORS[architecture],
            label=ARCH_LABELS[architecture],
        )
        ax.bar_label(bars, labels=labels[architecture], padding=3)
    ax.set_xticks(x, categories)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Executable outcomes and calibrated blocking")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.24)
    _save(fig, figures, "02_executable_and_block_rate")


def _build_paired_quality_figure(
    pairs: list[dict[str, Any]], figures: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for row in pairs:
        display_jitter = (
            (int(row["repeat"]) - 2) * 0.7
            + (FAMILY_ORDER.index(row["family"]) - 2) * 0.06
        )
        y_one = (
            float(row["one_shot_quality_score"])
            if row["one_shot_executable"]
            else -6.0
        ) + display_jitter
        y_flow = (
            float(row["flowpilot_quality_score"])
            if row["flowpilot_executable"]
            else -6.0
        ) + display_jitter
        color = (
            "#2A9D8F"
            if row["winner"] == "flowpilot"
            else "#4C78A8"
            if row["winner"] == "one_shot"
            else "#999999"
        )
        ax.plot([0, 1], [y_one, y_flow], color=color, alpha=0.55, linewidth=1.3)
        ax.scatter(
            [0],
            [y_one],
            color=ARCH_COLORS["one_shot"],
            marker="o" if row["one_shot_executable"] else "X",
            s=48,
        )
        ax.scatter(
            [1],
            [y_flow],
            color=ARCH_COLORS["flowpilot"],
            marker="o" if row["flowpilot_executable"] else "X",
            s=48,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.text(0.5, -8.5, "× = non-executable outcome", ha="center", fontsize=10)
    ax.text(
        0.5,
        1.01,
        "Display jitter (≤0.82 points) separates coincident repeats",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.set_xticks([0, 1], [ARCH_LABELS[item] for item in ARCHITECTURES])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-11, 105)
    ax.set_ylabel("Quality score (executable designs)")
    fig.suptitle("Paired feasible-case outcome comparison", fontsize=16, y=0.98)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.88)
    _save(fig, figures, "03_paired_outcome_quality")


def _build_win_figure(pairs: list[dict[str, Any]], figures: Path) -> None:
    counts = Counter(row["winner"] for row in pairs)
    labels = ("flowpilot", "tie", "one_shot")
    colors = (ARCH_COLORS["flowpilot"], "#B8B8B8", ARCH_COLORS["one_shot"])
    total = len(pairs)
    left = 0.0
    fig, ax = plt.subplots(figsize=(10.2, 2.8))
    for label, color in zip(labels, colors):
        value = 100 * counts[label] / total
        ax.barh([0], [value], left=left, color=color, height=0.48)
        if value > 8:
            display = (
                "FlowPilot wins"
                if label == "flowpilot"
                else "One-shot wins"
                if label == "one_shot"
                else "Ties"
            )
            ax.text(left + value / 2, 0, f"{display}\n{counts[label]}/{total}", ha="center", va="center")
        left += value
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Paired feasible cases (%)")
    ax.set_title("Lexicographic win–tie–loss outcome")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    _save(fig, figures, "04_paired_win_tie_loss")


def _build_dimension_figure(
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    figures: Path,
) -> list[dict[str, Any]]:
    lookup = {
        (row["architecture"], row["scenario_id"], row["repeat"]): row
        for row in rows
        if row["scenario_kind"] == "feasible"
    }
    statistics = []
    for index, dimension in enumerate(DIMENSION_ORDER):
        def value_fn(pair, key=dimension):
            flow = lookup[("flowpilot", pair["scenario_id"], pair["repeat"])]
            one = lookup[("one_shot", pair["scenario_id"], pair["repeat"])]
            return float(flow[f"{key}_score"]) - float(one[f"{key}_score"])

        observed, low, high = _cluster_bootstrap(
            pairs,
            value_fn,
            iterations=spec["thresholds"]["bootstrap_iterations"],
            seed=spec["thresholds"]["bootstrap_seed"] + index,
        )
        statistics.append(
            {
                "dimension": dimension,
                "mean_difference": observed,
                "ci_low": low,
                "ci_high": high,
            }
        )
    y = np.arange(len(statistics))
    effects = [row["mean_difference"] for row in statistics]
    lower = [row["mean_difference"] - row["ci_low"] for row in statistics]
    upper = [row["ci_high"] - row["mean_difference"] for row in statistics]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.errorbar(
        effects,
        y,
        xerr=[lower, upper],
        fmt="o",
        color="#2A9D8F",
        ecolor="#333333",
        capsize=4,
        markersize=8,
    )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(y, [DIMENSION_LABELS[row["dimension"]] for row in statistics])
    ax.invert_yaxis()
    ax.set_xlabel("FlowPilot minus one-shot score (points)")
    ax.set_title("Paired outcome-dimension differences\n95% family-cluster bootstrap intervals")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "05_dimension_level_effects")
    return statistics


def _violation_group(check_id: str) -> str:
    if check_id.startswith("CORE"):
        return "Core fields"
    if check_id == "INV-REACTOR":
        return "Reactor"
    if check_id == "INV-PUMP-FLOW":
        return "Pump flow"
    if check_id.startswith("INV-BPR") or check_id == "INV-PRESSURE-RATING":
        return "BPR/rating"
    if check_id in {"INV-TEMPERATURE-RATING", "INV-LIGHT"} or check_id.startswith("CHEM-"):
        return "Chemistry"
    if check_id == "NUM-V-Q-TAU":
        return "V-Q-tau"
    if "GAS-PRESSURE" in check_id or check_id == "NUM-GAS-STP":
        return "Gas flow"
    if "GAS-EQUIV" in check_id or "GAS-INLET" in check_id or "GAS-CHANNEL" in check_id:
        return "Gas equiv/timing"
    if check_id.startswith("PROC-TOPOLOGY"):
        return "Topology"
    if check_id == "INV-NO-ASSUMED-EQUIPMENT":
        return "Unlisted equipment"
    return "Other"


def _build_violation_heatmap(
    rows: list[dict[str, Any]],
    check_records: list[dict[str, Any]],
    figures: Path,
) -> None:
    groups = [
        "Core fields",
        "Reactor",
        "Pump flow",
        "BPR/rating",
        "Chemistry",
        "V-Q-tau",
        "Gas flow",
        "Gas equiv/timing",
        "Topology",
        "Unlisted equipment",
    ]
    feasible = [row for row in rows if row["scenario_kind"] == "feasible"]
    labels = sorted(
        {(row["scenario_id"], row["repeat"]) for row in feasible},
        key=lambda item: (FAMILY_ORDER.index(_family(item[0])), item[1]),
    )
    lookup: dict[tuple[str, str, int, str], list[bool]] = defaultdict(list)
    for record in check_records:
        if record["scenario_kind"] != "feasible":
            continue
        lookup[
            (
                record["architecture"],
                record["scenario_id"],
                record["repeat"],
                _violation_group(record["check_id"]),
            )
        ].append(record["passed"])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0), sharey=True)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#E76F51", "#2A9D8F"])
    for ax, architecture in zip(axes, ARCHITECTURES):
        matrix = np.ones((len(labels), len(groups)))
        for row_index, (scenario, repeat) in enumerate(labels):
            for column_index, group in enumerate(groups):
                values = lookup[(architecture, scenario, repeat, group)]
                matrix[row_index, column_index] = 1 if not values or all(values) else 0
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(groups)), groups, rotation=48, ha="right")
        ax.set_title(ARCH_LABELS[architecture])
        ax.set_yticks(
            np.arange(len(labels)),
            [f"{FAMILY_LABELS[_family(scenario)]} R{repeat}" for scenario, repeat in labels],
        )
        for row_index in range(len(labels)):
            for column_index in range(len(groups)):
                ax.text(
                    column_index,
                    row_index,
                    "✓" if matrix[row_index, column_index] else "×",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
    fig.suptitle("Deterministic check groups for feasible outcomes", fontsize=16)
    fig.tight_layout()
    _save(fig, figures, "06_critical_violation_heatmap")


def _build_reproducibility_figure(
    reproducibility: list[dict[str, Any]], figures: Path
) -> None:
    lookup = {
        (row["architecture"], row["family"]): row for row in reproducibility
    }
    x = np.arange(len(FAMILY_ORDER))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for offset, architecture in ((-width / 2, "one_shot"), (width / 2, "flowpilot")):
        quality_sd = [
            lookup[(architecture, family)]["quality_score_sd"]
            for family in FAMILY_ORDER
        ]
        agreement = [
            100 * lookup[(architecture, family)]["parameter_exact_agreement_rate"]
            for family in FAMILY_ORDER
        ]
        axes[0].bar(
            x + offset,
            quality_sd,
            width,
            color=ARCH_COLORS[architecture],
            label=ARCH_LABELS[architecture],
        )
        axes[1].bar(
            x + offset,
            agreement,
            width,
            color=ARCH_COLORS[architecture],
            label=ARCH_LABELS[architecture],
        )
    for ax in axes:
        ax.set_xticks(x, [FAMILY_LABELS[item] for item in FAMILY_ORDER])
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Quality-score SD (points)")
    axes[0].set_title("Lower variation is more repeatable")
    axes[1].set_ylabel("Parameters agreeing within 1% (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Higher agreement is more repeatable")
    axes[1].legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    fig.suptitle("Outcome reproducibility across three repeats", fontsize=16)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    _save(fig, figures, "07_reproducibility")


def _build_gas_case_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    selected = {
        row["architecture"]: row
        for row in rows
        if row["scenario_id"] == "photo_oxidation_feasible" and row["repeat"] == 1
    }
    one = selected["one_shot"]
    flow = selected["flowpilot"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    architectures = list(ARCHITECTURES)
    colors = [ARCH_COLORS[item] for item in architectures]
    display_architectures = ["One-shot", "FlowPilot"]
    axes[0, 0].bar(
        display_architectures,
        [float(one["gas_flow_sccm"] or 0), float(flow["gas_flow_sccm"] or 0)],
        color=colors,
    )
    axes[0, 0].set_title("Air flow at inlet/STP")
    axes[0, 0].set_ylabel("sccm")
    axes[0, 1].bar(
        display_architectures,
        [
            float(one["gas_flow_actual_mL_min"] or 0),
            float(flow["gas_flow_actual_mL_min"] or 0),
        ],
        color=colors,
    )
    axes[0, 1].set_title("Reported in-channel gas flow")
    axes[0, 1].set_ylabel("mL/min")
    x = np.arange(2)
    width = 0.34
    axes[1, 0].bar(
        x - width / 2,
        [
            float(one["residence_time_inlet_min"] or 0),
            float(flow["residence_time_inlet_min"] or 0),
        ],
        width,
        color="#457B9D",
        label="Inlet/STP",
    )
    axes[1, 0].bar(
        x + width / 2,
        [
            float(one["residence_time_in_channel_min"] or 0),
            float(flow["residence_time_in_channel_min"] or 0),
        ],
        width,
        color="#F4A261",
        label="In-channel",
    )
    axes[1, 0].set_xticks(x, ["One-shot", "FlowPilot"])
    axes[1, 0].set_ylabel("min")
    axes[1, 0].set_title("Reported residence-time bases")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(
        display_architectures,
        [float(one["BPR_bar"] or 0), float(flow["BPR_bar"] or 0)],
        color=colors,
    )
    axes[1, 1].set_ylabel("bar")
    axes[1, 1].set_title("Selected BPR")
    for ax in axes.flat:
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Photo-oxidation case: same protocol, different outcome", fontsize=16)
    fig.tight_layout()
    _save(fig, figures, "08_gas_liquid_case_study")


def _build_sensitivity_figure(
    sensitivity: list[dict[str, Any]], figures: Path
) -> None:
    labels = [row["weight_scheme"].replace("_", "\n") for row in sensitivity]
    flow = [row["flowpilot_wins"] for row in sensitivity]
    ties = [row["ties"] for row in sensitivity]
    one = [row["one_shot_wins"] for row in sensitivity]
    x = np.arange(len(sensitivity))
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x, flow, color=ARCH_COLORS["flowpilot"], label="FlowPilot wins")
    ax.bar(x, ties, bottom=flow, color="#B8B8B8", label="Ties")
    ax.bar(
        x,
        one,
        bottom=np.array(flow) + np.array(ties),
        color=ARCH_COLORS["one_shot"],
        label="One-shot wins",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Paired feasible outcomes (n=15)")
    ax.set_ylim(0, 16)
    ax.set_title("Win–tie–loss sensitivity to alternative quality weights")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "09_weight_sensitivity")


def _write_documents(
    output: Path,
    summary: dict[str, Any],
    model_label: str,
) -> None:
    one = summary["architectures"]["one_shot"]
    flow = summary["architectures"]["flowpilot"]
    wins = summary["paired_feasible"]["counts"]
    (output / "RESULTS.md").write_text(
        f"""# {model_label} Architecture Outcome Benchmark

## Primary results

The same {model_label} model was evaluated as a direct one-shot translator and
inside the complete FlowPilot architecture. Public input hashes matched for all
paired cells.

| Endpoint | One-shot | Full FlowPilot |
|---|---:|---:|
| Feasible executable designs | {one["feasible_executable_n"]}/15 | {flow["feasible_executable_n"]}/15 |
| Infeasible cases correctly blocked | {one["infeasible_correct_block_n"]}/15 | {flow["infeasible_correct_block_n"]}/15 |
| Mean feasible quality score | {one["mean_feasible_quality"]:.1f} | {flow["mean_feasible_quality"]:.1f} |
| Feasible non-executable outcomes | {15 - one["feasible_executable_n"]}/15 | {15 - flow["feasible_executable_n"]}/15 |

The paired lexicographic comparison produced:

- FlowPilot wins: `{wins.get("flowpilot", 0)}/15`
- Ties: `{wins.get("tie", 0)}/15`
- One-shot wins: `{wins.get("one_shot", 0)}/15`

An executable design always outranked a non-executable design. Quality score
was used only when both paired designs passed all critical deterministic checks.
The win/tie/loss counts were unchanged across the registered, equal-weight, and
four single-dimension-heavy sensitivity schemes.

## Interpretation

This pilot tests the quality of the serialized experimental outcome, not answer
style or response length. It evaluates inventory closure, arithmetic closure,
chemistry fidelity, required topology, gas bookkeeping, and operational
completeness.

The study is retrospective: the richer scoring rules were frozen before
aggregate calculation, but the saved model outputs already existed. The five
chemistry families also limit statistical power. These results should motivate,
not replace, a prospective confirmation on unseen protocols.
""",
        encoding="utf-8",
    )
    (output / "METHODS.md").write_text(
        f"""# Methods

## Controlled comparison

Both conditions used the same {model_label} model. For every scenario and
repeat, the raw protocol, objective, hard constraints, inventory, temperature,
seed, and public-input hash were identical. The sole treatment variable was
architecture: direct one-shot translation versus full FlowPilot.

The pilot reused 60 saved outputs: five chemistry families, paired feasible and
infeasible inventory states, three repeats, and two architectures.

## Endpoints

Feasible outcomes were evaluated for executable design rate. Execution required
the correct `SCREEN` disposition, a structured proposal, and passage of every
applicable critical deterministic check.

Infeasible outcomes were evaluated separately for correct `BLOCK` decisions.
They were not assigned artificial design-quality scores.

Quality scores contained four predeclared dimensions: inventory compliance
(30%), numerical closure (30%), chemistry fidelity (20%), and process
completeness (20%). The dimensions remain available separately and the paired
primary ranking is lexicographic, preventing a verbose but unsafe design from
outscoring an executable design.

## Statistics

Pairwise comparisons matched architecture outputs by scenario and repeat.
Confidence intervals resampled the five chemistry families as clusters. The
three repeated calls were not treated as independent chemistry observations.

## Reproducibility

For each family and architecture, reproducibility was summarized with median
coefficient of variation across available numerical design fields, parameter
agreement within 1%, quality-score standard deviation, and executable-repeat
rate.
""",
        encoding="utf-8",
    )
    (output / "LIMITATIONS.md").write_text(
        """# Limitations

- This is a retrospective pilot because the outputs existed before the richer
  outcome evaluator was created.
- Five independent chemistry families provide limited inferential power.
- Hidden reference envelopes encode feasibility and engineering closure, not
  experimentally optimal yield.
- The benchmark does not include blinded chemist scoring.
- The FlowPilot condition was generated after engineering corrections and used
  lexical retrieval when the external embedding service was quota-limited.
- One-shot and FlowPilot outputs have different internal schemas. Only shared
  experimental fields and required process information are scored.
- Wet-lab performance must be evaluated separately.
""",
        encoding="utf-8",
    )


def _write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    *,
    one_shot_run: Path,
    flowpilot_run: Path,
    scoring_spec_path: Path,
    output: Path,
    one_shot_condition_id: str = "qwen27b_one_shot",
    flowpilot_condition_id: str = "qwen27b_full_flowpilot",
    model_label: str = "Qwen3.6-27B",
) -> dict[str, Any]:
    spec = _read_json(scoring_spec_path)
    ARCH_LABELS["one_shot"] = f"{model_label} one-shot"
    ARCH_LABELS["flowpilot"] = f"{model_label} + FlowPilot"
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    figures = output / "figures"
    frozen = output / "frozen"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    frozen.mkdir(exist_ok=True)

    scenario_payload = _read_json(flowpilot_run / "execution_plan.json")
    scenario_map = {
        item["scenario_id"]: item for item in scenario_payload["cases"]
    }
    source_rows = {
        "one_shot": [
            item
            for item in _read_csv(one_shot_run / "run_manifest.csv")
            if item["condition_id"] == one_shot_condition_id
        ],
        "flowpilot": [
            item
            for item in _read_csv(flowpilot_run / "run_manifest.csv")
            if item["condition_id"] == flowpilot_condition_id
        ],
    }
    if any(len(items) != 30 for items in source_rows.values()):
        raise ValueError(
            "Expected exactly 30 recorded cells per architecture for "
            f"{one_shot_condition_id!r} and {flowpilot_condition_id!r}"
        )
    incomplete = [
        f"{architecture}:{item['scenario_id']}:R{item['repeat']}"
        for architecture, items in source_rows.items()
        for item in items
        if item.get("status") != "completed"
        or not (resolve_artifact_path(item["run_dir"]) / "result.json").is_file()
    ]
    if incomplete:
        raise ValueError(
            "All benchmark cells must be completed with saved results; "
            f"incomplete cells: {incomplete[:5]}"
        )

    cells: list[dict[str, Any]] = []
    check_records: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for raw in source_rows[architecture]:
            result = _read_json(
                resolve_artifact_path(raw["run_dir"]) / "result.json"
            )
            scenario = scenario_map[raw["scenario_id"]]
            row, checks = evaluate_cell(
                raw=raw,
                architecture=architecture,
                scenario=scenario,
                result=result,
                spec=spec,
            )
            cells.append(row)
            for check in checks:
                check_records.append(
                    {
                        "architecture": architecture,
                        "scenario_id": row["scenario_id"],
                        "family": row["family"],
                        "scenario_kind": row["scenario_kind"],
                        "repeat": row["repeat"],
                        **check,
                    }
                )

    hash_lookup: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in cells:
        hash_lookup[(row["scenario_id"], row["repeat"])].add(
            row["design_input_sha256"]
        )
    input_identity_passed = all(len(values) == 1 for values in hash_lookup.values())
    if not input_identity_passed:
        raise ValueError("Public design input hashes differ across architectures")

    pairs = pair_feasible_rows(
        cells, spec["thresholds"]["pairwise_quality_tie_points"]
    )
    reproducibility = build_reproducibility_rows(cells)
    sensitivity = build_weight_sensitivity_rows(
        cells, spec["thresholds"]["pairwise_quality_tie_points"]
    )
    violations = [
        {
            **record,
        }
        for record in check_records
        if not record["passed"]
    ]

    architecture_summary = {}
    for architecture in ARCHITECTURES:
        feasible = [
            row
            for row in cells
            if row["architecture"] == architecture and row["scenario_kind"] == "feasible"
        ]
        infeasible = [
            row
            for row in cells
            if row["architecture"] == architecture and row["scenario_kind"] == "infeasible"
        ]
        architecture_summary[architecture] = {
            "feasible_executable_n": sum(row["executable"] for row in feasible),
            "feasible_n": len(feasible),
            "infeasible_correct_block_n": sum(
                row["disposition_correct"] for row in infeasible
            ),
            "infeasible_n": len(infeasible),
            "mean_feasible_quality": mean(float(row["quality_score"]) for row in feasible),
            "median_feasible_quality": median(
                float(row["quality_score"]) for row in feasible
            ),
            "feasible_critical_failures": sum(
                int(row["critical_failure_count"]) for row in feasible
            ),
        }

    executable_effect = _cluster_bootstrap(
        pairs,
        lambda row: float(row["flowpilot_executable"])
        - float(row["one_shot_executable"]),
        iterations=spec["thresholds"]["bootstrap_iterations"],
        seed=spec["thresholds"]["bootstrap_seed"],
    )
    quality_effect = _cluster_bootstrap(
        pairs,
        lambda row: float(row["quality_difference_flowpilot_minus_one_shot"]),
        iterations=spec["thresholds"]["bootstrap_iterations"],
        seed=spec["thresholds"]["bootstrap_seed"] + 1,
    )
    dimension_statistics = _build_dimension_figure(
        pairs, cells, spec, figures
    )
    summary = {
        "schema_version": "flowpilot_paired_architecture_outcome_benchmark_v1.0",
        "benchmark_type": spec["benchmark_type"],
        "model": model_label,
        "condition_ids": {
            "one_shot": one_shot_condition_id,
            "flowpilot": flowpilot_condition_id,
        },
        "input_identity_passed": input_identity_passed,
        "cell_count": len(cells),
        "paired_feasible_count": len(pairs),
        "architectures": architecture_summary,
        "paired_feasible": {
            "counts": dict(Counter(row["winner"] for row in pairs)),
            "executable_rate_difference": {
                "estimate": executable_effect[0],
                "ci_low": executable_effect[1],
                "ci_high": executable_effect[2],
            },
            "quality_score_difference": {
                "estimate": quality_effect[0],
                "ci_low": quality_effect[1],
                "ci_high": quality_effect[2],
            },
        },
        "dimension_effects": dimension_statistics,
        "weight_sensitivity": sensitivity,
        "interpretation_boundary": spec["interpretation_boundary"],
    }

    _write_csv(tables / "cell_level_scores.csv", cells)
    _write_csv(tables / "check_level_results.csv", check_records)
    _write_csv(tables / "critical_and_quality_violations.csv", violations)
    _write_csv(tables / "paired_comparisons.csv", pairs)
    _write_csv(tables / "reproducibility_metrics.csv", reproducibility)
    _write_csv(tables / "dimension_effects.csv", dimension_statistics)
    _write_csv(tables / "weight_sensitivity.csv", sensitivity)
    _build_study_figure(figures, model_label)
    _build_executable_figure(cells, figures)
    _build_paired_quality_figure(pairs, figures)
    _build_win_figure(pairs, figures)
    _build_violation_heatmap(cells, check_records, figures)
    _build_reproducibility_figure(reproducibility, figures)
    _build_gas_case_figure(cells, figures)
    _build_sensitivity_figure(sensitivity, figures)
    _write_documents(output, summary, model_label)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (frozen / "scoring_spec.json").write_bytes(scoring_spec_path.read_bytes())
    (frozen / "source_manifest.json").write_text(
        json.dumps(
            {
                "one_shot_run": str(one_shot_run.resolve()),
                "flowpilot_run": str(flowpilot_run.resolve()),
                "model": model_label,
                "condition_ids": {
                    "one_shot": one_shot_condition_id,
                    "flowpilot": flowpilot_condition_id,
                },
                "one_shot_manifest_sha256": hashlib.sha256(
                    (one_shot_run / "run_manifest.csv").read_bytes()
                ).hexdigest(),
                "flowpilot_manifest_sha256": hashlib.sha256(
                    (flowpilot_run / "run_manifest.csv").read_bytes()
                ).hexdigest(),
                "source_cells": {
                    architecture: [
                        {
                            "scenario_id": item["scenario_id"],
                            "repeat": int(item["repeat"]),
                            "seed": int(item["seed"]),
                            "design_input_sha256": item["design_input_sha256"],
                            "run_dir": item["run_dir"],
                        }
                        for item in source_rows[architecture]
                    ]
                    for architecture in ARCHITECTURES
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-shot-run", type=Path, required=True)
    parser.add_argument("--flowpilot-run", type=Path, required=True)
    parser.add_argument("--scoring-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--one-shot-condition",
        default="qwen27b_one_shot",
    )
    parser.add_argument(
        "--flowpilot-condition",
        default="qwen27b_full_flowpilot",
    )
    parser.add_argument("--model-label", default="Qwen3.6-27B")
    args = parser.parse_args()
    summary = run_benchmark(
        one_shot_run=args.one_shot_run,
        flowpilot_run=args.flowpilot_run,
        scoring_spec_path=args.scoring_spec,
        output=args.output,
        one_shot_condition_id=args.one_shot_condition,
        flowpilot_condition_id=args.flowpilot_condition,
        model_label=args.model_label,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
