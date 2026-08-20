from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .cases import AblationCase, ROOT


NUMERIC_FIELDS = (
    "residence_time_min",
    "flow_rate_mL_min",
    "temperature_C",
    "concentration_M",
    "BPR_bar",
    "reactor_volume_mL",
    "tubing_ID_mm",
)

QUALITY_ASSURANCE_V2_WEIGHTS = {
    "formal_validity": 0.10,
    "engineering_integrity": 0.25,
    "process_completeness": 0.15,
    "safety_adequacy": 0.15,
    "evidence_provenance": 0.10,
    "decision_assurance": 0.20,
    "actionability_calibration": 0.05,
}

ENGINEERING_V2_WEIGHTS = {
    "numeric_completeness": 0.10,
    "geometry_consistency": 0.25,
    "pump_feasibility": 0.15,
    "tubing_feasibility": 0.15,
    "reactor_match": 0.10,
    "gas_bookkeeping": 0.25,
}

DEPLOYMENT_CAPS_V2 = {
    "schema_invalid": 0.35,
    "required_gas_bookkeeping_incomplete": 0.55,
    "geometry_inconsistent": 0.60,
    "pump_infeasible": 0.60,
    "tubing_infeasible": 0.60,
    "critical_topology_omission": 0.60,
    "critical_safety_omission": 0.60,
    "screen_required": 0.65,
}

FEATURE_KEYWORDS = {
    "liquid_pump": ("pump", "liquid feed"),
    "feed_pumps": ("pump", "feed"),
    "separate_feed_pumps": ("separate", "pump", "stream a", "stream b"),
    "stage_specific_feeds": ("stage", "feed", "stream"),
    "stage_1_feed_pumps": ("stage 1", "pump"),
    "oxygen_mass_flow_controller": ("oxygen", "mfc", "mass flow"),
    "hydrogen_mass_flow_controller": ("hydrogen", "mfc", "mass flow"),
    "chlorine_mass_flow_controller": ("chlorine", "mfc", "mass flow"),
    "CO_mass_flow_controller": ("carbon monoxide", "co ", "mfc", "mass flow"),
    "gas_liquid_mixer": ("gas-liquid", "gas liquid", "mixer", "t-mixer"),
    "mixer": ("mixer", "mixing"),
    "micromixer": ("micromixer", "micro mixer"),
    "high_intensity_mixer": ("mixer", "mixing"),
    "heated_reactor": ("heated", "temperature", "reactor"),
    "temperature_control": ("temperature", "cooling", "jacket"),
    "temperature_controlled_microreactor": ("microreactor", "temperature"),
    "cooled_reactor": ("cooled", "cooling", "-20"),
    "cooled_microreactor": ("cooled", "cooling", "microreactor"),
    "photoreactor": ("photoreactor", "irradiat", "led"),
    "back_pressure_regulator": ("back pressure", "bpr"),
    "catalyst_cartridge": ("cartridge", "packed bed", "catalyst bed"),
    "packed_bed": ("packed bed", "catalyst bed"),
    "packed_bed_or_slurry_strategy": ("packed bed", "slurry", "cstr"),
    "catalyst_retention": ("retain", "filter", "packed bed", "magnetic"),
    "collection": ("collect", "collection"),
    "collection_or_quench": ("collect", "quench"),
    "quench": ("quench",),
    "immediate_quench": ("immediate", "quench"),
    "inline_thiosulfate_quench": ("thiosulfate", "quench"),
    "quench_or_safe_collection": ("quench", "safe collection"),
    "off_gas_destruct": ("off-gas", "off gas", "destruct", "scrubber"),
    "scrubber": ("scrubber", "off-gas", "off gas"),
    "CO_detector": ("co detector", "carbon monoxide detector", "monitor"),
    "scrubber_or_vent": ("scrubber", "vent"),
    "pressure_monitoring": ("pressure monitor", "pressure sensor", "pressure"),
    "flush_strategy": ("flush", "cleaning"),
    "solids_compatible_feeding": ("slurry", "solids", "cstr", "agitated"),
    "large_bore_or_CSTR": ("large bore", "cstr", "stirred tank"),
    "stage_1_oxidation": ("stage 1", "oxidation"),
    "stage_1_reactor": ("stage 1", "reactor"),
    "interstage_addition": ("interstage", "downstream addition", "stage 2"),
    "delayed_amine_addition": ("amine", "downstream", "delayed", "stage 2"),
    "peroxide_conditioning_or_quench": ("peroxide", "quench", "condition"),
    "stage_2_amidation": ("stage 2", "amidation"),
    "stage_2_reactor": ("stage 2", "reactor"),
    "multiple_reactors": ("stage 1", "stage 2", "reactors"),
    "interstage_additions": ("interstage", "stage 2", "addition"),
    "separations_or_conditioning": ("separation", "condition", "quench"),
    "back_pressure_control": ("back pressure", "bpr"),
}

HAZARD_KEYWORDS = {
    "hydrogen": ("hydrogen", "flammable"),
    "oxygen_organic_solvent": ("oxygen", "flammab", "organic solvent"),
    "pressurized_gas_liquid": ("pressure", "bpr"),
    "carbon_monoxide": ("carbon monoxide", "co detector", "toxic gas"),
    "chlorine": ("chlorine", "toxic gas"),
    "ozone": ("ozone",),
    "ozonide": ("ozonide", "peroxide"),
    "organic_peroxide": ("peroxide", "runaway"),
    "rapid_exotherm": ("exotherm", "heat removal", "cool"),
    "strong_exotherm": ("exotherm", "heat removal", "cool"),
    "exotherm": ("exotherm", "heat removal", "cool"),
    "clogging": ("clog", "pressure"),
    "overpressure": ("overpressure", "pressure relief", "bpr"),
    "solids_management": ("solid", "filter", "packed bed", "slurry"),
    "incompatible_feeds": ("incompatible", "separate", "delayed"),
    "high_pressure": ("high pressure", "bpr", "pressure"),
    "pyrophoric_catalyst": ("pyrophoric", "wet catalyst", "inert"),
    "nitric_acid": ("nitric acid", "corros"),
    "energetic_product": ("energetic", "explos", "decomposition"),
}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _proposal(result: dict[str, Any]) -> dict[str, Any]:
    final_design = result.get("final_design")
    if isinstance(final_design, dict):
        if final_design.get("status") != "executable":
            return {}
        parameters = final_design.get("parameters")
        if isinstance(parameters, dict):
            proposal = dict(result.get("proposal") or {})
            proposal.update(parameters)
            proposal["streams"] = list(final_design.get("streams") or [])
            proposal["stage_parameters"] = list(final_design.get("stages") or [])
            return proposal
    proposal = result.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    candidate = result.get("final_design_candidate", {})
    if isinstance(candidate, dict) and isinstance(candidate.get("proposal"), dict):
        return candidate["proposal"]
    return {}


def _coverage(expected: list[str], text: str, vocabulary: dict[str, tuple[str, ...]]) -> tuple[float, dict[str, bool]]:
    checks: dict[str, bool] = {}
    for item in expected:
        terms = vocabulary.get(item, tuple(item.replace("_", " ").split()))
        checks[item] = any(term.lower() in text for term in terms)
    score = sum(checks.values()) / len(checks) if checks else 1.0
    return round(score, 4), checks


def _geometry_metrics(proposal: dict[str, Any]) -> dict[str, Any]:
    volume = _number(proposal.get("reactor_volume_mL"))
    liquid_flow = _number(proposal.get("flow_rate_mL_min"))
    tau = _number(proposal.get("residence_time_min"))
    if not volume or not liquid_flow or not tau:
        return {
            "geometry_checkable": False,
            "geometry_relative_error": None,
            "geometry_consistent_10pct": False,
        }

    gas_stp = 0.0
    gas_actual = 0.0
    gas_streams = 0
    equivalent_present = False
    streams = proposal.get("streams") or []
    if isinstance(streams, dict):
        streams = list(streams.values())
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        stream_phase = str(stream.get("phase", "")).lower()
        stream_role = str(stream.get("pump_role", "")).lower()
        stream_contents = " ".join(map(str, stream.get("contents") or [])).lower()
        stream_stp = _number(stream.get("gas_flow_sccm")) or 0.0
        stream_actual = _number(stream.get("gas_flow_actual_mL_min")) or 0.0
        is_gas = (
            stream_phase == "gas"
            or stream_stp > 0
            or stream_actual > 0
            or "mass flow" in stream_role
            or "mfc" in stream_role
            or any(name in stream_contents for name in ("oxygen", "hydrogen", "chlorine", "ozone", "carbon monoxide"))
        )
        if is_gas:
            gas_streams += 1
            gas_stp += stream_stp
            gas_actual += stream_actual
            stream_equiv = (
                _number(stream.get("molar_equiv"))
                if stream.get("molar_equiv") is not None
                else _number(stream.get("equivalents"))
            )
            equivalent_present = (
                equivalent_present
                or (stream_equiv is not None and stream_equiv > 0)
            )

    basis = str(proposal.get("residence_time_basis", "")).lower()
    if "inlet" in basis or "stp" in basis:
        expected_volume = (liquid_flow + gas_stp) * tau
    elif "channel" in basis and gas_actual > 0:
        expected_volume = (liquid_flow + gas_actual) * tau
    else:
        expected_volume = liquid_flow * tau
    error = abs(volume - expected_volume) / max(volume, 1e-9)
    return {
        "geometry_checkable": True,
        "geometry_expected_volume_mL": round(expected_volume, 6),
        "geometry_relative_error": round(error, 6),
        "geometry_consistent_10pct": error <= 0.10,
        "gas_stream_count": gas_streams,
        "gas_has_stp_flow": gas_streams == 0 or gas_stp > 0,
        "gas_has_in_channel_flow": gas_streams == 0 or gas_actual > 0,
        "gas_has_equivalents": gas_streams == 0 or equivalent_present,
    }


def _inventory_metrics(
    proposal: dict[str, Any], case: AblationCase
) -> dict[str, Any]:
    inventory = case.inventory or json.loads(
        (ROOT.parents[0] / "flora_translate" / "data" / "lab_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    flow = _number(proposal.get("flow_rate_mL_min"))
    volume = _number(proposal.get("reactor_volume_mL"))
    tubing_id = _number(proposal.get("tubing_ID_mm"))
    temperature = _number(proposal.get("temperature_C"))
    pressure = _number(proposal.get("BPR_bar")) or 0.0

    pump_by_id = {item.get("equipment_id", ""): item for item in inventory.get("pumps", [])}
    liquid_streams = [
        stream for stream in proposal.get("streams") or []
        if str(stream.get("phase") or "liquid").lower() != "gas"
    ]
    if liquid_streams and all(stream.get("pump_equipment_id") for stream in liquid_streams):
        pump_feasible = all(
            stream.get("pump_equipment_id") in pump_by_id
            and pump_by_id[stream["pump_equipment_id"]]["min_flow_rate_mL_min"]
            <= (_number(stream.get("flow_rate_mL_min")) or -1)
            <= pump_by_id[stream["pump_equipment_id"]]["max_flow_rate_mL_min"]
            and pressure <= pump_by_id[stream["pump_equipment_id"]]["max_pressure_bar"]
            for stream in liquid_streams
        )
    else:
        pump_feasible = bool(
            flow is not None
            and any(
                pump["min_flow_rate_mL_min"] <= flow <= pump["max_flow_rate_mL_min"]
                and pressure <= pump["max_pressure_bar"]
                for pump in inventory.get("pumps", [])
            )
        )
    reactor_match = bool(
        volume is not None
        and tubing_id is not None
        and any(
            abs(reactor["volume_mL"] - volume) <= 1e-3
            and abs(reactor["ID_mm"] - tubing_id) <= 1e-6
            for reactor in inventory.get("reactors", [])
        )
    )
    selected_id = str((proposal.get("inventory_selection") or {}).get("equipment_id") or "")
    selected_reactor = next(
        (
            reactor for reactor in inventory.get("reactors", [])
            if reactor.get("equipment_id") == selected_id
        ),
        None,
    )
    integrated_path = bool(
        selected_reactor
        and any(
            token in " ".join(
                str(selected_reactor.get(key) or "")
                for key in ("type", "configuration", "name")
            ).lower()
            for token in ("microchannel", "microreactor", "packed-bed", "packed bed", "integrated")
        )
    )
    tubing_feasible = integrated_path or bool(
        tubing_id is not None
        and any(
            abs(tube["ID_mm"] - tubing_id) <= 1e-6
            and (temperature is None or temperature <= tube["max_temperature_C"])
            and pressure <= tube["max_pressure_bar"]
            for tube in inventory.get("tubing", [])
        )
    )
    return {
        "inventory_pump_feasible": pump_feasible,
        "inventory_tubing_feasible": tubing_feasible,
        "inventory_exact_reactor_match": reactor_match,
    }


def _reference_metrics(case: AblationCase, proposal: dict[str, Any]) -> dict[str, Any]:
    if not case.reference_flow:
        return {
            "reference_available": False,
            "reference_accuracy_score": None,
        }
    comparisons: dict[str, Any] = {}
    normalized_errors: list[float] = []
    for field in ("residence_time_min", "flow_rate_mL_min", "reactor_volume_mL", "BPR_bar"):
        observed = _number(proposal.get(field))
        reference = _number(case.reference_flow.get(field))
        if observed is None or reference is None:
            comparisons[field] = None
            continue
        if field == "BPR_bar" and reference == 0:
            error = abs(observed - reference) / 5.0
        elif reference == 0:
            error = abs(observed - reference)
        else:
            error = abs(math.log10(max(observed, 1e-9) / reference))
        comparisons[field] = round(error, 6)
        normalized_errors.append(min(error, 2.0))

    observed_t = _number(proposal.get("temperature_C"))
    reference_t = _number(case.reference_flow.get("temperature_C"))
    if observed_t is not None and reference_t is not None:
        temp_error = abs(observed_t - reference_t)
        comparisons["temperature_C"] = round(temp_error, 6)
        normalized_errors.append(min(temp_error / 50.0, 2.0))
    else:
        comparisons["temperature_C"] = None

    score = math.exp(-sum(normalized_errors) / len(normalized_errors)) if normalized_errors else None
    return {
        "reference_available": True,
        "reference_quality": case.reference_quality,
        "reference_field_errors": comparisons,
        "reference_accuracy_score": round(score, 4) if score is not None else None,
        "reference_accuracy_primary": case.reference_quality == "machine_extracted",
    }


def _prompt_contamination(case: AblationCase, run_dir: Path) -> dict[str, Any]:
    text_parts: list[str] = []
    prompt_path = run_dir / "prompt.json"
    if prompt_path.exists():
        text_parts.append(prompt_path.read_text(encoding="utf-8").lower())
    llm_path = run_dir / "llm_events.jsonl"
    if llm_path.exists():
        for line in llm_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            text_parts.append(str(event.get("system_prompt", "")).lower())
            text_parts.append(str(event.get("user_prompt", "")).lower())
    prompt_text = "\n".join(text_parts)
    leaked_ids = [
        record_id
        for record_id in case.excluded_record_ids
        if record_id.lower() in prompt_text
    ]

    retrieved_ids: list[str] = []
    analogy_path = run_dir / "snapshots" / "analogies.json"
    if analogy_path.exists():
        analogies = json.loads(analogy_path.read_text(encoding="utf-8"))
        for analogy in analogies:
            record_id = str(
                analogy.get("record_id")
                or analogy.get("source_pdf")
                or analogy.get("metadata", {}).get("source_pdf")
                or ""
            )
            if record_id:
                retrieved_ids.append(record_id)
    excluded_lower = {value.lower() for value in case.excluded_record_ids}
    retrieved_leaks = [value for value in retrieved_ids if value.lower() in excluded_lower]
    return {
        "prompt_source_id_leaks": leaked_ids,
        "retrieval_source_leaks": retrieved_leaks,
        "leave_one_source_out_pass": not leaked_ids and not retrieved_leaks,
    }


def _find_calculated_gas_equiv(result: dict[str, Any]) -> float | None:
    calculations = (
        result.get("frozen_context", {}).get("calculations")
        if isinstance(result.get("frozen_context"), dict)
        else None
    )
    if not isinstance(calculations, dict):
        calculations = result.get("design_calculations")
    if not isinstance(calculations, dict):
        calculations = {}
    values: list[float] = []
    for key in (
        "target_gas_equiv_inlet",
        "gas_equiv_supplied",
        "o2_equiv_supplied",
    ):
        number = _number(calculations.get(key))
        if number is not None:
            values.append(number)
    for step in calculations.get("steps") or []:
        step_values = step.get("values") if isinstance(step, dict) else None
        if not isinstance(step_values, dict):
            continue
        for key in (
            "target_gas_equiv_inlet",
            "explicit_gas_equiv_inlet",
            "o2_equiv_supplied",
        ):
            number = _number(step_values.get(key))
            if number is not None:
                values.append(number)
    if not values:
        for stream in _proposal(result).get("streams") or []:
            if str(stream.get("phase") or "").lower() == "gas":
                number = _number(stream.get("molar_equiv"))
                if number is not None:
                    values.append(number)
    return max(values) if values else None


def _is_nonempty(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, set, str)):
        return bool(value)
    return value is not None


def _quality_assurance_v2(
    *,
    result: dict[str, Any],
    proposal: dict[str, Any],
    run_dir: Path,
    schema_valid: bool,
    numeric_completeness: float,
    geometry_consistent: bool,
    pump_feasible: bool,
    tubing_feasible: bool,
    reactor_match: bool,
    topology_score: float,
    safety_score: float,
    gas_bookkeeping_complete: bool,
    requires_gas: bool,
    leave_one_source_out_pass: bool,
) -> dict[str, Any]:
    engineering_checks = {
        "numeric_completeness": numeric_completeness,
        "geometry_consistency": float(geometry_consistent),
        "pump_feasibility": float(pump_feasible),
        "tubing_feasibility": float(tubing_feasible),
        "reactor_match": float(reactor_match),
        "gas_bookkeeping": float(gas_bookkeeping_complete),
    }
    engineering_integrity = sum(
        ENGINEERING_V2_WEIGHTS[name] * value
        for name, value in engineering_checks.items()
    )

    frozen = result.get("frozen_context")
    frozen = frozen if isinstance(frozen, dict) else {}
    final_candidate = result.get("final_design_candidate")
    final_candidate = final_candidate if isinstance(final_candidate, dict) else {}
    analogies = frozen.get("analogies")
    if not isinstance(analogies, list):
        analogies = result.get("analogies") or result.get("_analogies")
    analogies = analogies if isinstance(analogies, list) else []
    verified_analogies = [
        analogy
        for analogy in analogies
        if isinstance(analogy, dict)
        and str(
            analogy.get("record_id")
            or analogy.get("source_pdf")
            or analogy.get("metadata", {}).get("source_pdf")
            or ""
        ).strip()
    ]
    source_provenance = (
        min(len(verified_analogies) / 3.0, 1.0)
        if leave_one_source_out_pass
        else 0.0
    )
    reasoning = proposal.get("reasoning_per_field")
    reasoning_coverage = (
        min(len(reasoning) / 6.0, 1.0)
        if isinstance(reasoning, dict)
        else 0.0
    )
    evidence_provenance = 0.70 * source_provenance + 0.30 * reasoning_coverage

    canonical_inventory = (
        ROOT.parents[0] / "flora_translate" / "data" / "lab_inventory.json"
    ).resolve()
    inventory_path = frozen.get("inventory_path")
    inventory_trace = False
    if inventory_path:
        try:
            inventory_trace = Path(str(inventory_path)).resolve() == canonical_inventory
        except (OSError, RuntimeError):
            inventory_trace = False
    allocation = result.get("inventory_allocation")
    if isinstance(allocation, dict):
        inventory_trace = inventory_trace or bool(
            allocation.get("status") in {"complete", "complete_with_assumptions"}
            and allocation.get("inventory_sha256")
            and allocation.get("assignments")
        )
    calculation_trace = _is_nonempty(
        frozen.get("calculations") or result.get("design_calculations")
    )
    deliberation_trace = _is_nonempty(
        final_candidate.get("deliberation_log") or result.get("deliberation_log")
    )
    safety_review = _is_nonempty(
        final_candidate.get("safety_report")
        or (result.get("design_realization") or {}).get("safety_contract")
    )
    final_audit = (run_dir / "snapshots" / "stage3_5_final_audit.json").exists()
    confidence_present = bool(str(proposal.get("confidence") or "").strip())
    assurance_checks = {
        "calculation_trace": calculation_trace,
        "deliberation_trace": deliberation_trace,
        "safety_review": safety_review,
        "final_audit": final_audit,
        "inventory_provenance": inventory_trace,
        "confidence_documented": confidence_present,
    }
    decision_assurance = (
        0.25 * float(calculation_trace)
        + 0.20 * float(deliberation_trace)
        + 0.15 * float(safety_review)
        + 0.15 * float(final_audit)
        + 0.15 * float(inventory_trace)
        + 0.10 * float(confidence_present)
    )

    flags = proposal.get("safety_flags") or []
    if not isinstance(flags, list):
        flags = [flags]
    flag_text = " ".join(map(str, flags)).lower()
    screen_required = any(
        marker in flag_text
        for marker in ("screen_required", "screen required", "engine_fallback")
    )
    confidence_text = str(proposal.get("confidence") or "").lower()
    low_confidence = "low" in confidence_text or screen_required
    severe_defect = (
        not schema_valid
        or not geometry_consistent
        or not pump_feasible
        or not tubing_feasible
        or (requires_gas and not gas_bookkeeping_complete)
        or topology_score < 0.50
        or safety_score < 0.50
    )
    if severe_defect:
        uncertainty_calibrated = float(low_confidence)
    elif low_confidence:
        uncertainty_calibrated = 0.50
    else:
        uncertainty_calibrated = 1.0
    actionability_calibration = (
        0.60 * float(not screen_required) + 0.40 * uncertainty_calibrated
    )

    dimensions = {
        "formal_validity": float(schema_valid),
        "engineering_integrity": engineering_integrity,
        "process_completeness": topology_score,
        "safety_adequacy": safety_score,
        "evidence_provenance": evidence_provenance,
        "decision_assurance": decision_assurance,
        "actionability_calibration": actionability_calibration,
    }
    quality_score = sum(
        QUALITY_ASSURANCE_V2_WEIGHTS[name] * value
        for name, value in dimensions.items()
    )

    deployment_gate_reasons: list[str] = []
    if not schema_valid:
        deployment_gate_reasons.append("schema_invalid")
    if requires_gas and not gas_bookkeeping_complete:
        deployment_gate_reasons.append("required_gas_bookkeeping_incomplete")
    if not geometry_consistent:
        deployment_gate_reasons.append("geometry_inconsistent")
    if not pump_feasible:
        deployment_gate_reasons.append("pump_infeasible")
    if not tubing_feasible:
        deployment_gate_reasons.append("tubing_infeasible")
    if topology_score < 0.50:
        deployment_gate_reasons.append("critical_topology_omission")
    if safety_score < 0.50:
        deployment_gate_reasons.append("critical_safety_omission")
    if screen_required:
        deployment_gate_reasons.append("screen_required")
    deployment_cap = min(
        (DEPLOYMENT_CAPS_V2[reason] for reason in deployment_gate_reasons),
        default=1.0,
    )

    return {
        "quality_assurance_dimensions_v2": {
            name: round(value, 4) for name, value in dimensions.items()
        },
        "quality_assurance_score_v2": round(quality_score, 4),
        "deployment_readiness_score_v2": round(
            min(quality_score, deployment_cap),
            4,
        ),
        "deployment_ready_v2": not deployment_gate_reasons,
        "deployment_gate_reasons_v2": deployment_gate_reasons,
        "deployment_gate_count_v2": len(deployment_gate_reasons),
        "deployment_gate_flags_v2": {
            reason: reason in deployment_gate_reasons
            for reason in DEPLOYMENT_CAPS_V2
        },
        "uncertainty_calibrated_v2": round(uncertainty_calibrated, 4),
        "screen_required_v2": screen_required,
        "evidence_source_count_v2": len(verified_analogies),
        "assurance_checks_v2": assurance_checks,
        "quality_assurance_weights_v2": QUALITY_ASSURANCE_V2_WEIGHTS,
        "engineering_weights_v2": ENGINEERING_V2_WEIGHTS,
    }


def score_run(
    case: AblationCase,
    result: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    proposal = _proposal(result)
    present = {
        field: _number(proposal.get(field)) is not None
        for field in NUMERIC_FIELDS
    }
    numeric_completeness = sum(present.values()) / len(present)
    output_text = json.dumps(proposal, ensure_ascii=False).lower()
    expected = case.expected_features
    topology_score, topology_checks = _coverage(
        expected.get("topology", []),
        output_text,
        FEATURE_KEYWORDS,
    )
    safety_score, safety_checks = _coverage(
        expected.get("hazards", []),
        output_text,
        HAZARD_KEYWORDS,
    )
    geometry = _geometry_metrics(proposal)
    inventory = _inventory_metrics(proposal, case)
    reference = _reference_metrics(case, proposal)
    contamination = _prompt_contamination(case, run_dir)

    requires_gas = "gas" in str(expected.get("phase_regime", "")).lower()
    calculated_gas_equiv = _find_calculated_gas_equiv(result)
    gas_fields_ok = all(
        geometry.get(field, True)
        for field in (
            "gas_has_stp_flow",
            "gas_has_in_channel_flow",
            "gas_has_equivalents",
        )
    )
    if requires_gas:
        gas_fields_ok = gas_fields_ok and geometry.get("gas_stream_count", 0) > 0
        if calculated_gas_equiv is not None:
            gas_fields_ok = gas_fields_ok and calculated_gas_equiv > 0
    unexpected_gas_stream = (
        not requires_gas and geometry.get("gas_stream_count", 0) > 0
    )
    gas_fields_ok = gas_fields_ok and not unexpected_gas_stream
    deterministic_score = sum(
        (
            numeric_completeness,
            float(geometry.get("geometry_consistent_10pct", False)),
            float(inventory["inventory_pump_feasible"]),
            float(inventory["inventory_tubing_feasible"]),
            topology_score,
            safety_score,
            float(gas_fields_ok),
        )
    ) / 7.0

    schema_valid = bool(result.get("schema_valid", bool(proposal)))
    quality_assurance = _quality_assurance_v2(
        result=result,
        proposal=proposal,
        run_dir=run_dir,
        schema_valid=schema_valid,
        numeric_completeness=numeric_completeness,
        geometry_consistent=bool(geometry.get("geometry_consistent_10pct")),
        pump_feasible=bool(inventory["inventory_pump_feasible"]),
        tubing_feasible=bool(inventory["inventory_tubing_feasible"]),
        reactor_match=bool(inventory["inventory_exact_reactor_match"]),
        topology_score=topology_score,
        safety_score=safety_score,
        gas_bookkeeping_complete=gas_fields_ok,
        requires_gas=requires_gas,
        leave_one_source_out_pass=bool(
            contamination["leave_one_source_out_pass"]
        ),
    )

    return {
        "case_id": case.case_id,
        "suite_id": case.suite_id,
        "category": case.category,
        "schema_valid": schema_valid,
        "numeric_field_presence": present,
        "numeric_completeness": round(numeric_completeness, 4),
        **geometry,
        **inventory,
        "topology_coverage": topology_score,
        "topology_checks": topology_checks,
        "safety_coverage": safety_score,
        "safety_checks": safety_checks,
        "gas_bookkeeping_complete": gas_fields_ok,
        "gas_required_by_case": requires_gas,
        "unexpected_gas_stream": unexpected_gas_stream,
        "calculated_gas_equiv_inlet": calculated_gas_equiv,
        **reference,
        **contamination,
        **quality_assurance,
        "deterministic_composite_score": round(deterministic_score, 4),
        "proposal_summary": {
            field: proposal.get(field)
            for field in NUMERIC_FIELDS
        },
    }
