from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .cases import AblationCase


GENERAL_REL_TOL = 0.05
GAS_REL_TOL = 0.10
STP_TEMPERATURE_K = 273.15
STP_PRESSURE_BAR = 1.01325
STP_MOLAR_VOLUME_ML_MOL = 22414.0


CRITERIA = (
    ("NG-01", "chemistry", True, "Target transformation and product identity preserved"),
    ("NG-02", "chemistry", True, "Required materials accounted for"),
    ("NG-03", "chemistry", False, "Feed composition and stoichiometry explicit"),
    ("NG-04", "process", True, "Operations and additions in the correct order"),
    ("NG-05", "chemistry", True, "Reaction modality and phase regime correct"),
    ("NG-06", "source_fidelity", True, "No contradiction with fixed protocol conditions"),
    ("NG-07", "source_fidelity", False, "Recommendations identify calculation or inference basis"),
    ("NG-08", "completeness", True, "Mandatory output fields machine-readable"),
    ("NG-09", "calculation", True, "Units, signs, ranges, and pressure basis valid"),
    ("NG-10", "calculation", True, "Total liquid flow closes"),
    ("NG-11", "calculation", True, "Component molar flow reproducible"),
    ("NG-12", "calculation", True, "Liquid reagent equivalents close"),
    ("NG-13", "calculation", True, "Reactor volume, flow, and residence time close"),
    ("NG-14", "calculation", True, "Reactor geometry or inventory volume basis closes"),
    ("NG-15", "calculation", True, "STP gas flow converts to in-channel flow"),
    ("NG-16", "calculation", True, "Gas equivalents close"),
    ("NG-17", "calculation", True, "Gas residence-time bases are correct and distinct"),
    ("NG-18", "calculation", False, "Throughput reproducible"),
    ("NG-19", "inventory", True, "Selected equipment exists and is within limits"),
    ("NG-20", "process", True, "Topology and essential safety controls complete"),
)


@dataclass(frozen=True)
class AuditCheck:
    criterion_id: str
    domain: str
    critical: bool
    status: str
    observed: str
    expected: str
    evidence_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "domain": self.domain,
            "critical": self.critical,
            "status": self.status,
            "error": self.status == "FAIL",
            "observed": self.observed,
            "expected": self.expected,
            "evidence_path": self.evidence_path,
        }


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _close(a: float | None, b: float | None, rel: float = GENERAL_REL_TOL) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def _normalized_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _contains_group(text: str, group: Iterable[str]) -> bool:
    return any(term.lower() in text for term in group)


def _final_payload(result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    proposal = dict(result.get("proposal") or {})
    final = result.get("final_design")
    if isinstance(final, dict) and final.get("status") == "executable":
        proposal.update(final.get("parameters") or {})
        stage_payload = final.get("stages") or proposal.get("stage_parameters") or []
        streams = list(final.get("streams") or proposal.get("streams") or [])
    else:
        # One-shot runs retain the model-native object. Audit that output directly;
        # canonicalization loss is not a chemistry or engineering error by the model.
        if isinstance(result.get("raw_proposal"), dict):
            raw_proposal = result["raw_proposal"]
            scalar_contract_fields = {
                "residence_time_min",
                "flow_rate_mL_min",
                "temperature_C",
                "concentration_M",
                "BPR_bar",
                "reactor_volume_mL",
                "tubing_ID_mm",
            }
            proposal.update({
                key: value
                for key, value in raw_proposal.items()
                if key not in scalar_contract_fields
            })
        stage_payload = proposal.get("stage_parameters") or []
        streams = list(proposal.get("streams") or [])
    if isinstance(stage_payload, dict):
        stages = [value for value in stage_payload.values() if isinstance(value, dict)]
    else:
        stages = [value for value in stage_payload if isinstance(value, dict)]
    normalized_streams: list[dict[str, Any]] = []
    for raw_stream in streams:
        stream = dict(raw_stream)
        composition = stream.get("composition")
        if isinstance(composition, dict):
            stream.setdefault("phase", stream.get("type") or "liquid")
            stream.setdefault("solvent", composition.get("solvent"))
            stream.setdefault("concentration_M", composition.get("concentration_M"))
            if not stream.get("contents"):
                stream["contents"] = [
                    reagent.get("name")
                    for reagent in composition.get("reagents") or []
                    if isinstance(reagent, dict) and reagent.get("name")
                ]
        normalized_streams.append(stream)
    return proposal, stages, normalized_streams


def _stage_value(stage: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(stage.get(key))
        if value is not None:
            return value
    parameters = stage.get("parameters")
    if isinstance(parameters, dict):
        for key in keys:
            value = _number(parameters.get(key))
            if value is not None:
                return value
    return None


def _reactive_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        stage
        for stage in stages
        if _stage_value(stage, "reactor_volume_mL", "V_R_mL", "volume_mL") is not None
        and _stage_value(
            stage,
            "cumulative_flow_mL_min",
            "Q_liquid_mL_min",
            "flow_rate_mL_min",
        ) is not None
        and _stage_value(stage, "residence_time_min") is not None
    ]


def _inventory_ids(inventory: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for value in inventory.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and item.get("equipment_id"):
                ids.add(str(item["equipment_id"]))
    return ids


def _explicit_equipment_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.endswith(("equipment_id", "reactor_id", "mixer_id")) and isinstance(nested, str):
                ids.add(nested)
            elif key.endswith("equipment_ids") and isinstance(nested, list):
                ids.update(str(item) for item in nested if item)
            ids.update(_explicit_equipment_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            ids.update(_explicit_equipment_ids(nested))
    return {item for item in ids if item and item.lower() not in {"none", "null"}}


def _topology_presence(result: dict[str, Any], proposal: dict[str, Any], streams: list[dict[str, Any]]) -> set[str]:
    text = _normalized_text({
        "proposal": proposal,
        "unit_operations": result.get("unit_operations"),
        "process_topology": result.get("process_topology"),
        "inventory_allocation": result.get("inventory_allocation"),
        "instrument_manifest": result.get("instrument_manifest"),
    })
    liquid_count = sum(str(stream.get("phase") or "liquid").lower() == "liquid" for stream in streams)
    found: set[str] = set()
    vocabulary = {
        "pump": ("pump", "liquid feed"),
        "three_pumps": ("pump_ms_a", "pump_ms_b", "pump_ms_c"),
        "gas_mfc": ("mfc", "mass-flow controller", "mass flow controller"),
        "gas_liquid_mixer": ("gas-liquid", "gas liquid", "hydrogen-service t-mixer"),
        "packed_bed": ("packed-bed", "packed bed", "catcart"),
        "temperature_control": ("heater", "water bath", "oil bath", "temperature control"),
        "bpr": ("back pressure", "bpr"),
        "separator": ("separator",),
        "collection": ("collect", "collector"),
        "mixer_1": ("mixer_ms_1", "mixer 1", "stage 1 t-mixer"),
        "reactor_1": ("reactor_ms_196", "reactor 1", "stage 1 oxidation"),
        "mixer_2": ("mixer_ms_2", "mixer 2", "interstage t-mixer"),
        "reactor_2": ("reactor_ms_1308", "reactor 2", "stage 2 oxidative"),
    }
    for feature, terms in vocabulary.items():
        if feature == "three_pumps":
            if liquid_count >= 3 or all(term in text for term in terms):
                found.add(feature)
        elif any(term in text for term in terms):
            found.add(feature)
    return found


def _inventory_feasibility(
    case: AblationCase,
    result: dict[str, Any],
    proposal: dict[str, Any],
    stages: list[dict[str, Any]],
    streams: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    inventory = case.inventory
    errors: list[str] = []
    known_ids = _inventory_ids(inventory)
    explicit_ids = _explicit_equipment_ids({
        "proposal": proposal,
        "stages": stages,
        "streams": streams,
        "allocation": result.get("inventory_allocation"),
    })
    unknown = sorted(explicit_ids - known_ids)
    if unknown:
        errors.append(f"unknown equipment IDs: {', '.join(unknown)}")

    pumps = inventory.get("pumps") or []
    for stream in streams:
        if str(stream.get("phase") or "liquid").lower() != "liquid":
            continue
        flow = _number(stream.get("flow_rate_mL_min"))
        if flow is None:
            errors.append(f"stream {stream.get('stream_label', '?')} has no liquid flow")
            continue
        requested = str(stream.get("pump_equipment_id") or "")
        candidates = [pump for pump in pumps if not requested or pump.get("equipment_id") == requested]
        feasible = any(
            _number(pump.get("min_flow_rate_mL_min")) is not None
            and _number(pump.get("max_flow_rate_mL_min")) is not None
            and _number(pump.get("min_flow_rate_mL_min")) <= flow <= _number(pump.get("max_flow_rate_mL_min"))
            for pump in candidates
        )
        if not feasible:
            errors.append(f"stream {stream.get('stream_label', '?')} flow {flow:g} mL/min is outside pump limits")

    reactors = inventory.get("reactors") or []
    reactive = _reactive_stages(stages)
    if reactive:
        used: set[str] = set()
        for index, stage in enumerate(reactive, 1):
            volume = _stage_value(stage, "reactor_volume_mL", "V_R_mL", "volume_mL")
            temperature = _stage_value(stage, "temperature_C")
            requested = str(
                stage.get("reactor_equipment_id")
                or stage.get("reactor_id")
                or stage.get("reactor")
                or ""
            )
            matches = []
            for reactor in reactors:
                if requested and reactor.get("equipment_id") != requested:
                    continue
                if not _close(volume, _number(reactor.get("volume_mL")), 0.001):
                    continue
                allowed = reactor.get("allowed_temperatures_C") or []
                if temperature is not None and allowed and not any(_close(temperature, _number(item), 0.001) for item in allowed):
                    continue
                minimum = _number(reactor.get("min_temperature_C"))
                maximum = _number(reactor.get("max_temperature_C"))
                if temperature is not None and minimum is not None and temperature < minimum:
                    continue
                if temperature is not None and maximum is not None and temperature > maximum:
                    continue
                if reactor.get("equipment_id") not in used:
                    matches.append(reactor)
            if not matches:
                errors.append(f"stage {index} reactor volume/temperature is not supported by inventory")
            else:
                used.add(str(matches[0].get("equipment_id")))
    else:
        volume = _number(proposal.get("reactor_volume_mL"))
        tubing_id = _number(proposal.get("tubing_ID_mm"))
        temperature = _number(proposal.get("temperature_C"))
        match = any(
            _close(volume, _number(reactor.get("volume_mL")), 0.001)
            and _close(tubing_id, _number(reactor.get("ID_mm")), 0.001)
            and (temperature is None or _number(reactor.get("min_temperature_C")) is None or temperature >= _number(reactor.get("min_temperature_C")))
            and (temperature is None or _number(reactor.get("max_temperature_C")) is None or temperature <= _number(reactor.get("max_temperature_C")))
            for reactor in reactors
        )
        if not match:
            errors.append("top-level reactor volume/ID/temperature does not match inventory")

    pressure = _number(proposal.get("BPR_bar"))
    available = [_number(value) for value in inventory.get("BPR_available") or []]
    available = [value for value in available if value is not None]
    if available and (pressure is None or not any(_close(pressure, value, 0.001) for value in available)):
        errors.append(f"BPR {pressure!r} bar is not an available setpoint")
    if not available and pressure not in (None, 0.0):
        errors.append(f"BPR {pressure:g} bar was specified but no BPR is available")

    if case.expected_features.get("gas_required") or "gas" in str(case.expected_features.get("phase_regime", "")):
        gas_streams = [stream for stream in streams if str(stream.get("phase") or "").lower() == "gas"]
        mfcs = [item for item in inventory.get("gas_hardware") or [] if str(item.get("type") or "").lower() == "mfc"]
        for stream in gas_streams:
            flow = _number(stream.get("gas_flow_sccm"))
            if flow is None or not any(
                _number(mfc.get("min_flow_sccm")) <= flow <= _number(mfc.get("max_flow_sccm"))
                for mfc in mfcs
                if _number(mfc.get("min_flow_sccm")) is not None and _number(mfc.get("max_flow_sccm")) is not None
            ):
                errors.append(f"gas flow {flow!r} sccm is outside available MFC limits")

    allocation = result.get("inventory_allocation")
    if isinstance(allocation, dict) and allocation.get("status") not in {"complete", "complete_with_assumptions"}:
        errors.append(f"inventory allocation status={allocation.get('status')}")
    return not errors, errors


def audit_result(
    case: AblationCase,
    result: dict[str, Any],
    expectations: dict[str, Any],
) -> dict[str, Any]:
    proposal, stages, streams = _final_payload(result)
    native_text = _normalized_text({
        "proposal": proposal,
        "raw_proposal": result.get("raw_proposal"),
        "stages": stages,
        "streams": streams,
        "chemistry_plan": result.get("chemistry_plan"),
        "safety_report": result.get("safety_report"),
        "final_safety": (result.get("final_design") or {}).get("safety_controls") if isinstance(result.get("final_design"), dict) else None,
    })
    criteria_meta = {cid: (domain, critical, question) for cid, domain, critical, question in CRITERIA}
    checks: list[AuditCheck] = []

    def add(cid: str, passed: bool | None, observed: str, expected: str, path: str) -> None:
        domain, critical, _ = criteria_meta[cid]
        status = "NOT_APPLICABLE" if passed is None else "PASS" if passed else "FAIL"
        checks.append(AuditCheck(cid, domain, critical, status, observed, expected, path))

    identity_missing = [group for group in expectations["identity_term_groups"] if not _contains_group(native_text, group)]
    add("NG-01", not identity_missing, f"missing groups={identity_missing}", "All target identity term groups present", "final proposal and chemistry plan")

    material_missing = [group for group in expectations["required_material_term_groups"] if not _contains_group(native_text, group)]
    add("NG-02", not material_missing, f"missing groups={material_missing}", "All protocol-required materials present", "final proposal and chemistry plan")

    liquid_streams = [stream for stream in streams if str(stream.get("phase") or "liquid").lower() == "liquid"]
    feed_explicit = bool(liquid_streams) and all(
        _number(stream.get("flow_rate_mL_min")) is not None
        and (stream.get("contents") or stream.get("solvent"))
        for stream in liquid_streams
    )
    add("NG-03", feed_explicit, f"explicit liquid streams={sum(bool(s.get('contents') or s.get('solvent')) for s in liquid_streams)}/{len(liquid_streams)}", "Every liquid feed has contents and flow", "final streams")

    sequence_ok = True
    sequence_details: list[str] = []
    topology_found = _topology_presence(result, proposal, streams)
    required_intro = expectations.get("required_stream_introduction_stage") or {}
    if required_intro:
        by_label = {str(stream.get("stream_label") or "").upper(): stream for stream in streams}
        for label, expected_stage in required_intro.items():
            observed_stage = _number(by_label.get(label, {}).get("introduction_stage"))
            if observed_stage is None:
                # Older valid contracts may express sequencing in topology/reasoning
                # without duplicating it on every stream object.
                if label in {"A", "B"} and expected_stage == 1:
                    represented = "mixer_1" in topology_found and "reactor_1" in topology_found
                elif label == "C" and expected_stage == 2:
                    represented = "mixer_2" in topology_found and "reactor_2" in topology_found
                else:
                    represented = False
                if not represented:
                    sequence_ok = False
                    sequence_details.append(f"{label}: introduction stage omitted")
            elif observed_stage != float(expected_stage):
                sequence_ok = False
                sequence_details.append(f"{label}: stage {observed_stage!r}, expected {expected_stage}")
    topology_missing = sorted(set(expectations["required_topology"]) - topology_found)
    if not required_intro:
        sequence_ok = bool(proposal.get("pre_reactor_steps")) and bool(proposal.get("post_reactor_steps"))
        if not sequence_ok:
            sequence_details.append("pre-reactor or post-reactor sequence is absent")
    add("NG-04", sequence_ok, f"sequence errors={sequence_details}", "Required operation order and stream introduction stages", "final stages, streams, and topology")

    gas_streams = [stream for stream in streams if str(stream.get("phase") or "").lower() == "gas"]
    gas_required = bool(expectations.get("gas_required"))
    modality_ok = bool(gas_streams) == gas_required
    if expectations.get("packed_bed"):
        modality_ok = modality_ok and "packed_bed" in topology_found
    add("NG-05", modality_ok, f"gas streams={len(gas_streams)}; packed bed={'packed_bed' in topology_found}", f"gas_required={gas_required}; packed_bed={expectations.get('packed_bed')}", "final streams and topology")

    required_temps = list(expectations.get("required_stage_temperatures_C") or [])
    observed_temps = [
        value for value in (_stage_value(stage, "temperature_C") for stage in _reactive_stages(stages)) if value is not None
    ]
    if not observed_temps and _number(proposal.get("temperature_C")) is not None:
        observed_temps = [_number(proposal.get("temperature_C"))]
    temps_ok = len(observed_temps) >= len(required_temps) and all(
        any(_close(required, observed, 0.001) for observed in observed_temps)
        for required in required_temps
    )
    pressure = _number(proposal.get("BPR_bar"))
    required_pressure = _number(expectations.get("required_pressure_bar"))
    protocol_ok = temps_ok and _close(pressure, required_pressure, 0.001)
    add("NG-06", protocol_ok, f"temperatures={observed_temps}; BPR={pressure}", f"temperatures={required_temps}; BPR={required_pressure}", "final parameters and stages")

    reasoning = proposal.get("reasoning_per_field")
    provenance_ok = isinstance(reasoning, dict) and len(reasoning) >= 3
    add("NG-07", provenance_ok, f"reasoned fields={len(reasoning) if isinstance(reasoning, dict) else 0}", "At least three field-level calculation/inference rationales", "proposal.reasoning_per_field")

    required_fields = ("residence_time_min", "flow_rate_mL_min", "concentration_M", "BPR_bar", "reactor_volume_mL", "tubing_ID_mm")
    missing_fields = [field for field in required_fields if _number(proposal.get(field)) is None]
    if required_temps and not observed_temps:
        missing_fields.append("temperature_C or stage temperatures")
    add("NG-08", bool(result.get("schema_valid", bool(proposal))) and not missing_fields and bool(streams), f"missing={missing_fields}; streams={len(streams)}", "Valid structured output with mandatory fields and streams", "result.schema_valid and final proposal")

    numeric = {
        "tau": _number(proposal.get("residence_time_min")),
        "flow": _number(proposal.get("flow_rate_mL_min")),
        "concentration": _number(proposal.get("concentration_M")),
        "volume": _number(proposal.get("reactor_volume_mL")),
        "ID": _number(proposal.get("tubing_ID_mm")),
        "BPR": pressure,
    }
    physical_ok = all(numeric[key] is not None and numeric[key] > 0 for key in ("tau", "flow", "concentration", "volume", "ID")) and pressure is not None and pressure >= 0
    pressure_basis_ok = not gas_required or any(term in native_text for term in ("absolute", "gauge", "bar abs", "bar(a)"))
    add("NG-09", physical_ok and pressure_basis_ok, f"values={numeric}; gas pressure basis={pressure_basis_ok}", "Finite positive run values, nonnegative pressure, explicit gas pressure basis", "final proposal")

    liquid_flow_sum = sum(_number(stream.get("flow_rate_mL_min")) or 0.0 for stream in liquid_streams)
    top_flow = _number(proposal.get("flow_rate_mL_min"))
    add("NG-10", _close(liquid_flow_sum, top_flow), f"sum liquid streams={liquid_flow_sum:.6g}; outlet/top flow={top_flow!r}", "Relative difference <= 5%", "final streams and parameters")

    molar_flows = [
        (_number(stream.get("concentration_M")), _number(stream.get("flow_rate_mL_min")))
        for stream in liquid_streams
    ]
    molar_ok = bool(molar_flows) and all(c is not None and c >= 0 and q is not None and q > 0 for c, q in molar_flows)
    add("NG-11", molar_ok, f"C,Q pairs={molar_flows}", "Each reactive liquid stream has finite concentration and flow", "final streams")

    required_equiv = expectations.get("required_equivalents") or []
    if required_equiv:
        reported_equiv = [
            _number(stream.get("molar_equiv"))
            for stream in streams
            if _number(stream.get("molar_equiv")) is not None
        ]
        for stream in streams:
            composition = stream.get("composition")
            if not isinstance(composition, dict):
                continue
            reported_equiv.extend(
                value
                for value in (
                    _number(reagent.get("equiv"))
                    for reagent in composition.get("reagents") or []
                    if isinstance(reagent, dict)
                )
                if value is not None
            )
        def explicit_equivalent(required: float) -> bool:
            value = re.escape(f"{required:g}")
            decimal = re.escape(f"{required:.1f}")
            return bool(re.search(rf"\b(?:{value}|{decimal})\s*(?:equiv|equivalent)", native_text))

        equiv_ok = all(
            any(_close(value, required, GENERAL_REL_TOL) for value in reported_equiv)
            or explicit_equivalent(required)
            for required in required_equiv
        )
        add("NG-12", equiv_ok, f"reported equivalents={reported_equiv}", f"required equivalents={required_equiv}", "final streams")
    else:
        add("NG-12", None, "No fixed liquid-reagent equivalent audited", "Not applicable", "frozen expectations")

    reactive = _reactive_stages(stages)
    closure_errors: list[str] = []
    if expectations.get("multistage"):
        if len(reactive) != len(required_temps):
            closure_errors.append(f"reactive stages={len(reactive)}, expected={len(required_temps)}")
        for index, stage in enumerate(reactive, 1):
            volume = _stage_value(stage, "reactor_volume_mL", "V_R_mL", "volume_mL")
            flow = _stage_value(stage, "cumulative_flow_mL_min", "Q_liquid_mL_min", "flow_rate_mL_min")
            tau = _stage_value(stage, "residence_time_min")
            expected_tau = volume / flow if volume is not None and flow else None
            if not _close(tau, expected_tau):
                closure_errors.append(f"stage {index}: tau={tau}, V/Q={expected_tau}")
        stage_tau_sum = sum(_stage_value(stage, "residence_time_min") or 0.0 for stage in reactive)
        if not _close(_number(proposal.get("residence_time_min")), stage_tau_sum):
            closure_errors.append(f"total tau={proposal.get('residence_time_min')}, sum stages={stage_tau_sum}")
    else:
        volume = _number(proposal.get("reactor_volume_mL"))
        flow = _number(proposal.get("flow_rate_mL_min"))
        tau = _number(proposal.get("residence_time_min"))
        expected_tau = volume / flow if volume is not None and flow else None
        if not _close(tau, expected_tau):
            closure_errors.append(f"tau={tau}, V/Q_liquid={expected_tau}")
    add("NG-13", not closure_errors, f"closure errors={closure_errors}", "Every stage and total residence time within 5% of V/Q basis", "final stages and parameters")

    inventory_volumes = sorted(_number(item.get("volume_mL")) for item in case.inventory.get("reactors") or [] if _number(item.get("volume_mL")) is not None)
    reported_volumes = sorted(_stage_value(stage, "reactor_volume_mL", "V_R_mL", "volume_mL") for stage in reactive) if reactive else [_number(proposal.get("reactor_volume_mL"))]
    geometry_ok = len(reported_volumes) == len(inventory_volumes) and all(_close(a, b, 0.001) for a, b in zip(reported_volumes, inventory_volumes))
    add("NG-14", geometry_ok, f"reported reactor volumes={reported_volumes}", f"inventory reactor volumes={inventory_volumes}", "final stages/parameters and frozen inventory")

    if gas_required:
        gas = gas_streams[0] if gas_streams else {}
        q_stp = _number(gas.get("gas_flow_sccm"))
        q_channel = _number(gas.get("gas_flow_actual_mL_min"))
        temperature = observed_temps[0] if observed_temps else _number(proposal.get("temperature_C"))
        pressure_abs = pressure
        expected_channel = q_stp * ((temperature + 273.15) / STP_TEMPERATURE_K) * (STP_PRESSURE_BAR / pressure_abs) if q_stp is not None and temperature is not None and pressure_abs else None
        add("NG-15", _close(q_channel, expected_channel, GAS_REL_TOL), f"reported={q_channel}; ideal-gas expected={expected_channel}", "Relative difference <= 10%", "gas stream, temperature, and pressure")

        substrate = liquid_streams[0] if liquid_streams else {}
        substrate_rate = (_number(substrate.get("concentration_M")) or 0.0) * (_number(substrate.get("flow_rate_mL_min")) or 0.0)
        expected_equiv = (q_stp / (STP_MOLAR_VOLUME_ML_MOL / 1000.0)) / substrate_rate if q_stp is not None and substrate_rate > 0 else None
        reported_equiv = _number(gas.get("molar_equiv"))
        add("NG-16", _close(reported_equiv, expected_equiv, GAS_REL_TOL), f"reported={reported_equiv}; calculated={expected_equiv}", "Relative difference <= 10%", "gas and limiting-liquid streams")

        volume = _number(proposal.get("reactor_volume_mL"))
        q_liquid = _number(proposal.get("flow_rate_mL_min"))
        tau_inlet = _number(proposal.get("residence_time_inlet_min"))
        tau_channel = _number(proposal.get("residence_time_in_channel_min"))
        expected_inlet = volume / (q_liquid + q_stp) if volume is not None and q_liquid is not None and q_stp is not None else None
        expected_channel_tau = volume / (q_liquid + q_channel) if volume is not None and q_liquid is not None and q_channel is not None else None
        gas_tau_ok = _close(tau_inlet, expected_inlet, GAS_REL_TOL) and _close(tau_channel, expected_channel_tau, GAS_REL_TOL)
        add("NG-17", gas_tau_ok, f"inlet reported/calculated={tau_inlet}/{expected_inlet}; channel={tau_channel}/{expected_channel_tau}", "Both residence-time bases within 10% and explicitly separated", "final parameters and gas stream")
    else:
        for cid in ("NG-15", "NG-16", "NG-17"):
            add(cid, None, "No gas reagent in this case", "Not applicable", "frozen expectations")

    throughput = (_number(proposal.get("concentration_M")) or 0.0) * (_number(proposal.get("flow_rate_mL_min")) or 0.0) * 60.0
    add("NG-18", throughput > 0, f"calculated limiting-feed throughput={throughput:.6g} mmol/h", "Positive throughput reproducible from C x Q x 60", "final parameters")

    inventory_ok, inventory_errors = _inventory_feasibility(case, result, proposal, stages, streams)
    add("NG-19", inventory_ok, f"inventory errors={inventory_errors}", "All explicit equipment and operating settings supported by frozen inventory", "final design and frozen inventory")

    safety_missing = [group for group in expectations.get("required_safety_term_groups") or [] if not _contains_group(native_text, group)]
    topology_missing = sorted(set(expectations["required_topology"]) - topology_found)
    add("NG-20", not safety_missing and not topology_missing, f"missing safety groups={safety_missing}; missing topology={topology_missing}", "Complete required topology and executable safety controls", "final proposal, topology, and safety report")

    rows = [check.as_dict() for check in checks]
    applicable = [row for row in rows if row["status"] != "NOT_APPLICABLE"]
    failures = [row for row in applicable if row["status"] == "FAIL"]
    critical_failures = [row for row in failures if row["critical"]]
    return {
        "schema_version": "flowpilot_newgen_error_audit_v1.0",
        "case_id": case.case_id,
        "criteria": rows,
        "applicable_criteria": len(applicable),
        "total_errors": len(failures),
        "critical_errors": len(critical_failures),
        "error_rate": round(len(failures) / len(applicable), 6) if applicable else 0.0,
        "critical_error_free": not critical_failures,
        "failed_criterion_ids": [row["criterion_id"] for row in failures],
        "critical_failed_criterion_ids": [row["criterion_id"] for row in critical_failures],
        "tolerances": {
            "general_relative": GENERAL_REL_TOL,
            "gas_relative": GAS_REL_TOL,
        },
    }


def load_expectations(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"]
