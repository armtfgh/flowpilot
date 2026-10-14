"""Current arithmetic and historical scientific rationale have different authority.

No model statement is graded, rewritten, or silently promoted to a checked fact.
Original quantitative explanations remain available as non-executable history;
qualitative scientific/council records remain visible and explicitly unverified.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from flora_translate.design_evidence import NUMERIC_FIELDS, reasoning_snapshot
from flora_translate.residence_time_basis import (
    INLET_STP_BASIS, IN_CHANNEL_BASIS, LIQUID_ONLY_BASIS,
    P_STP_BAR, T_STP_K, STP_MOLAR_VOLUME_ML_PER_MMOL,
    normalize_residence_time_basis,
)


QUANTITATIVE_FIELDS = set(NUMERIC_FIELDS) | {
    "residence_time_basis", "residence_time_inlet_min", "residence_time_in_channel_min",
    "gas_flow_sccm", "gas_flow_actual_mL_min", "BPR_basis", "pressure_absolute_bar",
}
TIME_SCOPE = (
    "Gross reactor-volume/flow indices only. Gas compression assumes ideal gas at "
    "the stated temperature and absolute pressure; neither phase holdup, packed-bed "
    "void volume, actual substrate contact time nor conversion is established."
)


def _number(*values):
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            result = float(value)
            if math.isfinite(result):
                return result
        except (ValueError, TypeError):
            pass
    return None


def _positive(value):
    number = _number(value)
    return number if number is not None and number > 0 else None


def _total(values):
    return sum(values) if all(value is not None for value in values) else None


def _agreement(declared, calculated):
    if declared is None or calculated is None:
        return None
    # Final quantities are rounded to six decimals. This is an arithmetic audit,
    # not the looser equipment/experimental tolerance used by other validators.
    return math.isclose(declared, calculated, rel_tol=1e-4, abs_tol=2e-6)


def _check(declared, calculated, equation):
    return {"declared": declared, "calculated": calculated,
            "equation": equation, "agrees": _agreement(declared, calculated)}


def _time_indices(values, *, gas_standard, gas_actual, fallback_basis):
    volume = _positive(values.get("reactor_volume_mL", values.get("V_R_mL")))
    liquid = _positive(values.get("Q_liquid_mL_min", values.get("flow_rate_mL_min")))
    basis = normalize_residence_time_basis(values.get("residence_time_basis") or fallback_basis)
    nominal = volume / liquid if volume is not None and liquid is not None else None
    inlet = (volume / (liquid + gas_standard)
             if volume is not None and liquid is not None and gas_standard is not None else None)
    channel = (volume / (liquid + gas_actual)
               if volume is not None and liquid is not None and gas_actual is not None else None)
    primary = {LIQUID_ONLY_BASIS: nominal, INLET_STP_BASIS: inlet, IN_CHANNEL_BASIS: channel}.get(basis)
    primary_equation = {LIQUID_ONLY_BASIS: "V/Q_liquid", INLET_STP_BASIS: "V/(Q_liquid+Q_gas_STP)",
                        IN_CHANNEL_BASIS: "V/(Q_liquid+Q_gas_actual)"}.get(basis, "unresolved time basis")
    checks = {"primary_time_min": _check(_number(values.get("residence_time_min")), primary, primary_equation)}
    for key, calculated, equation in (
        ("residence_time_inlet_min", inlet, "V/(Q_liquid+Q_gas_STP)"),
        ("residence_time_in_channel_min", channel, "V/(Q_liquid+Q_gas_actual)"),
    ):
        if values.get(key) is not None:
            checks[key] = _check(_number(values[key]), calculated, equation)
    return {"reactor_volume_mL": volume, "liquid_flow_mL_min": liquid,
            "gas_standard_mL_min": gas_standard, "gas_actual_mL_min": gas_actual,
            "nominal_liquid_space_time_min": nominal, "inlet_STP_index_min": inlet,
            "in_channel_index_min": channel, "primary_basis": basis, "checks": checks}


def _arithmetic(proposal, parameters, streams, stages):
    gases = [stream for stream in streams if stream.get("phase") == "gas"]
    bpr = _number(parameters.get("BPR_bar"))
    pressure_basis = str(parameters.get("BPR_basis") or "").strip().lower()
    expected_pressure = (bpr + P_STP_BAR if bpr is not None and pressure_basis == "gauge"
                         else bpr if bpr is not None and pressure_basis == "absolute" else None)
    pressure_check = _check(_number(parameters.get("pressure_absolute_bar")), expected_pressure,
                           "P_gauge + 1.01325 bar" if pressure_basis == "gauge" else "P_absolute")
    gas_rows = []
    for index, stream in enumerate(streams):
        if stream.get("phase") != "gas":
            continue
        standard = _positive(stream.get("gas_flow_sccm"))
        # These are the explicit conventions of the current production realizer;
        # do not read a generic gas flow field as a standard-volume flow.
        temperature = _number(parameters.get("temperature_C"))
        pressure = _positive(parameters.get("pressure_absolute_bar"))
        actual = (standard * ((temperature + 273.15) / T_STP_K) * (P_STP_BAR / pressure)
                  if standard is not None and temperature is not None and temperature > -273.15
                  and pressure is not None and not stages else None)
        total_molar = standard / STP_MOLAR_VOLUME_ML_PER_MMOL if standard is not None else None
        fraction = _number(stream.get("gas_reagent_mole_fraction"))
        valid_fraction = fraction if fraction is not None and 0 < fraction <= 1 else None
        active = total_molar * valid_fraction if total_molar is not None and valid_fraction is not None else None
        gas_rows.append({"stream_label": stream.get("stream_label"),
                         "source_path": f"$.final_design.streams[{index}]",
                         "gas_standard_mL_min": standard, "active_mole_fraction": valid_fraction,
                         "active_species": stream.get("gas_active_species"),
                         "total_molar_flow_mmol_min": total_molar,
                         "active_molar_flow_mmol_min": active,
                         "actual_flow_mL_min": _check(_number(stream.get("gas_flow_actual_mL_min")), actual,
                             "Q_STP*(T_K/273.15)*(1.01325/P_absolute_bar)"),
                         "checks": {}})
        for key, calculated in (("gas_total_molar_flow_mmol_min", total_molar),
                                 ("gas_active_molar_flow_mmol_min", active)):
            if stream.get(key) is not None:
                gas_rows[-1]["checks"][key] = _check(_number(stream[key]), calculated,
                    "Q_STP/(R*T_STP/P_STP)" + (" * active mole fraction" if "active" in key else ""))
    rows = []
    if stages:
        for stage in stages:
            qgas = _number(stage.get("Q_gas_sccm"), stage.get("gas_flow_sccm"))
            actual = _number(stage.get("Q_gas_actual_mL_min"), stage.get("gas_flow_actual_mL_min"))
            if not gases:
                qgas, actual = 0.0, 0.0
            row = _time_indices(stage, gas_standard=qgas, gas_actual=actual,
                                fallback_basis=parameters.get("residence_time_basis"))
            row["stage_number"] = stage.get("stage_number")
            rows.append(row)
    else:
        rows.append(_time_indices(parameters,
            gas_standard=_total([row["gas_standard_mL_min"] for row in gas_rows]),
            gas_actual=_total([row["actual_flow_mL_min"]["calculated"] for row in gas_rows]),
            fallback_basis=parameters.get("residence_time_basis")))
    checks = []
    for row in rows:
        checks.extend(row["checks"].values())
    for row in gas_rows:
        checks.extend(row["checks"].values())
        # A multistage design needs local gas temperature/pressure histories.
        # Its time-index arithmetic may close, but that does not verify gas
        # compression from one global operating temperature.
        checks.append(row["actual_flow_mL_min"])
    if gases:
        checks.append(pressure_check)
    if stages:
        primary_sum = _total([row["checks"]["primary_time_min"]["calculated"] for row in rows])
        total_check = _check(_number(parameters.get("residence_time_min")), primary_sum,
                             "sum(stage-specific time indices); not total V/final outlet Q")
        checks.append(total_check)
    else:
        total_check = None
    status = ("mismatch_requires_review" if any(check["agrees"] is False for check in checks)
              else "incomplete_requires_review" if any(check["agrees"] is None for check in checks)
              else "arithmetic_consistent")
    return {"status": status, "scope": TIME_SCOPE,
            "absolute_pressure_check": pressure_check if gases else None,
            "gas_standard_reference": {"temperature_K": T_STP_K, "pressure_bar": P_STP_BAR,
                                       "molar_volume_mL_per_mmol": STP_MOLAR_VOLUME_ML_PER_MMOL,
                                       "sccm_definition": "1 cm3/min = 1 mL/min at the stated reference; no factor of 60 or 1000"},
            "gas_streams": gas_rows, "time_indices": rows, "multistage_total_time_check": total_check}


def build_final_rationale(proposal: dict[str, Any], parameters: dict[str, Any],
                          streams: list[dict[str, Any]], stages: list[dict[str, Any]]):
    """Publish checked arithmetic and retain every original model record once.

    Source records are preserved verbatim inside history entries. This function
    never rewrites setpoints and never validates scientific claims from a match
    between parameter snapshots. A stale qualitative concern stays visible.
    """
    current, history = {}, {}
    for field, source in (proposal.get("decision_rationale") or {}).items():
        record = deepcopy(source)
        if field in QUANTITATIVE_FIELDS:
            history[field] = {"authority": "non-executable original model rationale; arithmetic superseded by final calculation audit",
                              "source_record": record}
        else:
            # Do not filter scientific cautions or council judgments according
            # to whether they favor the selected design.
            current[field] = {"text": record.get("text", ""),
                              "source": record.get("source"),
                              "status": "unverified_model_scientific_rationale",
                              "parameter_applicability": record.get("status", "not_recorded"),
                              "authority": "scientific argument or concern, not a verified calculation or operating instruction",
                              "source_record": {key: value for key, value in record.items() if key != "text"}}
    arithmetic = _arithmetic(proposal, parameters, streams, stages)
    current["final_quantitative_basis"] = {
        "source": "deterministic_final_parameter_audit", "status": arithmetic.pop("status"),
        "text": "Current numerical explanations use the final delivered settings. Original model explanations remain in rationale_history; matching settings do not validate their mathematics or scientific claims.",
        "parameter_snapshot": reasoning_snapshot({**parameters, "streams": streams, "stage_parameters": stages}),
        **arithmetic,
    }
    current["scientific_review_requirement"] = {
        "status": "experimental_review_required",
        "text": "Numerical closure does not resolve the scientific or safety concerns retained above or in rationale_history. Review the original chemical arguments, evidence transfer, phase behavior and experimental validation before use; no yield or safe-operation claim follows from this audit.",
    }
    return current, history
