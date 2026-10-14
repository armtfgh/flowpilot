"""Architecture-neutral arithmetic checks; never change a candidate or score."""
import math
import re


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


def delivered_arithmetic(outcome, *, multistage=False):
    p = outcome.get("parameters") or {}
    q, c = number(p.get("flow_rate_mL_min")), number(p.get("concentration_M"))
    v = number(p.get("reactor_volume_mL"))
    report = {"scope": "Arithmetic from delivered values only; no kinetic, transport, safety or yield validation.",
              "parameter_path": "$.delivered_final_outcome.parameters", "gas_streams": []}
    if multistage:
        report["unassessable_reason"] = "Stage-specific limiting molar delivery required; a global Q*C cannot verify a telescoped gas feed."
        return report
    if q is not None and c is not None and q > 0 and c > 0:
        report["reference_substrate_mmol_min"] = q * c
        report["reference_substrate_equation"] = "C [M = mmol/mL] * Q_liquid [mL/min]"
        if v is not None and v > 0:
            report["nominal_volume_over_liquid_flow_min"] = v / q
            report["time_limitation"] = "V/Q is nominal space time, not measured multiphase or packed-bed contact time."
    for index, stream in enumerate(outcome.get("streams") or []):
        sccm = number(stream.get("gas_flow_sccm"))
        if sccm is None or sccm < 0:
            continue
        contents = str(stream.get("contents") or "").lower()
        fraction = number(stream.get("gas_reagent_mole_fraction"))
        species = stream.get("gas_active_species")
        if re.search(r"\bair\b", contents):
            identity_fraction, identity_species = .21, "O2"
        elif not any(word in contents for word in ("mixture", "dilut", "/")) and re.search(r"\bh2\b|\bhydrogen\b|\bo2\b|\boxygen\b", contents):
            identity_fraction = fraction if fraction is not None else 1.
            identity_species = "H2" if re.search(r"\bh2\b|\bhydrogen\b", contents) else "O2"
        else:
            identity_fraction, identity_species = fraction, species
        row = {"source_path": f"$.delivered_final_outcome.streams[{index}]", "gas_inlet_sccm": sccm,
               "active_species": identity_species, "active_mole_fraction": identity_fraction,
               "reference": "nominal STP, 273.15 K and 1.01325 bar; 22.414 mL/mmol; verify compatibility with the declared meter reference"}
        if fraction is not None and identity_fraction is not None:
            row["declared_fraction_matches_identity"] = math.isclose(fraction, identity_fraction, abs_tol=1e-9)
        if identity_fraction is not None and 0 < identity_fraction <= 1:
            row["total_gas_mmol_min"] = sccm / 22.414
            row["active_gas_mmol_min"] = sccm * identity_fraction / 22.414
            if report.get("reference_substrate_mmol_min"):
                eq = row["active_gas_mmol_min"] / report["reference_substrate_mmol_min"]
                row["calculated_active_equivalents"] = eq
                stated = number(stream.get("gas_active_equivalents") if stream.get("gas_active_equivalents") is not None else stream.get("molar_equiv"))
                row["reported_equivalents"] = stated
                row["equivalents_match_active_species_within_rounding"] = None if stated is None else math.isclose(stated, eq, rel_tol=.001, abs_tol=.0001)
                row["equivalent_interpretation"] = "Compare the reported basis with active-species equivalents, not total carrier-gas equivalents."
        temperature = number(p.get("temperature_C"))
        absolute = number(p.get("pressure_absolute_bar"))
        if absolute is None and str(p.get("BPR_basis", "")).lower() == "gauge":
            bpr = number(p.get("BPR_bar"))
            absolute = bpr + 1.01325 if bpr is not None else None
        if absolute and absolute > 0 and temperature is not None:
            actual = sccm * ((temperature + 273.15) / 273.15) * (1.01325 / absolute)
            row["calculated_channel_gas_mL_min"] = actual
            reported = number(stream.get("gas_flow_actual_mL_min"))
            row["channel_flow_matches_within_rounding"] = None if reported is None else math.isclose(actual, reported, rel_tol=.001, abs_tol=.0001)
        else:
            row["channel_flow_unassessable_reason"] = "No explicit numerical absolute pressure/gauge-basis field; inspect the delivered narrative."
        report["gas_streams"].append(row)
    return report


def resolves_json_path(packet, path):
    if not isinstance(path, str) or not path:
        return False
    path = path.removeprefix("$.")
    path = re.sub(r"\[['\"]([^'\"]+)['\"]\]", r".\1", path)
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    try:
        value = packet
        for key in path.split("."):
            value = value[int(key)] if isinstance(value, list) else value[key]
        return True
    except (KeyError, IndexError, TypeError, ValueError):
        return False
