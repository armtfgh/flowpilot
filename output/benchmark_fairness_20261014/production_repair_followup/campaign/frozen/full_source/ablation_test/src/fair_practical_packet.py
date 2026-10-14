"""Lossless final-delivery projection for the separately frozen practical study.

No Pydantic repair, scoring, hidden reference substitution, or council evidence.
The original fourteen-criterion rubric and judge remain outside this adapter.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from ablation_test.src.newgen_2_outcome import _scrub
from ablation_test.src.outcome_verification import delivered_arithmetic, number


CONTENT_POLICY = (
    "Credit relevant facts anywhere in the complete selected delivered outcome, "
    "including prose, extra fields, recipes, graph and rationale. Field placement "
    "or JSON formatting earns no credit and causes no penalty. Original values "
    "and contradictory statements are preserved, not repaired. Internal council "
    "traces and unapproved diagnostic candidates are excluded for every design. "
    "Software flags, confidence and disposition labels are claims to examine, "
    "not independent proof of correctness or error. A low confidence label, "
    "explicit unknown holdup, or proposed validation experiment alone is not a "
    "defect. A withheld executable artifact remains a delivery limitation. "
    "Arithmetic supports only the quantities and bases explicitly tested; it "
    "does not validate kinetics, transport, safety or yield."
)

_INTERNAL_KEYS = {
    "diagnostic", "diagnostic_topology", "council_messages", "council_rounds",
    "deliberation_log", "engineering_history", "pre_council_proposal",
    "validation_checks", "semantic_checks",
}


def _delivered(value: Any) -> Any:
    """Exclude named internal traces, preserving all other values verbatim."""
    if isinstance(value, dict):
        return {
            key: _delivered(item) for key, item in value.items()
            if key not in _INTERNAL_KEYS and not key.startswith("council_")
            and not (isinstance(item, dict) and item.get("source") == "council_selection")
        }
    if isinstance(value, list):
        return [_delivered(item) for item in value]
    return deepcopy(value)


def _one_shot(result: dict[str, Any]) -> dict[str, Any]:
    # An empty raw proposal is still authoritative. Never replace it with a
    # normalized object that added or rewrote values.
    envelope = result.get("response_envelope") or {}
    if isinstance(result.get("raw_proposal"), dict):
        proposal = result["raw_proposal"]
    elif isinstance(envelope.get("proposal"), dict):
        proposal = envelope["proposal"]
    else:
        proposal = result.get("proposal") or {}
    proposal = _delivered(proposal)
    mapped = {
        "streams", "stage_parameters", "stages", "topology", "process_graph",
        "chemistry_notes", "chemistry_identity", "stream_components",
        "stationary_components", "operating_procedure", "pre_reactor_steps",
        "post_reactor_steps", "safety_flags", "safety_controls", "safety",
        "reasoning_per_field", "evidence_calibration", "literature_analogies",
        "decision_rationale", "validation_experiments", "inventory_selection",
    }
    topology = proposal.get("topology", proposal.get("process_graph"))
    if topology is None:
        topology = {key: proposal.get(key) for key in (
            "reactor_type", "mixer_type", "mixing_order_reasoning", "inventory_selection")}
    alternatives = {}
    if "stage_parameters" in proposal and "stages" in proposal:
        alternatives["stages"] = proposal["stages"]
    if "topology" in proposal and "process_graph" in proposal:
        alternatives["process_graph"] = proposal["process_graph"]
    safety_key = next((key for key in ("safety_controls", "safety", "safety_flags") if key in proposal), None)
    alternatives.update({key: proposal[key] for key in ("safety_controls", "safety", "safety_flags")
                         if key in proposal and key != safety_key})
    return {
        "parameters": {key: value for key, value in proposal.items() if key not in mapped},
        "streams": proposal.get("streams", []),
        "stages": proposal.get("stage_parameters", proposal.get("stages", [])),
        "topology": topology,
        "chemistry_and_operations": {key: proposal.get(key) for key in (
            "chemistry_notes", "chemistry_identity", "stream_components",
            "stationary_components", "operating_procedure", "pre_reactor_steps", "post_reactor_steps")},
        "safety_controls": proposal.get("safety_controls", proposal.get("safety", proposal.get("safety_flags", []))),
        "evidence_and_reasoning": {key: proposal.get(key) for key in (
            "reasoning_per_field", "evidence_calibration", "literature_analogies",
            "decision_rationale", "validation_experiments")},
        "inventory_selection": proposal.get("inventory_selection"),
        "additional_delivered_fields": alternatives,
        "reported_disposition": envelope.get("recommended_disposition", result.get("reported_disposition")),
        "disposition_rationale": envelope.get("disposition_rationale", result.get("disposition_rationale")),
        "additional_response_fields": _delivered({key: value for key, value in envelope.items()
            if key not in {"proposal", "recommended_disposition", "disposition_rationale"}}),
    }


def _pipeline(result: dict[str, Any]) -> dict[str, Any]:
    final = _delivered(result.get("final_design") or {})
    graph = final.get("process_graph") or {}
    # An explicitly empty/blocked final graph cannot be rescued by diagnostics.
    topology = graph.get("topology", graph) if isinstance(graph, dict) else graph
    known = {"parameters", "streams", "stages", "process_graph", "status",
        "chemistry_identity", "stream_components", "stationary_components",
        "operating_procedure", "safety", "decision_rationale", "validation_experiments",
        "consistency"}
    report = _delivered(result.get("result_report") or {})
    return {
        "parameters": final.get("parameters", {}),
        "streams": final.get("streams", []),
        "stages": final.get("stages", []),
        "topology": topology,
        "graph_context": {key: value for key, value in graph.items() if key != "topology"} if isinstance(graph, dict) and "topology" in graph else {},
        "delivery_status": final.get("status", "missing"),
        "delivery_issues": (final.get("consistency") or {}).get("issues", []),
        "chemistry_and_operations": {key: final.get(key) for key in (
            "chemistry_identity", "stream_components", "stationary_components", "operating_procedure")},
        "safety_controls": final.get("safety", {}),
        "evidence_and_reasoning": {
            "decision_rationale": final.get("decision_rationale", {}),
            "validation_experiments": final.get("validation_experiments", []),
            "confidence": result.get("confidence"),
            "canonical_explanation": result.get("canonical_explanation"),
        },
        "reported_disposition": result.get("reported_disposition", result.get("recommended_disposition")),
        "disposition_rationale": result.get("disposition_rationale"),
        "additional_delivered_fields": {key: value for key, value in final.items() if key not in known},
        # Report input responses are already represented in the case context.
        # Keep repeated final values: a discrepancy is evidence, not deduplicated.
        "report_details": {key: value for key, value in report.items()
            if key not in {"schema_version", "source", "canonical_sha256", "status", "responses"}},
    }


def _time_check(outcome: dict[str, Any], *, has_gas: bool, is_multistage: bool) -> dict[str, Any]:
    p = outcome.get("parameters") or {}
    if not isinstance(p, dict):
        return {"assessable": False, "reason": "Delivered parameters are not a structured object; inspect original evidence."}
    q, volume, tau = (number(p.get(key)) for key in (
        "flow_rate_mL_min", "reactor_volume_mL", "residence_time_min"))
    check = {"assessable": False, "scope": "Declared time index only; no phase holdup or measured contact time inferred.",
             "source_path": "$.delivered_final_outcome.parameters"}
    if is_multistage:
        return {**check, "reason": "Requires individual stage volumes and cumulative flows; no global closure asserted."}
    if any(value is None or value <= 0 for value in (q, volume, tau)):
        return {**check, "reason": "Positive delivered liquid flow, volume and time are not all available."}
    basis = str(p.get("residence_time_basis_code") or p.get("residence_time_basis") or "").casefold().replace("-", "_")
    streams = outcome.get("streams") or []
    if not isinstance(streams, list):
        return {**check, "reason": "Delivered streams are not a list; no numeric stream interpretation imposed."}
    gas = [stream for stream in streams if isinstance(stream, dict) and (
        str(stream.get("phase", "")).casefold() == "gas" or stream.get("gas_flow_sccm") is not None)]
    if not has_gas:
        expected, equation = volume / q, "V / Q_liquid; nominal empty-volume time, not measured packed-bed contact time"
    elif "inlet" in basis and ("stp" in basis or "standard" in basis):
        flows = [number(stream.get("gas_flow_sccm")) for stream in gas]
        if not flows or any(value is None or value < 0 for value in flows):
            return {**check, "reason": "Inlet/STP basis lacks complete declared standard gas flows."}
        expected, equation = volume / (q + sum(flows)), "V / (Q_liquid + sum(Q_gas_standard)); inlet/standard-volume index only"
    elif "channel" in basis:
        flows = [number(stream.get("gas_flow_actual_mL_min")) for stream in gas]
        if not flows or any(value is None or value < 0 for value in flows):
            return {**check, "reason": "Channel basis lacks complete declared channel gas flows."}
        expected, equation = volume / (q + sum(flows)), "V / (Q_liquid + sum(Q_gas_channel)); homogeneous channel-volume index only"
    else:
        return {**check, "reason": "Gas process without an unambiguous inlet/standard or channel basis; no liquid-only closure imposed."}
    return {**check, "assessable": True, "reported_time_min": tau,
            "calculated_index_min": expected, "equation": equation,
            "relative_error": abs(tau - expected) / expected,
            "agrees_within_10_percent": math.isclose(tau, expected, rel_tol=.10)}


def _conditional_gas_examples(arithmetic: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Do not turn the helper's nominal gas reference into candidate authority."""
    rows = arithmetic.pop("gas_streams", [])
    if not rows:
        return
    streams = outcome.get("streams") or []
    for row in rows:
        index = int(row["source_path"].rsplit("[", 1)[1][:-1])
        source = streams[index]
        # The helper's fallback molar_equiv can mean total air, not reactive O2.
        # Preserve its declared value/basis without relabeling it as active gas.
        row.pop("reported_equivalents", None)
        row["declared_generic_molar_equiv"] = source.get("molar_equiv")
        row["declared_generic_equivalent_basis"] = source.get("molar_equiv_basis")
        row["declared_active_gas_equivalents"] = source.get("gas_active_equivalents")
        for key in ("declared_fraction_matches_identity", "equivalents_match_active_species_within_rounding", "channel_flow_matches_within_rounding"):
            row.pop(key, None)
        row["comparison_status"] = (
            "CONDITIONAL EXAMPLE ONLY, not a pass/fail verification of this design. "
            "Calculations assume the nominal STP reference and gas composition stated "
            "in this row. Check the original meter reference, declared composition and "
            "equivalent basis before comparing with delivered claims. A missing or "
            "different reference does not establish a numerical defect."
        )
    arithmetic["conditional_nominal_gas_reference_examples"] = rows


def build_fair_packet(*, candidate: str, case_context: dict[str, Any], result: dict[str, Any],
                      deterministic_verification: dict[str, Any] | None = None, has_gas: bool,
                      is_multistage: bool) -> dict[str, Any]:
    """Drop-in packet builder for fresh matched-format outcomes only.

    The legacy verification argument is accepted for caller compatibility but
    cannot inject phase-insensitive tests or internal validation verdicts.
    """
    pipeline = isinstance(result.get("final_design"), dict) or result.get("variant") == "full"
    outcome = _pipeline(result) if pipeline else _one_shot(result)
    # This helper never rewrites values and explicitly labels its nominal STP
    # reference. Any contrary meter reference must be reviewed, not overwritten.
    streams = outcome.get("streams") or []
    if (not isinstance(outcome.get("parameters"), dict) or not isinstance(streams, list)
            or any(not isinstance(stream, dict) for stream in streams)):
        arithmetic = {"unassessable_reason": "Delivered parameter/stream shapes cannot be read by the arithmetic helper; original evidence is retained without repair."}
    else:
        arithmetic = delivered_arithmetic(outcome, multistage=is_multistage)
        _conditional_gas_examples(arithmetic, outcome)
    packet = {
        "schema_version": "fair_practical_selected_delivery_v1.0",
        "candidate_id": candidate,
        "case_context": {key: deepcopy(case_context.get(key)) for key in (
            "case_label", "protocol", "objective", "hard_constraints", "authoritative_inventory",
            "held_out_source_fact_check", "source_policy")},
        "delivered_final_outcome": outcome,
        "deterministic_verification_sheet": {
            "scope": "Recalculation from delivered values only; no candidate repair and no assigned scores.",
            "reference_caution": "Gas worksheet uses its stated nominal STP reference. A different or missing meter reference must be reviewed explicitly; no holdup is inferred.",
            "delivered_arithmetic": arithmetic,
            "declared_time_index": _time_check(outcome, has_gas=has_gas, is_multistage=is_multistage),
        },
        "evaluation_boundary": CONTENT_POLICY,
    }
    packet["case_context"]["applicability"] = {"gas_process": has_gas, "multistage_process": is_multistage}
    return _scrub(packet)
