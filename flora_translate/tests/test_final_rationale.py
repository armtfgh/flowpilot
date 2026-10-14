"""A current calculation must not endorse a stale or incorrect model explanation."""
from copy import deepcopy
import json

import pytest

from flora_translate.design_evidence import capture_reasoning, reconcile_reasoning
from flora_translate.final_rationale import build_final_rationale
from flora_translate.schemas import FlowProposal


def hydrogen_design():
    qgas = 1.412002
    actual = qgas * (333.15 / 273.15) * (1.01325 / 22.01325)
    parameters = {"temperature_C": 60., "BPR_bar": 21., "BPR_basis": "gauge",
                  "pressure_absolute_bar": 22.01325, "reactor_volume_mL": 3.,
                  "flow_rate_mL_min": .5, "residence_time_min": 3 / (.5 + qgas),
                  "residence_time_basis": "inlet/STP", "residence_time_inlet_min": 3 / (.5 + qgas),
                  "residence_time_in_channel_min": 3 / (.5 + actual)}
    streams = [{"stream_label": "A", "phase": "liquid", "flow_rate_mL_min": .5,
                "concentration_M": .126, "contents": ["substrate"]},
               {"stream_label": "B", "phase": "gas", "gas_flow_sccm": qgas,
                "gas_flow_actual_mL_min": actual, "gas_reagent_mole_fraction": 1,
                "gas_active_species": "H2", "gas_total_molar_flow_mmol_min": .063,
                "gas_active_molar_flow_mmol_min": .063}]
    return parameters, streams


def test_old_arithmetic_is_not_certified_when_top_level_settings_match():
    params, streams = hydrogen_design()
    old = {"text": "5 sccm / 60 = 0.083 mL/min, so primary time is 6 min.",
           "status": "conditions_preserved", "parameter_snapshot": deepcopy(params),
           "source": "model_proposed_scientific_rationale"}
    proposal = {"decision_rationale": {"residence_time_min": old}}
    current, history = build_final_rationale(proposal, params, streams, [])
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "arithmetic_consistent"
    values = audit["time_indices"][0]
    assert values["nominal_liquid_space_time_min"] == 6
    assert values["inlet_STP_index_min"] == pytest.approx(1.569036)
    assert values["in_channel_index_min"] == pytest.approx(5.178932)
    assert values["primary_basis"] == "inlet_stp"
    assert old["text"] not in json.dumps(current)
    assert history["residence_time_min"]["source_record"] == old
    assert proposal["decision_rationale"]["residence_time_min"] == old
    assert "phase holdup" in audit["scope"]


def test_gas_only_change_invalidates_captured_rationale():
    params, streams = hydrogen_design()
    proposal = FlowProposal(**params, streams=streams,
        reasoning_per_field={"residence_time_min": "Gas contact is adequate for this target."})
    capture_reasoning(proposal)
    reconcile_reasoning(proposal)
    assert proposal.decision_rationale["residence_time_min"]["status"] == "conditions_preserved"
    proposal.streams[1].gas_flow_sccm *= 2
    reconcile_reasoning(proposal)
    record = proposal.decision_rationale["residence_time_min"]
    assert record["status"] == "requires_review_after_parameter_change"
    assert "streams" in record["changed_dependencies"]
    assert "residence_time_min" not in proposal.reasoning_per_field
    assert record["parameter_match_is_not_scientific_validation"]


def test_all_scientific_concerns_survive_and_are_not_promoted_to_validated_facts():
    params, streams = hydrogen_design()
    concern = {"text": "Gas transfer may limit conversion; catalyst leaching remains unmeasured.",
               "status": "requires_review_after_parameter_change", "changed_dependencies": ["streams"],
               "source": "council_review", "parameter_snapshot": {"flow_rate_mL_min": .1}}
    proposal = {"decision_rationale": {"chemistry_notes": deepcopy(concern)}}
    current, history = build_final_rationale(proposal, params, streams, [])
    delivered = current["chemistry_notes"]
    assert delivered["text"] == concern["text"]
    assert delivered["status"] == "unverified_model_scientific_rationale"
    assert delivered["parameter_applicability"] == concern["status"]
    assert {**delivered["source_record"], "text": delivered["text"]} == concern
    assert not history  # No duplicated qualitative paragraphs in the history.


def test_real_numeric_mismatch_remains_visible_and_is_not_repaired_by_reporting():
    params, streams = hydrogen_design()
    params["residence_time_min"] = 9.
    frozen = deepcopy((params, streams))
    current, _ = build_final_rationale({}, params, streams, [])
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "mismatch_requires_review"
    check = audit["time_indices"][0]["checks"]["primary_time_min"]
    assert check["declared"] == 9
    assert check["calculated"] == pytest.approx(1.569036)
    assert check["agrees"] is False
    assert (params, streams) == frozen


def test_gas_compression_and_molar_flow_errors_are_reported():
    params, streams = hydrogen_design()
    streams[1]["gas_flow_actual_mL_min"] = .61
    streams[1]["gas_active_molar_flow_mmol_min"] = 100
    current, _ = build_final_rationale({}, params, streams, [])
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "mismatch_requires_review"
    row = audit["gas_streams"][0]
    assert row["actual_flow_mL_min"]["agrees"] is False
    assert row["checks"]["gas_active_molar_flow_mmol_min"]["agrees"] is False
    assert row["actual_flow_mL_min"]["calculated"] == pytest.approx(.0792695461)


def test_missing_reference_inputs_cannot_become_a_successful_audit():
    params, streams = hydrogen_design()
    params.pop("pressure_absolute_bar")
    current, _ = build_final_rationale({}, params, streams, [])
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "incomplete_requires_review"
    assert audit["time_indices"][0]["in_channel_index_min"] is None
    assert audit["gas_streams"][0]["actual_flow_mL_min"]["agrees"] is None


def test_cumulative_feed_stages_use_local_flows_not_final_outlet_flow():
    params = {"reactor_volume_mL": 10, "flow_rate_mL_min": .2,
              "residence_time_min": 56.25, "residence_time_basis": "liquid-only"}
    stages = [{"stage_number": 1, "reactor_volume_mL": 5, "Q_liquid_mL_min": .16,
               "residence_time_min": 31.25},
              {"stage_number": 2, "reactor_volume_mL": 5, "Q_liquid_mL_min": .2,
               "residence_time_min": 25}]
    current, _ = build_final_rationale({}, params, [], stages)
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "arithmetic_consistent"
    assert audit["multistage_total_time_check"]["calculated"] == 56.25
    assert [row["nominal_liquid_space_time_min"] for row in audit["time_indices"]] == [31.25, 25]


def test_unknown_time_basis_remains_unresolved_even_with_equal_time_indices():
    params = {"reactor_volume_mL": 1, "flow_rate_mL_min": .5,
              "residence_time_min": 2, "residence_time_basis": "unresolved"}
    current, _ = build_final_rationale({}, params, [], [])
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "incomplete_requires_review"
    assert audit["time_indices"][0]["checks"]["primary_time_min"]["calculated"] is None


def test_inconsistent_pressure_bases_are_not_used_to_certify_compression():
    params, streams = hydrogen_design()
    params["pressure_absolute_bar"] = 21
    current, _ = build_final_rationale({}, params, streams, [])
    audit = current["final_quantitative_basis"]
    assert audit["status"] == "mismatch_requires_review"
    assert audit["absolute_pressure_check"]["agrees"] is False
    assert audit["absolute_pressure_check"]["calculated"] == 22.01325


def test_multistage_gas_compression_is_not_verified_with_one_global_temperature():
    params, streams = hydrogen_design()
    stage = {**params, "stage_number": 1, "Q_liquid_mL_min": .5,
             "Q_gas_sccm": streams[1]["gas_flow_sccm"],
             "Q_gas_actual_mL_min": streams[1]["gas_flow_actual_mL_min"]}
    current, _ = build_final_rationale({}, params, streams, [stage])
    audit = current["final_quantitative_basis"]
    assert audit["time_indices"][0]["checks"]["primary_time_min"]["agrees"] is True
    assert audit["gas_streams"][0]["actual_flow_mL_min"]["calculated"] is None
    assert audit["status"] == "incomplete_requires_review"


def test_final_contract_publishes_current_calculations_and_original_evidence():
    from flora_translate.final_design_contract import build_final_design_contract
    from flora_translate.tests.test_final_design_contract import _ready_result
    result = _ready_result()
    original = {"residence_time_min": {"text": "An unsupported residence-time assertion.",
                                       "status": "conditions_preserved"},
                "council_selection": {"text": "Outlet conversion remains unmeasured.",
                                      "status": "conditions_preserved"}}
    result["proposal"]["decision_rationale"] = deepcopy(original)
    final = build_final_design_contract(result)
    assert final["status"] == "executable"
    assert "final_quantitative_basis" in final["decision_rationale"]
    assert final["rationale_history"]["residence_time_min"]["source_record"] == original["residence_time_min"]
    assert final["decision_rationale"]["council_selection"]["text"] == original["council_selection"]["text"]
    assert result["proposal"]["decision_rationale"] == original
