from copy import deepcopy
import json
import pytest

from ablation_test.src.fair_practical_packet import build_fair_packet, CONTENT_POLICY
from ablation_test.src.newgen_2_outcome import packet_gate


def proposal():
    return {'flow_rate_mL_min':.5,'concentration_M':.126,'reactor_volume_mL':3.,
        'temperature_C':60.,'BPR_bar':21.,'BPR_basis':'gauge','pressure_absolute_bar':22.01325,
        'residence_time_min':3/(.5+1.412002),'residence_time_basis':'inlet_stp',
        'streams':[{'stream_label':'A','phase':'liquid','concentration_M':.126,'flow_rate_mL_min':.5},
            {'stream_label':'B','phase':'gas','contents':['H2'],'gas_flow_sccm':1.412002,
             'gas_flow_actual_mL_min':.07927,'gas_reagent_mole_fraction':1.,'molar_equiv':1.}],
        'chemistry_notes':'Preserve the original 1.26 mmol charge and measure holdup.',
        'pre_reactor_steps':['Prime with methanol.'],'post_reactor_steps':['Cool before depressurizing.'],
        'safety_flags':['Purge before hydrogen.'], 'evidence_calibration':{'yield':'Not predicted.'}}


def build(result,has_gas=True,is_multistage=False):
    return build_fair_packet(candidate='F-TEST',case_context={'protocol':'Hydrogenolysis protocol',
        'authoritative_inventory':{'reactors':[{'volume_mL':3.}]},'held_out_source_fact_check':{'judge_only':'reference'}},
        result=result, deterministic_verification={'top_level_volume_flow_time_closure':{'passes_10_percent':False}},
        has_gas=has_gas,is_multistage=is_multistage)


def pipeline(p):
    fields={k:v for k,v in p.items() if k not in {'streams','chemistry_notes','pre_reactor_steps','post_reactor_steps','safety_flags','evidence_calibration'}}
    return {'variant':'full','confidence':'LOW','final_design':{'status':'executable','parameters':fields,
        'streams':p['streams'],'stages':[],'process_graph':{'topology':{'unit_operations':[{'id':'R'}]}},
        'stream_components':[{'name':'substrate','concentration_M':.126}],
        'stationary_components':[{'name':'Pd catalyst','location':'packed bed'}],
        'operating_procedure':[{'instruction':'Charge 1.26 mmol once into 10 mL.'}],
        'safety':{'controls':['Purge before hydrogen.']},
        'decision_rationale':{'temperature':{'text':'Protocol gives 60 C.','status':'conditions_preserved'},
            'residence_time':{'text':'Earlier rationale says 6 min.','status':'requires_review_after_parameter_change'},
            'council_selection':{'source':'council_selection','text':'Hidden candidate scores.'}},
        'validation_experiments':[{'purpose':'Measure liquid holdup.'}],
        'diagnostic':{'preliminary_candidate':{'residence_time_min':99}},
        'consistency':{'passed':True,'issues':[],'validation_checks':[{'internal':'DO NOT SCORE'}]}}}


def test_raw_one_shot_values_preserved_instead_of_normalized_proposal():
    raw=proposal(); normalized=deepcopy(raw);normalized['residence_time_min']=6.
    result={'variant':'general_one_shot','raw_proposal':raw,'proposal':normalized}
    before=deepcopy(result);out=build(result)
    assert out['delivered_final_outcome']['parameters']['residence_time_min']==raw['residence_time_min']
    assert result==before and not packet_gate(out)


def test_empty_raw_proposal_does_not_fall_back_to_normalized_values():
    packet=build({'raw_proposal':{},'proposal':proposal()})
    assert packet['delivered_final_outcome']['parameters']=={}
    assert not packet['deterministic_verification_sheet']['declared_time_index']['assessable']


def test_inlet_gas_basis_does_not_receive_liquid_only_volume_failure():
    packet=build({'raw_proposal':proposal()})
    sheet=packet['deterministic_verification_sheet']
    assert 'top_level_volume_flow_time_closure' not in sheet
    assert sheet['declared_time_index']['agrees_within_10_percent']
    assert sheet['delivered_arithmetic']['nominal_volume_over_liquid_flow_min']==6
    assert 'not measured' in sheet['delivered_arithmetic']['time_limitation']


def test_channel_basis_uses_declared_channel_gas_without_holdup_inference():
    p=proposal();p['residence_time_basis']='in_channel';p['residence_time_min']=3/(.5+.07927)
    packet=build({'raw_proposal':p});check=packet['deterministic_verification_sheet']['declared_time_index']
    assert check['agrees_within_10_percent'] and 'homogeneous' in check['equation']
    assert 'no phase holdup' in check['scope']


def test_ambiguous_gas_basis_remains_unassessable():
    p=proposal();p['residence_time_basis']='actual contact time';p['residence_time_min']=6.
    check=build({'raw_proposal':p})['deterministic_verification_sheet']['declared_time_index']
    assert not check['assessable'] and 'no liquid-only closure' in check['reason']
    p['residence_time_basis']='inlet_total'
    assert not build({'raw_proposal':p})['deterministic_verification_sheet']['declared_time_index']['assessable']


def test_liquid_only_and_multistage_timing_scopes():
    p=proposal();p['streams']=p['streams'][:1];p['residence_time_min']=6.
    assert build({'raw_proposal':p},has_gas=False)['deterministic_verification_sheet']['declared_time_index']['agrees_within_10_percent']
    multi=build({'raw_proposal':p},is_multistage=True)['deterministic_verification_sheet']
    assert not multi['declared_time_index']['assessable']
    assert 'unassessable_reason' in multi['delivered_arithmetic']


def test_same_delivered_numbers_get_same_worksheet_in_both_architectures():
    p=proposal()
    direct=build({'raw_proposal':p})['deterministic_verification_sheet']
    agent=build(pipeline(p))['deterministic_verification_sheet']
    assert direct==agent


def test_complete_final_recipe_evidence_and_stale_status_survive_without_council():
    result=pipeline(proposal());before=deepcopy(result);packet=build(result)
    out=packet['delivered_final_outcome']
    assert out['chemistry_and_operations']['operating_procedure'][0]['instruction']=='Charge 1.26 mmol once into 10 mL.'
    assert out['chemistry_and_operations']['stationary_components'][0]['name']=='Pd catalyst'
    assert out['evidence_and_reasoning']['decision_rationale']['residence_time']['status']=='requires_review_after_parameter_change'
    text=json.dumps(packet)
    assert 'Hidden candidate scores' not in text and 'DO NOT SCORE' not in text and 'preliminary_candidate' not in text
    assert result==before and not packet_gate(packet)


def test_raw_extra_calculations_and_conflicting_alternative_fields_are_retained():
    p=proposal();p['unexpected_output_fact']={'amount':12.345};p['heat_transfer_metrics']={'raw_claim':4.321}
    p['stages']={'alternate':'keep'};p['stage_parameters']=[{'original':1}]
    p['safety_controls']=['One statement'];p['safety']={'conflicting':'retain'}
    packet=build({'raw_proposal':p});s=json.dumps(packet)
    assert all(value in s for value in ['12.345','4.321','alternate','conflicting','Purge before hydrogen.'])
    assert packet['evaluation_boundary']==CONTENT_POLICY


def test_blocked_final_artifact_not_rescued_by_diagnostic_or_old_topology():
    result=pipeline(proposal());result['final_design'].update(status='blocked',process_graph={},operating_procedure=[])
    result['process_topology']={'unit_operations':[{'id':'HIDDEN'}]}
    packet=build(result)
    assert packet['delivered_final_outcome']['topology']=={}
    assert 'HIDDEN' not in json.dumps(packet)


def test_generator_labels_scrubbed_and_held_out_context_not_copied_to_outcome():
    p=proposal();p['chemistry_notes']='Claude and FlowPilot use these numbers.'
    packet=build({'raw_proposal':p})
    assert not packet_gate(packet)
    assert 'judge_only' not in json.dumps(packet['delivered_final_outcome'])
    assert packet['case_context']['held_out_source_fact_check']['judge_only']=='reference'


def test_report_discrepancies_are_retained_instead_of_silently_deduplicated():
    result=pipeline(proposal());result['result_report']={'streams':[{'concentration_M':.252}],
        'gas_reference':{'basis':'meter reference unknown'},'responses':[{'private_input':'already supplied'}]}
    out=build(result)['delivered_final_outcome']
    assert out['report_details']['streams'][0]['concentration_M']==.252
    assert 'unknown' in out['report_details']['gas_reference']['basis']
    assert 'responses' not in out['report_details']


def test_invalid_stream_shapes_preserved_without_aborting_or_numeric_repair():
    p=proposal();p['streams']='Unstructured feed statement'
    packet=build_fair_packet(candidate='BAD-SHAPE',case_context={},result={'raw_proposal':p},has_gas=True,is_multistage=False)
    assert packet['delivered_final_outcome']['streams']==p['streams']
    assert 'unassessable_reason' in packet['deterministic_verification_sheet']['delivered_arithmetic']
    assert not packet['deterministic_verification_sheet']['declared_time_index']['assessable']


def test_total_air_equivalents_are_not_compared_as_active_oxygen_equivalents():
    p=proposal();p['streams'][1].update(contents=['air'],gas_reagent_mole_fraction=.21,
        molar_equiv=4.76,molar_equiv_basis='total mol air / mol substrate')
    packet=build({'raw_proposal':p})
    row=packet['deterministic_verification_sheet']['delivered_arithmetic']['conditional_nominal_gas_reference_examples'][0]
    assert row['declared_generic_molar_equiv']==4.76
    assert row['declared_active_gas_equivalents'] is None
    assert 'equivalents_match_active_species_within_rounding' not in row
    assert 'reported_equivalents' not in row
    assert 'CONDITIONAL EXAMPLE ONLY' in row['comparison_status']


def test_different_gas_reference_is_retained_without_nominal_stp_failure_verdict():
    p=proposal();p['streams'][1]['gas_reference']={'temperature_K':298.15,'pressure_bar':1.0}
    packet=build({'raw_proposal':p});out=packet['delivered_final_outcome']
    assert out['streams'][1]['gas_reference']==p['streams'][1]['gas_reference']
    arith=packet['deterministic_verification_sheet']['delivered_arithmetic']
    assert 'gas_streams' not in arith
    row=arith['conditional_nominal_gas_reference_examples'][0]
    assert 'channel_flow_matches_within_rounding' not in row
    assert 'different reference does not establish a numerical defect' in row['comparison_status']
