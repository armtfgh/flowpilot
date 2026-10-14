from copy import deepcopy
from ablation_test.scripts.run_practical_input_pilot import practical_cases, practical_packet, OBJECTIVE
from ablation_test.scripts.run_manuscript_five_case_benchmark import load_frozen_cases
from ablation_test.src.newgen_2_outcome import packet_gate

def test_practical_input_preserves_full_inventory_and_batch_quantities():
    originals=dict(load_frozen_cases())
    for label,case in practical_cases():
        assert case.inventory==originals[label].inventory
        assert case.excluded_record_ids==originals[label].excluded_record_ids
        assert case.objective==OBJECTIVE
        assert 'Report feed preparation' not in case.protocol
        assert 'HARD CONSTRAINTS:' not in case.design_input_text
        assert not case.chemistry_identity_confirmation
    cases=dict(practical_cases())
    assert '0.792 g/mL' in cases['Hydrogenolysis'].protocol
    assert '3.8 wt%' in cases['Hydrogenolysis'].protocol
    assert '150 °C' in cases['CuAAC'].protocol
    assert '420 nm' in cases['Photochemical oxidation'].protocol

def test_prose_survives_without_numeric_inference_or_missing_field_penalties():
    text='Use 0.2 mL/min liquid in a 1 mL coil. This is a nominal five-minute liquid space time.'
    result={'variant':'general_one_shot','proposal':{'chemistry_notes':text}}
    before=deepcopy(result)
    packet=practical_packet(candidate='P-1',case_context={},result=result,
        deterministic_verification={'automatic_failure':'must be removed'},has_gas=True,is_multistage=False)
    assert not packet_gate(packet)
    assert packet['delivered_final_outcome']['chemistry_and_operations']['chemistry_notes']==text
    assert packet['delivered_final_outcome']['parameters']=={}
    assert 'automatic_failure' not in str(packet)
    assert 'delivered_arithmetic' not in packet['deterministic_verification_sheet']
    assert 'formatting earns or loses credit' in packet['evaluation_boundary']
    assert result==before

def test_paid_usage_accounts_for_cache_and_provider_fields():
    from ablation_test.scripts.run_practical_input_pilot import usage_cost
    from pytest import approx
    assert usage_cost('anthropic', {'input_tokens':1000,'output_tokens':100,
        'cache_read_input_tokens':2000,'cache_creation_input_tokens':1000,
        'cache_creation':{'ephemeral_1h_input_tokens':200}}) == approx(.0093)
    assert usage_cost('openai', {'prompt_tokens':1000,'completion_tokens':100,
        'prompt_tokens_details':{'cached_tokens':200}}) == approx(.00325)

def test_same_claude_generator_in_both_arms_and_independent_primary_judge():
    from ablation_test.scripts import run_practical_input_pilot as pilot
    pilot.configure()
    cells=pilot.campaign.campaign_cells(pilot.practical_cases())
    assert len(cells)==6
    assert {row['model_id'] for row in cells}=={'claude-sonnet-4-6'}
    assert pilot.campaign.JUDGES==('openai','qwen')
    assert pilot.PRIMARY_JUDGE=='openai'

def test_budget_guard_blocks_call_before_transport(tmp_path, monkeypatch):
    from ablation_test.scripts import run_practical_input_pilot as pilot
    import pytest
    monkeypatch.setattr(pilot,'SPEND_LIMIT',.001)
    called=[]
    with pytest.raises(RuntimeError,match='guard'):
        pilot.guarded_create(tmp_path,'anthropic',lambda **kw: called.append(kw),
            {'model':'claude-sonnet-4-6','messages':[],'max_tokens':8192})
    assert not called
