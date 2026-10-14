from copy import deepcopy
import importlib.util,json,sys
from pathlib import Path
import pytest
from flora_translate.translation_llm import TranslationLLM,_parse_json

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('isolated_proposal_payload',BASE/'proposal_payload.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
extract=m.extract_proposal_payload

def test_flat_and_nested_preserve_identical_scientific_payload():
    p={'flow_rate_mL_min':.5,'temperature_C':21,'wavelength_nm':420,'streams':[{'gas_flow_sccm':5.34,'gas_flow_actual_mL_min':.089}]};flat=extract(p);nested=extract({'recommended_disposition':'SCREEN','disposition_rationale':'unmeasured yield','proposal':p})
    assert flat.payload==nested.payload==p and flat.envelope=={}
    assert nested.envelope=={'recommended_disposition':'SCREEN','disposition_rationale':'unmeasured yield'}
    nested.payload['streams'][0]['gas_flow_sccm']=1
    assert p['streams'][0]['gas_flow_sccm']==5.34

def test_block_disposition_remains_explicit_metadata_not_execution_authority():
    r=extract({'recommended_disposition':'BLOCK','disposition_rationale':'required hardware missing','proposal':{'flow_rate_mL_min':.1}})
    assert r.envelope['recommended_disposition']=='BLOCK'
    assert r.envelope['disposition_rationale']=='required hardware missing'

@pytest.mark.parametrize('value',[{}, {'confidence':'high'}, {'streams':[]}, {'proposal':{}}, {'proposal':None}, {'proposal':'{"flow_rate_mL_min":0.1}'}, {'proposal':{'unrelated':'chemistry'}}])
def test_malformed_or_empty_payload_cannot_become_default_design(value):
    with pytest.raises(ValueError):extract(value)

@pytest.mark.parametrize('field,value',[('temperature_C',80),('flow_rate_mL_min',.5),('streams',[{'contents':['wrong feed']}])])
def test_flat_nested_ambiguity_fails_closed_even_if_one_value_matches(field,value):
    with pytest.raises(ValueError):extract({'proposal':{'flow_rate_mL_min':.5},field:value})

def test_all_saved_fresh_wrappers_and_previous_flat_answers_preserve_raw_payload():
    rows=json.loads((BASE.parent/'wrapper_loss_comparison.json').read_text());assert len(rows)==9
    for row in rows:
        path=ROOT/row['path'];before=path.read_bytes();event=json.loads(path.read_text().splitlines()[row['event_index']]);raw=_parse_json(event['response_text']);original=deepcopy(raw)
        r=extract(raw);assert r.payload==raw.get('proposal',raw) and raw==original and path.read_bytes()==before
        normalized=TranslationLLM._normalize_proposal_data(deepcopy(r.payload))
        for field in ['temperature_C','wavelength_nm','light_setup','flow_rate_mL_min','reactor_volume_mL']:
            assert normalized.get(field)==r.payload.get(field)
        assert len(normalized['streams'])==len(r.payload['streams'])
