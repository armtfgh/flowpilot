from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json
import pytest
from flora_translate.translation_llm import TranslationLLM as Original,_parse_json
from flora_translate.schemas import FlowProposal

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]
spec=importlib.util.spec_from_file_location('isolated_translation_guard',BASE/'candidate_translation_llm.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
rows=json.loads((BASE.parents[1]/'wrapper_loss_comparison.json').read_text())
def response(row):return json.loads((ROOT/row['path']).read_text().splitlines()[row['event_index']])['response_text']

@pytest.mark.parametrize('raw',[None,[],{},'',{'analysis':'still deciding'},{'proposal':{}},{'proposal':{'flow_rate_mL_min':.1},'flow_rate_mL_min':.2}])
def test_invalid_boundary_rejected_before_default_mutation(raw):
    before=deepcopy(raw)
    with pytest.raises(ValueError):candidate.TranslationLLM._normalize_proposal_data(raw)
    assert raw==before

def test_three_frozen_wrappers_reject_and_six_previous_flat_answers_unchanged():
    wrappers=flat=0
    for row in rows:
        raw=_parse_json(response(row));before=deepcopy(raw)
        if row['has_nested_proposal']:
            wrappers+=1
            with pytest.raises(ValueError):candidate.TranslationLLM._normalize_proposal_data(raw)
            assert raw==before
        else:
            flat+=1
            a=candidate.TranslationLLM._normalize_proposal_data(deepcopy(raw))
            b=Original._normalize_proposal_data(deepcopy(raw))
            assert a==b
            assert FlowProposal(**a)==FlowProposal(**b)
    assert wrappers==3 and flat==6

def test_generate_retries_wrapper_then_flat_and_preserves_lamp(monkeypatch):
    wrap=next(r for r in rows if r['has_nested_proposal'] and r['case']=='photochemical_oxidation')
    flat=next(r for r in rows if not r['has_nested_proposal'] and r['case']=='photochemical_oxidation')
    replies=iter([response(wrap),response(flat)]);calls=[]
    def fake(**kwargs):calls.append(kwargs);return SimpleNamespace(text=next(replies))
    monkeypatch.setattr(candidate,'call_model_text',fake)
    p=candidate.TranslationLLM().generate('Internal flat FlowProposal schema','Example protocol')
    assert len(calls)==2 and calls[1]['api_name']=='translation_llm_retry'
    assert p.wavelength_nm==420 and p.temperature_C==21 and len(p.streams)==2 and p.flow_rate_mL_min==.5

@pytest.mark.parametrize('bad',[{'recommended_disposition':'BLOCK','disposition_rationale':'no feasible apparatus','proposal':{'flow_rate_mL_min':.1}},{'analysis':'not a proposal'},{}])
def test_three_bad_responses_raise_after_existing_retry_and_repair(monkeypatch,bad):
    calls=[]
    def fake(**kwargs):calls.append(kwargs);return SimpleNamespace(text=json.dumps(bad))
    monkeypatch.setattr(candidate,'call_model_text',fake)
    with pytest.raises(ValueError):candidate.TranslationLLM().generate('Internal schema','Example protocol')
    assert [x['api_name'] for x in calls]==['translation_llm','translation_llm_retry','translation_llm_repair']
    assert 'INVALID OUTPUT:' in calls[-1]['user_content']
    assert json.dumps(bad) in calls[-1]['user_content']

def test_production_source_unchanged():
    assert hashlib.sha256((ROOT/'flora_translate/translation_llm.py').read_bytes()).hexdigest()==(BASE/'production_translation_llm_sha256.txt').read_text().strip()
