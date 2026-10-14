"""Offline integrity tests for the fixed historical-evaluator bridge."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ablation_test.scripts import run_fair_practical_benchmark as fair


def test_historical_selection_is_exact_postfix_plus_all_current():
    rows=fair.select_saved()
    assert len(rows)==15
    historical=[(p,r) for p,c,r in rows if c=='historical_manuscript']
    assert len(historical)==6
    assert all('postfix_20260820' in str(p) for p,r in historical)
    assert {r['architecture'] for p,r in historical}=={'One-shot','FlowPilot'}
    assert len([r for p,c,r in rows if c=='current_practical'])==9


def test_judge_panel_matches_actual_historical_configuration():
    assert fair.JUDGES['openai']['model']=='gpt-5.4-2026-03-05'
    assert fair.JUDGES['claude']['model']=='claude-sonnet-4-6'
    assert fair.JUDGES['qwen']['model']=='/models/Qwen3.6-27B'


def test_case_facts_retained_with_shared_output_contract():
    baseline=dict(fair.base.practical_cases())
    for label,case in fair.fair_cases():
        old=baseline[label]
        assert case.protocol==old.protocol
        assert case.inventory==old.inventory
        assert case.hard_constraints==old.hard_constraints
        assert case.excluded_record_ids==old.excluded_record_ids
        assert case.chemistry_identity_confirmation=={}
        assert case.objective.startswith(old.objective)
        assert fair.SHARED_OUTPUT in case.objective
        assert 'REQUEST' in case.design_input_text


def test_normal_one_shot_restored_and_six_fixed_cells():
    fair.configure()
    assert fair.runner._general_one_shot is fair.NORMAL_ONE_SHOT
    assert fair.NORMAL_ONE_SHOT.__name__=='_general_one_shot'
    rows=fair.campaign.campaign_cells(fair.fair_cases())
    assert len(rows)==6
    assert [r['candidate_budget'] for r in rows].count(1)==3
    assert [r['candidate_budget'] for r in rows].count(2)==3
    assert {r['repeat_id'] for r in rows}=={'repeat_01'}


def test_historical_prompt_has_no_new_addendum_or_catalog():
    module=fair.historical_module()
    rubric=fair.read(fair.HISTORICAL/'frozen/outcome_rubric.json')
    system,user=module.judge_prompt(rubric,{})
    assert 'CONTENT AND NUMERICAL VERIFICATION' not in user
    assert 'EVIDENCE PATH CATALOG' not in user
    assert 'deterministic verification sheet as authoritative' in system
    assert len(rubric['criteria'])==14


def test_gpt54_cost_uses_actual_cached_and_output_rates():
    value=fair.usage_cost('openai',{'prompt_tokens':1000,'completion_tokens':2000,'prompt_tokens_details':{'cached_tokens':400}})
    assert value==Decimal('0.0316')


def test_gpt54_transport_and_ledger_are_bounded(tmp_path):
    seen={}
    usage={'prompt_tokens':1000,'completion_tokens':2000,'prompt_tokens_details':{'cached_tokens':400}}
    def call(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(model='gpt-5.4-2026-03-05',usage=SimpleNamespace(model_dump=lambda:usage))
    fair.guarded_create(tmp_path,'openai',call,{'model':'gpt-5.4-2026-03-05','max_tokens':7500,'temperature':0,'seed':42,'messages':[]})
    assert seen['max_completion_tokens']==7500
    assert 'max_tokens' not in seen
    assert 'temperature' not in seen
    assert 'reasoning_effort' not in seen
    assert seen['seed']==42
    assert fair.cost(tmp_path)['estimated_paid_usd']==.0316
    assert fair.cost(tmp_path)['pending_reserved_usd']==0
    rows=fair.base.ledger_rows(tmp_path)
    assert rows[0]['max_output_tokens']==7500
    assert rows[-1]['returned_model']=='gpt-5.4-2026-03-05'


@pytest.mark.parametrize('field',['max_tokens','max_completion_tokens','max_output_tokens'])
def test_guard_uses_every_token_limit_key(tmp_path,field):
    calls=[]
    with pytest.raises(RuntimeError,match='reservation'):
        fair.guarded_create(tmp_path,'openai',lambda **kw:calls.append(kw),{'model':'gpt-5.4-2026-03-05',field:1_000_000})
    assert not calls


def test_guard_rejects_other_paid_model(tmp_path):
    with pytest.raises(RuntimeError,match='Unexpected paid model'):
        fair.guarded_create(tmp_path,'openai',lambda **kw:None,{'model':'gpt-4o','max_tokens':7500})


def test_existing_valid_grade_is_not_retried(tmp_path,monkeypatch):
    row={'candidate_id':'N2-test'}
    p=tmp_path/'status.json';p.write_text('{"status":"valid"}')
    monkeypatch.setattr(fair,'accepted',lambda *a,**kw:(p,{}))
    monkeypatch.setattr(fair.llm,'call_model_text',lambda **kw:pytest.fail('Valid grade was repeated'))
    assert fair.call_judge(tmp_path,'openai',row)['status']=='valid'


def test_failed_postprocessing_does_not_erase_delivered_design(tmp_path):
    original={'variant':'general_one_shot','raw_proposal':{'concentration_M':'0.126','chemistry_notes':'source text'},'proposal':{'concentration_M':.126}}
    (tmp_path/'result.json').write_text(json.dumps(original))
    (tmp_path/'run_summary.json').write_text('{"status":"failed"}')
    result,present,complete=fair.generation_artifact(tmp_path,'general_one_shot')
    assert present and not complete
    assert result==original


def test_historical_loader_never_writes_archive_bytecode():
    path=fair.HISTORICAL/'frozen/source_code'
    before={str(p) for p in path.rglob('*')}
    fair.historical_module()
    assert before=={str(p) for p in path.rglob('*')}


def test_judging_requires_frozen_prepared_stage(tmp_path,monkeypatch):
    monkeypatch.setattr(fair,'verify',lambda output:None)
    with pytest.raises(AssertionError,match='Prepare and freeze'):
        fair.judge(tmp_path,'openai','B')


def test_initialize_prepares_fifteen_byte_identical_packets_offline(tmp_path,monkeypatch):
    fair.configure()
    monkeypatch.setattr(fair.base,'source_files',lambda:iter([Path(fair.__file__)]))
    monkeypatch.setattr(fair,'archive_hashes',lambda:{})
    out=tmp_path/'campaign'
    fair.initialize(out)
    assert len(fair.keys(out,'A'))==15
    assert (out/'prepared_manifest_A.json').exists()
    assert not (out/'candidate_key_B.json').exists()
    for row in fair.keys(out,'A'):
        assert (out/'packets'/f"{row['candidate_id']}.json").read_bytes()==(fair.ROOT/row['source_packet']).read_bytes()
    fair.verify(out)
    frozen=out/'frozen/full_source'/Path(fair.__file__).relative_to(fair.ROOT)
    frozen.write_text(frozen.read_text()+'\n# tampered\n')
    with pytest.raises(AssertionError,match='Frozen source copy changed'):
        fair.verify(out)
