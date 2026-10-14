from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

PATH=Path(__file__).with_name('na_format_recovery.py')
spec=importlib.util.spec_from_file_location('recovery_under_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def context():
    reader=m.Reader();camp=m.DEFAULT_CAMPAIGN
    legacy=m.load_legacy(camp,reader)
    rubric=reader.read(camp/'frozen/outcome_rubric.json')
    packet=reader.read(camp/'packets/N2-91B57A73.json')
    response=reader.read(camp/'judgments/qwen/N2-91B57A73/parsed_response.json')
    return legacy,rubric,packet,response


def test_only_two_na_scores_change_and_applicable_grade_unchanged(context):
    legacy,rubric,packet,raw=context;before=deepcopy(raw)
    result,diff,error=m.recover_na_only(raw,rubric,packet,legacy.validate_response)
    assert error is None
    assert len(diff)==2
    assert {d['criterion_id'] for d in diff}=={'UO-08','UO-09'}
    assert raw==before
    assert all(d['before']==4 and d['after'] is None for d in diff)
    assert [r for r in raw['criterion_scores'] if r['applicability']=='APPLICABLE']==[r for r in result['criterion_scores'] if r['applicability']=='APPLICABLE']
    assert legacy.validate_response(result,rubric,packet)==[]

@pytest.mark.parametrize('field,value',[('critical_error',True),('severity','CRITICAL'),('score','4'),('score',False),('score',5)])
def test_rejects_other_na_repairs(context,field,value):
    legacy,rubric,packet,raw=context;raw=deepcopy(raw)
    row=next(r for r in raw['criterion_scores'] if r['criterion_id']=='UO-08');row[field]=value
    result,diff,error=m.recover_na_only(raw,rubric,packet,legacy.validate_response)
    assert result is None and not diff and error


def test_rejects_wrong_applicability(context):
    legacy,rubric,packet,raw=context;raw=deepcopy(raw)
    raw['criterion_scores'][0]['applicability']='NOT_APPLICABLE'
    result,diff,error=m.recover_na_only(raw,rubric,packet,legacy.validate_response)
    assert result is None and 'applicability' in error


def test_never_fixes_applicable_score(context):
    legacy,rubric,packet,raw=context;raw=deepcopy(raw)
    raw['criterion_scores'][0]['score']='4'
    result,diff,error=m.recover_na_only(raw,rubric,packet,legacy.validate_response)
    assert result is None and 'other historical validation errors' in error


def test_rejects_na_that_packet_does_not_authorize(context):
    legacy,rubric,packet,raw=context;packet=deepcopy(packet)
    packet['case_context']['applicability']['gas_process']=True
    result,diff,error=m.recover_na_only(raw,rubric,packet,legacy.validate_response)
    assert result is None and 'applicability' in error


def test_rejects_wrong_candidate_or_incomplete_set(context):
    legacy,rubric,packet,raw=context
    bad=deepcopy(raw);bad['candidate_id']='different'
    assert m.recover_na_only(bad,rubric,packet,legacy.validate_response)[0] is None
    bad=deepcopy(raw);bad['criterion_scores'].pop()
    assert m.recover_na_only(bad,rubric,packet,legacy.validate_response)[0] is None


def test_attempt_order_is_base_then_numeric(tmp_path):
    for name in ['attempt_10','attempt_2','attempt_3']:
        m.write(tmp_path/'attempts'/name/'status.json',{})
    m.write(tmp_path/'status.json',{})
    assert [p.parent.name for p in m.attempt_paths(tmp_path)]==[tmp_path.name,'attempt_2','attempt_3','attempt_10']


def test_original_valid_grade_preempts_earlier_recoverable_invalid(tmp_path,context):
    legacy,rubric,packet,raw=context
    root=tmp_path/'campaign';out=tmp_path/'recovery';cid=raw['candidate_id']
    base=root/'judgments/qwen'/cid
    m.write(base/'status.json',{'status':'invalid'})
    m.write(base/'parsed_response.json',raw)
    valid=deepcopy(raw)
    for r in valid['criterion_scores']:
        if r['applicability']=='NOT_APPLICABLE':r['score']=None
    m.write(base/'attempts/attempt_2/status.json',{'status':'valid'})
    m.write(base/'attempts/attempt_2/parsed_response.json',valid)
    result,_=m.resolve_cell(root,out,{'candidate_id':cid},'qwen',rubric,packet,legacy,m.Reader())
    assert result['kind']=='original_valid'
    assert not out.exists()


def test_raw_truncation_is_not_recovered():
    with pytest.raises(ValueError,match='truncated'):
        m.parse_raw_object({'text':'{}','finish_reason':'length'})


def test_earliest_qualifying_attempt_selected_even_if_later_grade_higher(tmp_path,context):
    legacy,rubric,packet,raw=context
    root=tmp_path/'campaign';out=tmp_path/'recovery';cid=raw['candidate_id'];base=root/'judgments/qwen'/cid
    m.write(root/'packets'/f'{cid}.json',packet)
    m.write(out/'rule_declaration.json',{'schema_version':'test'})
    earlier=deepcopy(raw);earlier['criterion_scores'][0]['score']=3;earlier['criterion_scores'][0]['severity']='MINOR'
    for folder,response in [(base,earlier),(base/'attempts/attempt_2',raw)]:
        m.write(folder/'status.json',{'status':'invalid'})
        m.write(folder/'raw_response.json',{'text':json.dumps(response),'finish_reason':'stop'})
        m.write(folder/'parsed_response.json',response)
    result,_=m.resolve_cell(root,out,{'candidate_id':cid},'qwen',rubric,packet,legacy,m.Reader())
    assert result['kind']=='recovered_na_serialization'
    assert result['response']['criterion_scores'][0]['score']==3
    assert result['recovery']['source_attempt']==str(base)
