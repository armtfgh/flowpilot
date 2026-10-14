from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import pytest

PATH=Path(__file__).with_name('run_followup.py')
spec=importlib.util.spec_from_file_location('schema_routing_followup_driver',PATH)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(driver.primary.runner,'_production_full',driver.NATIVE_PRODUCTION)
    monkeypatch.setattr(driver.primary.runner,'_general_one_shot',driver.primary.NORMAL_ONE_SHOT)
    driver.primary.configure(driver.PRIMARY)
    return driver


def test_semantics_are_preserved_verbatim(configured):
    d=configured
    assert d.REMOVED_SERIALIZATION+d.RETAINED_GUIDANCE==d.primary.SHARED_OUTPUT
    for requirement in ('Distinguish inlet/STP gas flow from\npressure-corrected in-channel gas flow.',
                        'Ensure reactor_volume_mL is consistent\nwith flow and residence time.',
                        'State assumptions rather than inventing missing\nfacts.',
                        'recommended_disposition must be exactly EXECUTE, SCREEN, or BLOCK.',
                        'Use BLOCK\nwhen the hard constraints make a safe design physically or operationally\nimpossible.',
                        'Use SCREEN for a feasible first experiment that still requires\nexperimental validation.',
                        'Use EXECUTE only for a directly executable design\nsupported by sufficient evidence.'):
        assert requirement in d.INTERNAL_OBJECTIVE
    assert 'one JSON object' not in d.INTERNAL_OBJECTIVE
    assert 'The proposal uses these fields:' not in d.INTERNAL_OBJECTIVE
    assert 'Each stream is an object with' not in d.INTERNAL_OBJECTIVE


def test_all_three_original_seeds_and_budgets_preserved(configured):
    d=configured
    expected=[r for r in d.primary.read(d.PRIMARY/'frozen/execution_plan.json')['cells'] if r['architecture_key']=='flowpilot']
    assert d.plan_cells()==expected
    assert len(expected)==3
    assert all(r['candidate_budget']==2 for r in expected)


def test_only_objective_changes_inside_native_pipeline(configured,monkeypatch):
    d=configured
    seen={}
    class Recorder:
        def write_json(self,name,data):seen[name]=data
    def native(case,recorder,budget,temp,seed,execution):
        seen.update(case=case,budget=budget,temp=temp,seed=seed,execution=execution)
        return {'unchanged_native_result':True}
    monkeypatch.setattr(d,'NATIVE_PRODUCTION',native)
    for _,case in d.primary.fair_cases():
        before=case.manifest_payload()
        result=d.routed_production(case,Recorder(),2,.2,100,'config')
        assert result=={'unchanged_native_result':True}
        assert seen['case']==replace(case,objective=d.INTERNAL_OBJECTIVE)
        assert case.manifest_payload()==before
        assert seen['case'].protocol==case.protocol
        assert seen['case'].inventory==case.inventory
        assert seen['case'].hard_constraints==case.hard_constraints
        assert seen['seed']==100 and seen['budget']==2
        assert seen['schema_routing.json']['public_objective']==case.objective


def test_arbitrary_objectives_not_silently_rewritten(configured):
    d=configured
    case=d.primary.fair_cases()[0][1]
    with pytest.raises(AssertionError):d.routing_record(replace(case,objective='changed goal'))


def test_grade_only_new_flowpilot(configured,monkeypatch,tmp_path):
    d=configured
    (tmp_path/'prepared_manifest_B.json').write_text('{}')
    monkeypatch.setattr(d,'verify',lambda _:None)
    rows=[{'candidate_id':'os','reused_from_primary':True},{'candidate_id':'fp','reused_from_primary':False,'case':'test'}]
    monkeypatch.setattr(d.primary,'keys',lambda *a:rows)
    monkeypatch.setattr(d.primary,'accepted',lambda *a:None)
    called=[]
    monkeypatch.setattr(d.primary,'call_judge',lambda out,name,row:(called.append(row['candidate_id']) or {'status':'valid'}))
    d.judge(tmp_path,'claude')
    assert called==['fp']


def test_valid_grade_not_retried(configured,monkeypatch,tmp_path):
    d=configured
    (tmp_path/'prepared_manifest_B.json').write_text('{}')
    monkeypatch.setattr(d,'verify',lambda _:None)
    monkeypatch.setattr(d.primary,'keys',lambda *a:[{'candidate_id':'fp','reused_from_primary':False,'case':'test'}])
    monkeypatch.setattr(d.primary,'accepted',lambda *a:('valid',{}))
    monkeypatch.setattr(d.primary,'call_judge',lambda *a:pytest.fail('valid grade retried'))
    d.judge(tmp_path,'claude')


def test_existing_generation_never_replaced(configured,monkeypatch,tmp_path):
    d=configured
    monkeypatch.setattr(d,'verify',lambda _:None)
    row=d.plan_cells()[0]
    run=d.primary.campaign.run_directory(tmp_path,row)
    run.mkdir(parents=True)
    (run/'routing_attempt_started.json').write_text('{}')
    monkeypatch.setattr(d.primary.runner,'execute_cell',lambda **k:pytest.fail('regeneration'))
    d.generate(tmp_path,0)


def test_shared_guard_points_to_original_campaign():
    source=PATH.read_text()
    assert 'primary.install_spending_guard(PRIMARY)' in source
    assert 'primary.cost(PRIMARY)' in source
    assert driver.PRIMARY==driver.primary.DEFAULT


def test_actual_three_saved_one_shots(configured):
    rows=configured.primary.keys(configured.PRIMARY,'B')
    one_shots=[r for r in rows if r['architecture_key']=='one_shot']
    assert len(one_shots)==3
    assert {r['architecture'] for r in one_shots}=={'One-shot'}
    assert all((Path(r['run_directory'])/'raw_response.json').is_file() for r in one_shots)


def test_report_complete_eighteen_with_original_labels(configured,monkeypatch,tmp_path):
    d=configured
    rows=[{**r,'reused_from_primary':r['architecture_key']=='one_shot'} for r in d.primary.keys(d.PRIMARY,'B')]
    monkeypatch.setattr(d,'verify',lambda _:None)
    monkeypatch.setattr(d.primary,'keys',lambda *a:rows)
    status=tmp_path/'status.json';status.write_text('{}')
    response={'criterion_scores':[{'score':4,'applicability':'APPLICABLE','critical_error':False}]}
    status.with_name('parsed_response.json').write_text(json.dumps(response))
    monkeypatch.setattr(d.primary,'accepted',lambda *a:(status,response))
    monkeypatch.setattr(d.primary,'cost',lambda *a:{'estimated_paid_usd':0})
    summary=d.report(tmp_path)
    assert summary['complete'] and summary['accepted_judgments']==18
    assert summary['reused_one_shot_judgments']==9
    assert set(summary['groups']['schema_routing_control'])=={'One-shot','FlowPilot'}
    assert all(v['complete'] for v in summary['groups']['schema_routing_control'].values())
