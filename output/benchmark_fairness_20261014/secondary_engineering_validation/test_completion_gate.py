from pathlib import Path
from types import ModuleType
import pytest

P=Path(__file__).with_name('replay_isolated_pump_patch.py')
m=ModuleType('secondary_gate_test');m.__file__=str(P.resolve());exec(compile(P.read_text(),str(P),'exec'),m.__dict__)

@pytest.fixture
def state(monkeypatch,tmp_path):
    from ablation_test.scripts import run_fair_practical_benchmark as d
    root=tmp_path;out=root/'secondary';out.mkdir();campaign=root/'campaign';campaign.mkdir()
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'OUT',out);monkeypatch.setattr(m,'CAMPAIGN',campaign)
    m.write(out/'gate_amendment_20261014.json',{})
    rows=[{'candidate_id':f'C{i}'} for i in range(21)]
    summary={'expected_judgments':63,'accepted_judgments':62,'complete':False,'source_hashes_before_and_after':{},'recovered_cells':[]}
    m.write(root/'format_recovery/summary_all.json',summary)
    monkeypatch.setattr(d,'configure',lambda *a:None);monkeypatch.setattr(d,'verify',lambda *a:None);monkeypatch.setattr(d,'keys',lambda *a:rows)
    monkeypatch.setattr(d,'accepted',lambda out,judge,row:None if row['candidate_id']=='C20' and judge=='qwen' else ('accepted',{}))
    return campaign,summary,root


def test_three_invalid_attempts_settle_without_accepting_score(state):
    campaign,summary,root=state
    for suffix in ('','attempts/attempt_2','attempts/attempt_3'):
        m.write(campaign/'judgments/qwen/C20'/suffix/'status.json',{'status':'invalid'})
    proof=m.require_complete_primary()
    assert len(proof['settled_cells'])==63 and proof['accepted_judgments']==62
    assert proof['valid_score_panel_complete'] is False
    assert sum(r['state']=='exhausted_invalid_or_failed_attempts' for r in proof['settled_cells'])==1


def test_pending_cell_blocks_authoritative_replay(state):
    campaign,_,_=state
    m.write(campaign/'judgments/qwen/C20/status.json',{'status':'invalid'})
    with pytest.raises(AssertionError,match='pending/unattempted'):m.require_complete_primary()


def test_declared_recovery_permitted_without_reinterpreting_raw(state):
    _,summary,root=state
    summary['accepted_judgments']=63;summary['complete']=True
    summary['recovered_cells']=[{'candidate_id':'C20','judge':'qwen'}]
    m.write(root/'format_recovery/summary_all.json',summary)
    proof=m.require_complete_primary()
    assert sum(r['state']=='declared_na_serialization_recovery' for r in proof['settled_cells'])==1
