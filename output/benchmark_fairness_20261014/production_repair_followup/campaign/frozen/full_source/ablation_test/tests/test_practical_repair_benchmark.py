from copy import deepcopy
from pathlib import Path
import pytest
from ablation_test.scripts import run_practical_repair_benchmark as repair
from ablation_test.src.newgen_2_outcome import packet_gate

@pytest.fixture(autouse=True)
def restore_campaign_configuration(monkeypatch):
    # These script entry points configure shared modules; tests must leave them
    # exactly as they were so the original pilot remains independently testable.
    for name in ('MODELS','REPEAT_IDS','JUDGES','TEMPERATURE','CANDIDATE_BUDGET',
                 'SEED_NAMESPACE','MATCHED_SEED_ACROSS_ARCHITECTURES','selected_cases',
                 'build_outcome_packet','deterministic_sheet','ARCHITECTURES','CANDIDATE_BUDGETS','ORACLES'):
        monkeypatch.setattr(repair.campaign,name,getattr(repair.campaign,name))
    monkeypatch.setattr(repair.runner,'_general_one_shot',repair.runner._general_one_shot)
    monkeypatch.setattr(repair.judging,'judge_prompt',repair.judging.judge_prompt)
    monkeypatch.setattr(repair,'GUIDANCE',repair.GUIDANCE)
    monkeypatch.setattr(repair,'JUDGE_ADDENDUM',repair.JUDGE_ADDENDUM)

def test_three_arms_hold_facts_and_output_format_constant(tmp_path):
    repair.configure(tmp_path)
    cells=repair.campaign.campaign_cells(repair.base.practical_cases())
    assert len(cells)==9
    assert {r['model_id'] for r in cells}=={'claude-sonnet-4-6'}
    assert {r['architecture'] for r in cells}==set(repair.ARMS)
    for label,case in repair.base.practical_cases():
        rows=[r for r in cells if r['case_id']==case.case_id]
        assert len({r['design_input_sha256'] for r in rows})==1
        assert not case.chemistry_identity_confirmation
    assert all(r['candidate_budget']==(2 if r['variant']=='full' else 1) for r in cells)
    assert 'ordinary prose or tables' in repair.GUIDANCE
    assert all(name not in repair.GUIDANCE.lower() for name in ['dmaol','fmoc','cuaac','3-azetidinol'])

def test_symmetric_judge_addendum_keeps_rubric_and_labels_out(tmp_path):
    repair.configure(tmp_path)
    result={'variant':'general_one_shot','proposal':{'chemistry_notes':'The feed is 0.1 mL/min.'}}
    before=deepcopy(result)
    packet=repair.packet(candidate='N2-TEST',case_context={},result=result,deterministic_verification={},has_gas=True,is_multistage=False)
    assert not packet_gate(packet)
    assert 'provisional supporting artifacts' in packet['evaluation_boundary']
    assert packet['delivered_final_outcome']['chemistry_and_operations']['chemistry_notes']==result['proposal']['chemistry_notes']
    assert 'one sccm equals one standard mL/min' in repair.JUDGE_ADDENDUM
    assert 'low confidence' in repair.JUDGE_ADDENDUM
    assert result==before
