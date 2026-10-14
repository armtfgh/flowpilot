"""Offline protocol and accounting checks; no network or inference calls."""
from decimal import Decimal
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).with_name('run_campaign.py')
SPEC = importlib.util.spec_from_file_location('production_repair_campaign', PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_preflight_preserves_cases_adapter_seeds_and_nine_baseline_grades():
    campaign.configure()
    evidence = campaign.preflight()
    assert evidence['one_shot_grades_reused'] == 9
    assert evidence['one_shot_outputs_reused'] == 3
    assert evidence['new_flowpilot_attempts'] == 3


def test_all_planned_generations_are_flowpilot_once_per_case():
    campaign.configure()
    rows = campaign.plan_cells()
    assert len(rows) == len({r['case_id'] for r in rows}) == 3
    assert all(r['architecture_key'] == 'flowpilot' for r in rows)
    assert all(r['candidate_budget'] == 2 for r in rows)
    assert {r['repeat_id'] for r in rows} == {'repeat_01'}


def test_complete_primary_uses_equal_judges_and_cases():
    rows = [dict(architecture=arm, judge=judge, score=value, critical_flags=0)
            for arm, value in [('One-shot', .8), ('FlowPilot', .9)]
            for judge in campaign.primary.JUDGES for _ in range(3)]
    result = campaign.summarize(rows, tuple(campaign.primary.JUDGES))
    assert result['complete']
    assert result['flowpilot_minus_one_shot'] == pytest.approx(.1)


def test_missing_qwen_grade_prevents_primary_but_not_fixed_secondary():
    rows = [dict(architecture=arm, judge=judge, score=value, critical_flags=0)
            for arm, value in [('One-shot', .9), ('FlowPilot', .8)]
            for judge in campaign.primary.JUDGES for _ in range(3)]
    rows.remove(next(r for r in rows if r['architecture'] == 'FlowPilot' and r['judge'] == 'qwen'))
    primary = campaign.summarize(rows, tuple(campaign.primary.JUDGES))
    secondary = campaign.summarize(rows, campaign.SECONDARY_JUDGES)
    assert not primary['complete']
    assert primary['groups']['FlowPilot']['mean'] is None
    assert primary['flowpilot_minus_one_shot'] is None
    assert secondary['complete']
    assert secondary['flowpilot_minus_one_shot'] == pytest.approx(-.1)


def test_secondary_cannot_substitute_another_judge():
    assert campaign.SECONDARY_JUDGES == ('openai', 'claude')
    rows = [dict(architecture=arm, judge=judge, score=1., critical_flags=0)
            for arm in ('One-shot', 'FlowPilot')
            for judge in ('qwen', 'openai') for _ in range(3)]
    assert not campaign.summarize(rows, campaign.SECONDARY_JUDGES)['complete']


def test_valid_grade_is_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.primary, 'accepted', lambda *args: (Path('grade'), {}))
    assert campaign.remaining_attempts(tmp_path, 'claude', {'candidate_id': 'N2-X'}) == 0


def test_prior_invalid_attempts_count_toward_total_three(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.primary, 'accepted', lambda *args: None)
    row = {'candidate_id': 'N2-X'}
    assert campaign.remaining_attempts(tmp_path, 'claude', row) == 3
    base = tmp_path/'judgments/claude/N2-X'
    for index in range(3):
        target = base/f'attempt{index}/status.json'
        target.parent.mkdir(parents=True)
        target.write_text('{"status":"invalid"}')
    assert campaign.remaining_attempts(tmp_path, 'claude', row) == 0


def test_interrupted_judgment_is_not_overwritten_or_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.primary, 'accepted', lambda *args: None)
    folder = tmp_path/'judgments/claude/N2-X'
    folder.mkdir(parents=True)
    (folder/'prompt.json').write_text('{"system":"started"}')
    assert campaign.remaining_attempts(tmp_path, 'claude', {'candidate_id': 'N2-X'}) == 0


def test_separate_cap_is_five_dollars(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.primary, 'install_spending_guard', lambda *args: None)
    campaign.install_spending_guard(tmp_path)
    assert campaign.primary.SPEND_LIMIT == Decimal('5')
    assert campaign.cost(tmp_path)['limit_usd'] == 5


def test_budget_reservation_blocks_before_any_paid_call(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.primary, 'SPEND_LIMIT', Decimal('5'))
    monkeypatch.setattr(campaign.primary, 'cost', lambda out: {'estimated_paid_usd': 4.99, 'pending_reserved_usd': 0})
    invoked = []
    with pytest.raises(RuntimeError, match='campaign \\$5 guard'):
        campaign.guarded_create(tmp_path, 'anthropic', lambda **kw: invoked.append(kw),
                                {'model': 'claude-sonnet-4-6', 'max_tokens': 7500,
                                 'messages': [{'role': 'user', 'content': 'Test'}]})
    assert invoked == []
