"""Independent complete-judge sensitivity; consumes audited original integers only."""
from pathlib import Path
import hashlib
import json
import statistics

OUT = Path(__file__).resolve().parent / 'schema_routing_followup'
JUDGES = ('qwen', 'openai', 'claude')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    verification = OUT / 'independent_verification.json'
    rule = OUT / 'common_complete_judges_rule.json'
    declared = read(rule)
    assert declared['additional_model_calls'] == 0
    audit = read(verification)
    scores = audit['raw_scores']
    cases = sorted({r['case'] for r in scores})
    arms = ('One-shot', 'FlowPilot')
    assert len(cases) == 3
    full_cells = {(arm, case) for arm in arms for case in cases}
    included, excluded = [], []
    for judge in JUDGES:
        rows = [r for r in scores if r['judge'] == judge]
        if len(rows) == 6 and {(r['architecture'], r['case']) for r in rows} == full_cells:
            included.append(judge)
        else:
            excluded.append({'judge': judge, 'accepted_cells': len(rows), 'required_cells': 6, 'reason': 'Not complete across all six paired outcomes; both arms excluded symmetrically.'})
    assert included, 'No judge is complete across all six outcomes.'
    selected = [r for r in scores if r['judge'] in included]
    groups = {}
    for arm in arms:
        rows = [r for r in selected if r['architecture'] == arm]
        assert len(rows) == 3 * len(included)
        groups[arm] = {
            'mean': statistics.mean(r['score'] for r in rows),
            'critical_flags': sum(r['critical_flags'] for r in rows),
            'n_outcomes': 3,
            'n_judgments': len(rows),
            'by_case': {case: statistics.mean(r['score'] for r in rows if r['case'] == case) for case in cases},
            'by_judge': {judge: statistics.mean(r['score'] for r in rows if r['judge'] == judge) for judge in included},
        }
    result = {
        'label': 'Post-hoc paired sensitivity using judges complete for every outcome; not the preregistered primary result',
        'rule_path': str(rule), 'rule_sha256': sha(rule),
        'independent_verification_path': str(verification), 'verification_sha256': sha(verification),
        'primary_expected_judgments': 18,
        'primary_strict_accepted_judgments': audit['strict_accepted_judgments'],
        'primary_complete': audit['complete'],
        'primary_missing': audit['missing'],
        'included_judges': included, 'excluded_judges': excluded,
        'selection_basis': 'Completeness of all six paired outcomes only. No access to score direction in eligibility condition.',
        'groups': groups,
        'difference_FP_minus_OS': groups['FlowPilot']['mean'] - groups['One-shot']['mean'],
        'case_differences_FP_minus_OS': {case: groups['FlowPilot']['by_case'][case] - groups['One-shot']['by_case'][case] for case in cases},
        'judge_differences_FP_minus_OS': {judge: groups['FlowPilot']['by_judge'][judge] - groups['One-shot']['by_judge'][judge] for judge in included},
        'raw_scores': selected,
        'new_model_calls': 0, 'original_scores_changed': False, 'missing_scores_imputed': False,
        'limitation': 'One generation per case and architecture. This diagnostic cannot establish a reliable superiority claim or isolate token-count effects. Original three-judge incompleteness remains explicit.'
    }
    (OUT / 'common_complete_judge_sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('primary_strict_accepted_judgments', 'included_judges', 'excluded_judges', 'groups', 'difference_FP_minus_OS')}, indent=2))


if __name__ == '__main__':
    build()
