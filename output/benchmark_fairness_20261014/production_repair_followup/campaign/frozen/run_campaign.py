#!/usr/bin/env python3
"""Fixed production-repair evaluation with immutable existing one-shot comparators.

No inference occurs during import, preflight, init, prepare or report. Init is
the source-freeze boundary and must run only after production repairs finish.
"""
from __future__ import annotations
import argparse
import ast
from decimal import Decimal
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts import run_fair_practical_benchmark as primary
from ablation_test.src.fair_practical_packet import build_fair_packet
from ablation_test.src.holistic_audit import build_case_context
from ablation_test.src.newgen_2_outcome import candidate_id

BASELINE = primary.DEFAULT
DEFAULT = Path(__file__).resolve().parent / 'campaign'
COHORT = 'production_repair_followup'
LIMIT = Decimal('5')
SECONDARY_JUDGES = ('openai', 'claude')
read, write, sha, now = primary.read, primary.write, primary.sha, primary.now
_PRIMARY_COST = primary.cost
_PRIMARY_GUARDED_CREATE = primary.guarded_create


def configure():
    primary.configure()


def plan_cells():
    return [r for r in primary.campaign.campaign_cells(primary.fair_cases())
            if r['architecture_key'] == 'flowpilot']


def baseline_rows():
    return [r for r in primary.keys(BASELINE, 'B')
            if r['architecture_key'] == 'one_shot']


def helper_hashes():
    return {str(p.resolve()): sha(p) for p in
            (Path(__file__), Path(__file__).with_name('test_campaign.py'),
             Path(__file__).with_name('PROTOCOL.txt'))}


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == name), include_attributes=False)


def preflight():
    """Read-only checks against archived task and evaluation definitions."""
    frozen = BASELINE / 'frozen'
    cases = [{'case_label': label, **case.manifest_payload()}
             for label, case in primary.fair_cases()]
    assert json.loads(json.dumps(cases)) == read(frozen/'case_manifest.json')['cases']
    original_cells = read(frozen/'execution_plan.json')['cells']
    assert plan_cells() == [r for r in original_cells if r['architecture_key'] == 'flowpilot']
    archive_manifest = read(frozen/'full_source_manifest.json')
    # Evaluation and public information may not be modified for this follow-up.
    unchanged = ('ablation_test/src/fair_practical_packet.py',
                 'ablation_test/src/holistic_audit.py',
                 'ablation_test/scripts/run_practical_input_pilot.py')
    for relative in unchanged:
        assert sha(ROOT/relative) == archive_manifest[relative], relative
    current = ROOT/'ablation_test/scripts/run_fair_practical_benchmark.py'
    archived = frozen/'full_source/ablation_test/scripts/run_fair_practical_benchmark.py'
    for name in ('call_judge', 'accepted', 'generation_artifact', 'usage_cost',
                 'guarded_create', 'install_spending_guard'):
        assert function_ast(current, name) == function_ast(archived, name), name
    original = read(frozen/'campaign_manifest.json')
    assert primary.JUDGES == original['judges']
    assert primary.GENERATOR == original['generator']
    rows = baseline_rows()
    assert len(rows) == 3 and len(plan_cells()) == 3
    grades = []
    for row in rows:
        packet = read(BASELINE/'packets'/f"{row['candidate_id']}.json")
        label, case = next((label, case) for label, case in primary.fair_cases()
                           if label == row['case'])
        oracle = read(frozen/'held_out_oracles'/f'{primary.campaign.safe_name(label)}.json')
        context = build_case_context(case_label=label, public_input=case.public_payload(),
                                     inventory=case.inventory, oracle=oracle)
        normalized = build_fair_packet(candidate=row['candidate_id'], case_context=context,
                                       result={}, deterministic_verification={},
                                       has_gas=bool(case.expected_features.get('gas_required')),
                                       is_multistage='multistage' in case.category or 'multistep' in case.category)
        assert packet['case_context'] == normalized['case_context']
        for name in primary.JUDGES:
            item = primary.accepted(BASELINE, name, row)
            assert item, f'Missing frozen one-shot grade: {name}, {row["case"]}'
            path, _ = item
            grades.append({'candidate_id': row['candidate_id'], 'judge': name,
                           'response_sha256': sha(path.with_name('parsed_response.json'))})
    return {'status': 'ready_to_freeze_when_repairs_are_complete',
            'one_shot_outputs_reused': 3, 'one_shot_grades_reused': len(grades),
            'new_flowpilot_attempts': 3, 'new_judge_cells': 9,
            'paid_cap_usd': float(LIMIT), 'baseline_grades': grades}


def cost(out):
    result = _PRIMARY_COST(out)
    result['limit_usd'] = float(LIMIT)
    result['scope'] = 'This campaign only; prior campaigns are not billed again.'
    return result


def guarded_create(out, provider, original, kwargs):
    try:
        return _PRIMARY_GUARDED_CREATE(out, provider, original, kwargs)
    except RuntimeError as exc:
        if 'shared $10 guard' in str(exc):
            raise RuntimeError('Paid reservation would exceed this campaign $5 guard') from exc
        raise


def install_spending_guard(out):
    primary.SPEND_LIMIT = LIMIT
    primary.cost = cost
    primary.guarded_create = guarded_create
    primary.install_spending_guard(out)


def initialize(out):
    evidence = preflight()
    assert not out.exists(), 'Use a new directory; an existing campaign is immutable'
    frozen = out/'frozen'
    frozen.mkdir(parents=True)
    sources = primary.source_hashes()
    for relative in sources:
        dest = frozen/'full_source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, dest)
    write(frozen/'full_source_manifest.json', sources)
    write(frozen/'helper_manifest.json', helper_hashes())
    for p in (Path(__file__), Path(__file__).with_name('test_campaign.py'),
              Path(__file__).with_name('PROTOCOL.txt')):
        shutil.copy2(p, frozen/p.name)
    # Development evidence is copied at freeze. The live work log may later
    # receive progress notes, while these frozen copies remain immutable.
    support = frozen/'development_support'
    for name in ('WORK_LOG.txt', 'replay_delivered_artifacts.py'):
        source = Path(__file__).with_name(name)
        if source.exists():
            support.mkdir(exist_ok=True)
            shutil.copy2(source, support/name)
    for name in ('historical_source', 'held_out_oracles'):
        shutil.copytree(BASELINE/'frozen'/name, frozen/name)
    for name in ('outcome_rubric.json', 'case_manifest.json'):
        shutil.copy2(BASELINE/'frozen'/name, frozen/name)
    write(frozen/'execution_plan.json', {'cells': plan_cells()})
    (frozen/'shared_output_contract.txt').write_text(primary.SHARED_OUTPUT+'\n')
    (frozen/'flowpilot_internal_objective.txt').write_text(primary.FLOWPILOT_INTERNAL_OBJECTIVE+'\n')
    reused = baseline_rows()
    write(frozen/'reused_one_shots.json', {'candidates': reused})
    sources_to_pin = [BASELINE/'candidate_key_B.json', BASELINE/'prepared_manifest_B.json']
    (out/'packets').mkdir()
    for row in reused:
        cid = row['candidate_id']
        source_packet = BASELINE/'packets'/f'{cid}.json'
        shutil.copy2(source_packet, out/'packets'/source_packet.name)
        sources_to_pin.append(source_packet)
        sources_to_pin += [Path(row['run_directory'])/name
                           for name in ('result.json', 'raw_response.json')]
        for name in primary.JUDGES:
            source = BASELINE/'judgments'/name/cid
            target = out/'judgments'/name/cid
            shutil.copytree(source, target)
            sources_to_pin += [p for p in source.rglob('*') if p.is_file()]
    write(frozen/'baseline_source_manifest.json',
          {str(p.resolve()): sha(p) for p in sources_to_pin})
    write(frozen/'preflight.json', evidence)
    write(frozen/'campaign_manifest.json', {
        'created_at': now(), 'cohort': COHORT, 'baseline_campaign': str(BASELINE),
        'generator': primary.GENERATOR, 'judges': primary.JUDGES,
        'generation': 'One new FlowPilot attempt for each of the three original practical cases; no new one-shot generation or judging.',
        'source_policy': 'Production repairs frozen at initialization; complete source snapshot and hashes retained. No source edits during the campaign.',
        'unchanged': ['public cases', 'inventory', 'held-out reference boundary',
                      'seeds', 'generator and routing', 'temperature 0.2',
                      'candidate budget 2 and original revision budgets',
                      'retrieval configuration and source exclusions',
                      'packet adapter', 'fourteen criteria and original anchors',
                      'historical judge prompts, validators and transport',
                      'all three one-shot outputs and nine strict accepted grades'],
        'generation_attempts': 3, 'judge_cells_new': 9,
        'max_attempts_per_judge_cell': 3,
        'primary': 'Equal three-judge, applicable-criterion and case means; report no primary FlowPilot mean if any grade is missing.',
        'secondary': {'judges': list(SECONDARY_JUDGES),
                      'rule': 'Predeclared fixed GPT-5.4 and Claude paired mean using all three cases in both arms. Report only when all twelve grades are valid. Never substitute another pair.'},
        'stopping': 'All three fixed generation attempts and nine fixed judge cells, or resource limit. Never retry a valid grade, regenerate for score, drop a case, impute a missing grade, or alter rubric/packet after seeing results.',
        'format_recovery': 'None in this campaign. Invalid responses remain invalid; strict original-validator results only.',
        'price_basis': read(BASELINE/'frozen/campaign_manifest.json')['pricing'],
        'paid_limit_usd': float(LIMIT), 'paid_ledger': str(out/'paid_usage.jsonl'),
        'billing_limit': 'Conservative reservations on returned token usage; unknown charged work after transport failure remains unknown. Local hardware cost not estimated.',
        'interpretation': 'Development evaluation on previously inspected cases, not independent validation, repeatability proof, a guaranteed improvement, or a pure prompt-length experiment.',
    })
    write(out/'source_git_state.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)})
    immutable = [p for p in frozen.rglob('*') if p.is_file()
                 and 'full_source' not in p.relative_to(frozen).parts]
    immutable += [p for directory in (out/'packets', out/'judgments')
                  for p in directory.rglob('*') if p.is_file()]
    write(frozen/'input_manifest.json', {str(p.relative_to(out)): sha(p) for p in immutable})
    verify(out)
    print(json.dumps({'initialized': str(out), **evidence}, indent=2), flush=True)


def verify(out):
    frozen = out/'frozen'
    sources = read(frozen/'full_source_manifest.json')
    assert sources == primary.source_hashes(), 'Live production source changed after freeze'
    for relative, digest in sources.items():
        assert sha(frozen/'full_source'/relative) == digest, relative
    assert helper_hashes() == read(frozen/'helper_manifest.json'), 'Campaign helpers changed after freeze'
    assert plan_cells() == read(frozen/'execution_plan.json')['cells']
    for path, digest in read(frozen/'baseline_source_manifest.json').items():
        assert sha(Path(path)) == digest, 'Original baseline changed: '+path
    for relative, digest in read(frozen/'input_manifest.json').items():
        assert sha(out/relative) == digest, 'Frozen input changed: '+relative
    if (out/'prepared_manifest_B.json').exists():
        for relative, digest in read(out/'prepared_manifest_B.json').items():
            assert sha(out/relative) == digest, 'Prepared artifact changed: '+relative


def generate(out, index):
    verify(out)
    row = plan_cells()[index]
    run = primary.campaign.run_directory(out, row)
    marker = run/'production_repair_attempt_started.json'
    if marker.exists() or (run/'run_summary.json').exists():
        print('Retaining existing generation attempt', index, flush=True)
        return
    case = next(c for _, c in primary.fair_cases() if c.case_id == row['case_id'])
    write(marker, {'at': now(), 'cell': row})
    try:
        status = primary.runner.execute_cell(
            case=case, variant=row['variant'], bundle_name=row['condition_id'],
            bundle=primary.GENERATOR, run_dir=run,
            candidate_budget=row['candidate_budget'], temperature=.2, seed=row['seed'])
    except Exception as exc:
        write(run/'production_repair_attempt_settled.json',
              {'at': now(), 'status': 'generation_raised', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    write(run/'production_repair_attempt_settled.json', {'at': now(), 'status': status.get('status')})
    print('GENERATION', index, row['case_label'], status.get('status'), flush=True)
    verify(out)


def prepare(out):
    verify(out)
    if (out/'prepared_manifest_B.json').exists():
        print('Retaining frozen prepared packets', flush=True)
        return
    rows = [{**r, 'cohort': COHORT, 'reused_from_baseline': True}
            for r in read(out/'frozen/reused_one_shots.json')['candidates']]
    for row in plan_cells():
        run = primary.campaign.run_directory(out, row)
        assert (run/'production_repair_attempt_settled.json').exists(), 'Wait for every fixed attempt to finish'
        label, case = next((label, c) for label, c in primary.fair_cases() if c.case_id == row['case_id'])
        result, present, complete = primary.generation_artifact(run, row['variant'])
        oracle = read(out/'frozen/held_out_oracles'/f'{primary.campaign.safe_name(label)}.json')
        context = build_case_context(case_label=label, public_input=case.public_payload(),
                                     inventory=case.inventory, oracle=oracle)
        cid = candidate_id(COHORT+'/'+str(run.relative_to(out)))
        packet = build_fair_packet(candidate=cid, case_context=context, result=result,
                                   deterministic_verification={},
                                   has_gas=bool(case.expected_features.get('gas_required')),
                                   is_multistage='multistage' in case.category or 'multistep' in case.category)
        assert not primary.historical_module(out).packet_gate(packet)
        baseline = next(r for r in rows if r['case'] == label and r['reused_from_baseline'])
        assert packet['case_context'] == read(out/'packets'/f"{baseline['candidate_id']}.json")['case_context']
        write(out/'packets'/f'{cid}.json', packet)
        rows.append({**row, 'candidate_id': cid, 'case': label, 'stage': 'B', 'cohort': COHORT,
                     'run_directory': str(run), 'generation_status': 'completed' if complete else 'failed_or_interrupted',
                     'delivered_result_present': present, 'postprocessing_complete': complete,
                     'reused_from_baseline': False})
    write(out/'candidate_key_B.json', {'candidates': rows})
    paths = [out/'candidate_key_B.json']+list((out/'packets').glob('*.json'))
    paths += [p for p in (out/'generation').rglob('*') if p.is_file()]
    write(out/'prepared_manifest_B.json', {str(p.relative_to(out)): sha(p) for p in paths})
    verify(out)


def remaining_attempts(out, name, row):
    if primary.accepted(out, name, row):
        return 0
    folder = out/'judgments'/name/row['candidate_id']
    # A killed process may have incurred a paid call without recording status.
    # Do not overwrite its prompt/raw response or restart that ambiguous cell.
    if any(not p.with_name('status.json').exists() for p in folder.rglob('prompt.json')):
        return 0
    count = len(list(folder.rglob('status.json')))
    return max(0, 3-count)


def judge(out, name, index=None):
    verify(out)
    assert name in primary.JUDGES
    assert (out/'prepared_manifest_B.json').exists()
    if name == 'qwen':
        primary.llm._bounded_local_messages = lambda system, messages, max_tokens: primary.base.exact_local_context(
            system, messages, max_tokens=max_tokens, audit_path=out/'local_context.jsonl')
    rows = [r for r in primary.keys(out, 'B') if not r['reused_from_baseline']]
    if index is not None:
        rows = [rows[index]]
    for row in rows:
        for _ in range(remaining_attempts(out, name, row)):
            status = primary.call_judge(out, name, row)
            print('JUDGE', name, row['case'], status['status'], flush=True)
            if status['status'] == 'valid':
                break
            if status.get('error') and 'reservation' in status['error'].lower():
                break
    verify(out)


def summarize(results, judges):
    groups = {}
    for arm in ('One-shot', 'FlowPilot'):
        rows = [r for r in results if r['architecture'] == arm and r['judge'] in judges]
        complete = len(rows) == 3*len(judges)
        groups[arm] = {'complete': complete, 'n_judgments': len(rows),
                       'mean': statistics.fmean(r['score'] for r in rows) if complete else None,
                       'critical_flags': sum(r['critical_flags'] for r in rows),
                       'by_judge': {j: {'n': sum(r['judge'] == j for r in rows),
                                        'mean': statistics.fmean(r['score'] for r in rows if r['judge'] == j)
                                        if sum(r['judge'] == j for r in rows) == 3 else None}
                                    for j in judges}}
    complete = all(r['complete'] for r in groups.values())
    return {'judges': list(judges), 'complete': complete, 'groups': groups,
            'flowpilot_minus_one_shot': groups['FlowPilot']['mean']-groups['One-shot']['mean'] if complete else None}


def report(out):
    verify(out)
    rows = primary.keys(out, 'B')
    assert len(rows) == 6
    results, accepted, missing = [], [], []
    for row in rows:
        for name in primary.JUDGES:
            item = primary.accepted(out, name, row)
            if not item:
                missing.append({'candidate_id': row['candidate_id'], 'case': row['case'], 'judge': name})
                continue
            path, response = item
            scores = [r['score'] for r in response['criterion_scores'] if r['applicability'] == 'APPLICABLE']
            results.append({'candidate_id': row['candidate_id'], 'case': row['case'],
                            'architecture': row['architecture'], 'judge': name,
                            'score': statistics.fmean(scores)/4,
                            'critical_flags': sum(bool(r['critical_error']) for r in response['criterion_scores']),
                            'criteria': response['criterion_scores'],
                            'reused_from_baseline': row['reused_from_baseline']})
            accepted.append({'candidate_id': row['candidate_id'], 'judge': name,
                             'response_path': str(path.with_name('parsed_response.json')),
                             'response_sha256': sha(path.with_name('parsed_response.json'))})
    summary = {'created_at': now(), 'complete': not missing, 'expected_judgments': 18,
               'accepted_judgments': len(results), 'missing': missing,
               'primary': summarize(results, tuple(primary.JUDGES)),
               'predeclared_secondary': summarize(results, SECONDARY_JUDGES),
               'results': results, 'usage': cost(out),
               'reused_one_shot_judgments': sum(r['reused_from_baseline'] for r in results)}
    write(out/'report/summary.json', summary)
    write(out/'accepted_judgments.json', accepted)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True,
                        choices=['preflight', 'init', 'generate', 'prepare', 'judge', 'report'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--item', default='0')
    parser.add_argument('--judge-case', type=int, default=None)
    args = parser.parse_args()
    out = args.output.resolve()
    configure()
    if args.phase in ('generate', 'judge'):
        verify(out)
        install_spending_guard(out)
    if args.phase == 'preflight':
        print(json.dumps(preflight(), indent=2), flush=True)
    elif args.phase == 'init':
        initialize(out)
    elif args.phase == 'generate':
        generate(out, int(args.item))
    elif args.phase == 'prepare':
        prepare(out)
    elif args.phase == 'judge':
        judge(out, args.item, args.judge_case)
    else:
        report(out)


if __name__ == '__main__':
    main()
