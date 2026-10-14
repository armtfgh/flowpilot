#!/usr/bin/env python3
"""Independent, read-only campaign audit. Never imports the campaign driver.

Writes only its own audit report outside the frozen campaign. Scientific output
is hashed and passed to the frozen judge validator; it is never revised. Grade
arithmetic and accounting are recomputed here without calling driver helpers.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = Path(__file__).resolve().parents[1]/'campaign'
SOURCE_DIRS = ('flora_translate', 'benchmark', 'ablation_test/src',
               'ablation_test/scripts', 'ablation_test/tests')
SOURCE_SUFFIXES = {'.py', '.txt', '.json', '.yaml', '.yml'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def function_ast(path, name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == name),
                    include_attributes=False)


def load_frozen_validator(path):
    module = ModuleType('independent_frozen_validator')
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def usage_key(provider, model, usage):
    return (provider, model,
            usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0,
            usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0,
            usage.get('cache_creation_input_tokens', 0) or 0,
            usage.get('cache_read_input_tokens', 0) or 0)


def independent_cost(provider, usage, rates):
    d = lambda value: Decimal(str(value or 0))
    rate = rates[provider]
    if provider == 'anthropic':
        hour = d((usage.get('cache_creation') or {}).get('ephemeral_1h_input_tokens'))
        short = max(Decimal(0), d(usage.get('cache_creation_input_tokens'))-hour)
        amount = (d(usage.get('input_tokens'))*d(rate['input_per_million'])
                  + d(usage.get('output_tokens'))*d(rate['output_per_million'])
                  + d(usage.get('cache_read_input_tokens'))*d(rate['cache_read_per_million'])
                  + short*d(rate['cache_write_5m_per_million'])
                  + hour*d(rate['cache_write_1h_per_million']))
    else:
        detail = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
        cached = d(detail.get('cached_tokens'))
        prompt = d(usage.get('prompt_tokens', usage.get('input_tokens')))
        completion = d(usage.get('completion_tokens', usage.get('output_tokens')))
        amount = ((prompt-cached)*d(rate['input_per_million'])
                  + cached*d(rate['cached_input_per_million'])
                  + completion*d(rate['output_per_million']))
    return amount/Decimal(1000000)


def panel_summary(rows, judges):
    result = {}
    for arm in ('One-shot', 'FlowPilot'):
        selected = [r for r in rows if r['architecture'] == arm and r['judge'] in judges]
        complete = len(selected) == 3*len(judges)
        mean = sum(Decimal(str(r['score'])) for r in selected)/len(selected) if complete else None
        by_case = {}
        for case in sorted({r['case'] for r in rows}):
            subset = [r for r in selected if r['case'] == case]
            by_case[case] = float(sum(Decimal(str(r['score'])) for r in subset)/len(subset)) if len(subset) == len(judges) else None
        result[arm] = {'complete': complete, 'n_judgments': len(selected),
                       'mean': float(mean) if mean is not None else None,
                       'critical_flags': sum(r['critical_flags'] for r in selected),
                       'by_case': by_case}
    complete = all(r['complete'] for r in result.values())
    return {'judges': list(judges), 'complete': complete, 'groups': result,
            'flowpilot_minus_one_shot': result['FlowPilot']['mean']-result['One-shot']['mean'] if complete else None}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, condition, detail=None):
        self.checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    def manifest(self, name, manifest, root=None):
        failed = []
        for name_in_manifest, digest in manifest.items():
            path = root/name_in_manifest if root else Path(name_in_manifest)
            if not path.is_file() or sha(path) != digest:
                failed.append(str(path))
        self.check(name, not failed, {'files': len(manifest), 'mismatches': failed})


def verify(campaign):
    audit = Audit()
    frozen = campaign/'frozen'
    manifest = read(frozen/'campaign_manifest.json')
    baseline = Path(manifest['baseline_campaign'])
    baseline_frozen = baseline/'frozen'
    audit.manifest('All frozen inputs and copied baseline judgments remain unchanged',
                   read(frozen/'input_manifest.json'), campaign)
    audit.manifest('All original pinned baseline source files remain unchanged',
                   read(frozen/'baseline_source_manifest.json'))
    audit.manifest('Campaign helper code and protocol remain unchanged',
                   read(frozen/'helper_manifest.json'))
    sources = read(frozen/'full_source_manifest.json')
    audit.manifest('Frozen source copies match frozen hashes', sources, frozen/'full_source')
    audit.manifest('Live source matches generation freeze', sources, ROOT)
    actual_sources = {str(p.relative_to(ROOT)) for directory in SOURCE_DIRS
                      for p in (ROOT/directory).rglob('*') if p.is_file()
                      and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts}
    audit.check('No source files added or removed after freeze', actual_sources == set(sources),
                {'added': sorted(actual_sources-set(sources)), 'removed': sorted(set(sources)-actual_sources)})
    audit.manifest('Prepared generation artifacts and judge packets remain unchanged',
                   read(campaign/'prepared_manifest_B.json'), campaign)
    baseline_sources = read(baseline_frozen/'full_source_manifest.json')
    for relative in ('ablation_test/src/fair_practical_packet.py',
                     'ablation_test/src/holistic_audit.py',
                     'ablation_test/scripts/run_practical_input_pilot.py'):
        audit.check('Unchanged historical evaluation/input helper '+relative,
                    sha(frozen/'full_source'/relative) == baseline_sources[relative])
    primary_relative = 'ablation_test/scripts/run_fair_practical_benchmark.py'
    for name in ('call_judge', 'accepted', 'generation_artifact', 'usage_cost',
                 'guarded_create', 'install_spending_guard'):
        audit.check('Unchanged historical function '+name,
                    function_ast(frozen/'full_source'/primary_relative, name)
                    == function_ast(baseline_frozen/'full_source'/primary_relative, name))
    for directory in ('historical_source', 'held_out_oracles'):
        originals = {str(p.relative_to(baseline_frozen/directory)): sha(p)
                     for p in (baseline_frozen/directory).rglob('*') if p.is_file()}
        audit.manifest('Copied '+directory+' identical to baseline', originals, frozen/directory)
    for name in ('outcome_rubric.json', 'case_manifest.json'):
        audit.check(name+' identical to baseline', sha(frozen/name) == sha(baseline_frozen/name))
    original_manifest = read(baseline_frozen/'campaign_manifest.json')
    audit.check('Generator and all three judges unchanged', manifest['generator'] == original_manifest['generator']
                and manifest['judges'] == original_manifest['judges'])
    plan = read(frozen/'execution_plan.json')['cells']
    previous_plan = [r for r in read(baseline_frozen/'execution_plan.json')['cells']
                     if r['architecture_key'] == 'flowpilot']
    audit.check('Exact three-case generation plan, seeds and candidate budgets unchanged', plan == previous_plan)
    cases = read(frozen/'case_manifest.json')['cases']
    rows = read(campaign/'candidate_key_B.json')['candidates']
    audit.check('Exactly three baseline and three new outcomes, no case selection',
                len(rows) == 6 and sum(r['reused_from_baseline'] for r in rows) == 3
                and {r['case_id'] for r in rows if not r['reused_from_baseline']} == {r['case_id'] for r in plan})
    telemetry_paid = Counter()
    generation_summary = []
    for row in rows:
        cid = row['candidate_id']
        packet_path = campaign/'packets'/f'{cid}.json'
        if row['reused_from_baseline']:
            audit.check('Byte-identical baseline packet '+cid, sha(packet_path) == sha(baseline/'packets'/packet_path.name))
            continue
        planned = next(r for r in plan if r['case_id'] == row['case_id'])
        baseline_row = next(r for r in rows if r['reused_from_baseline'] and r['case_id'] == row['case_id'])
        run = Path(row['run_directory'])
        marker = read(run/'production_repair_attempt_started.json')
        settled = read(run/'production_repair_attempt_settled.json')
        metadata = read(run/'metadata.json')['metadata']
        old_public = read(Path(baseline_row['run_directory'])/'input_public.json')
        audit.check('Public input parity '+row['case'], read(run/'input_public.json') == old_public)
        audit.check('Inventory parity '+row['case'], read(run/'input_inventory.json') == old_public['inventory'])
        audit.check('Only one fixed generation attempt '+row['case'], marker['cell'] == planned
                    and len(list(run.glob('*attempt_started.json'))) == 1)
        expected_meta = {'seed': planned['seed'], 'candidate_budget': 2, 'temperature': .2,
                         'model': manifest['generator']['model'], 'provider': manifest['generator']['provider'],
                         'variant': 'full', 'design_input_sha256': planned['design_input_sha256']}
        audit.check('Generation metadata unchanged '+row['case'],
                    all(metadata.get(k) == v for k, v in expected_meta.items()))
        original_fp = next(r for r in read(baseline/'candidate_key_B.json')['candidates']
                           if r['case_id'] == row['case_id'] and r['architecture_key'] == 'flowpilot')
        old_meta = read(Path(original_fp['run_directory'])/'metadata.json')['metadata']
        audit.check('Original source exclusions retained '+row['case'],
                    metadata.get('excluded_record_ids') == old_meta.get('excluded_record_ids'))
        routing = read(run/'objective_routing.json')
        audit.check('Final-output routing matches frozen public/internal objectives '+row['case'],
                    routing['public_objective'] == old_public['objective']
                    and routing['internal_objective']+'\n' == (frozen/'flowpilot_internal_objective.txt').read_text())
        packet = read(packet_path)
        baseline_packet = read(campaign/'packets'/f"{baseline_row['candidate_id']}.json")
        audit.check('Judge context identical across arms '+row['case'],
                    packet['case_context'] == baseline_packet['case_context'])
        events = lines(run/'llm_events.jsonl')
        for event in events:
            if event.get('provider') in ('anthropic', 'openai') and event.get('usage'):
                telemetry_paid[usage_key(event['provider'], event['model'], event['usage'])] += 1
        summary = read(run/'run_summary.json') if (run/'run_summary.json').exists() else {}
        audit.check('All recorded generation calls use frozen generator '+row['case'],
                    all(event.get('provider') == manifest['generator']['provider']
                        and event.get('model') == manifest['generator']['model'] for event in events))
        audit.check('Generation event count reconciles '+row['case'],
                    summary.get('llm_call_count') == len(events))
        generation_summary.append({'case': row['case'], 'status': summary.get('status'),
                                   'settled': settled.get('status'), 'llm_calls': len(events),
                                   'input_sha256': sha(run/'input_public.json')})
    rubric = read(frozen/'outcome_rubric.json')
    criterion_ids = {r['criterion_id'] for r in rubric['criteria']}
    validator = load_frozen_validator(frozen/'historical_source/newgen_2_outcome.py')
    grade_rows, missing, attempts, accepted_pin = [], [], [], []
    for row in rows:
        cid = row['candidate_id']
        packet = read(campaign/'packets'/f'{cid}.json')
        expected_system, expected_user = validator.judge_prompt(rubric, packet)
        for judge, definition in manifest['judges'].items():
            folder = campaign/'judgments'/judge/cid
            status_paths = list(folder.rglob('status.json'))
            prompts = list(folder.rglob('prompt.json'))
            audit.check(f'At most three judge attempts {judge}/{cid}', len(status_paths) <= 3 and len(prompts) <= 3)
            valid = []
            for status_path in status_paths:
                status = read(status_path)
                prompt = read(status_path.with_name('prompt.json'))
                if row['reused_from_baseline']:
                    old_path = baseline/'judgments'/judge/cid/status_path.relative_to(folder)
                    audit.check(f'Byte-identical baseline status {judge}/{cid}/{status_path.parent.name}',
                                sha(status_path) == sha(old_path))
                audit.check(f'Historical judge prompt exact {judge}/{cid}/{status_path.parent.name}',
                            prompt['system'] == expected_system and prompt['user'] == expected_user)
                audit.check(f'Judge model and output budget unchanged {judge}/{cid}/{status_path.parent.name}',
                            prompt['model'] == definition['model'] and prompt['provider'] == definition['provider']
                            and prompt['max_tokens'] == 7500)
                parsed_path = status_path.with_name('parsed_response.json')
                if status.get('status') == 'valid':
                    response = read(parsed_path)
                    errors = validator.validate_response(response, rubric, packet)
                    audit.check(f'Accepted grade passes frozen validator {judge}/{cid}', not errors, errors)
                    valid.append((status_path, response))
                events = read(status_path.with_name('telemetry.json')) if status_path.with_name('telemetry.json').exists() else []
                returned_usage_calls = sum(bool(event.get('usage')) for event in events)
                rejected_before_inference = ('Complete judge prompt cannot fit:' in status.get('error', '')
                                             and returned_usage_calls == 0
                                             and not status_path.with_name('raw_response.json').exists())
                if not row['reused_from_baseline']:
                    for event in events:
                        if event.get('provider') in ('anthropic', 'openai') and event.get('usage'):
                            telemetry_paid[usage_key(event['provider'], event['model'], event['usage'])] += 1
                attempts.append({'case': row['case'], 'architecture': row['architecture'],
                                 'judge': judge, 'status': status.get('status'),
                                 'reused': row['reused_from_baseline'], 'path': str(status_path),
                                 'provider_calls_with_returned_usage': returned_usage_calls,
                                 'context_rejected_before_inference': rejected_before_inference,
                                 'error': status.get('error')})
            audit.check(f'No duplicated valid grades {judge}/{cid}', len(valid) <= 1)
            if not valid:
                missing.append({'candidate_id': cid, 'judge': judge, 'case': row['case']})
                continue
            path, response = valid[0]
            audit.check(f'Valid grade never retried {judge}/{cid}',
                        len([p for p in status_paths if p.stat().st_mtime_ns > path.stat().st_mtime_ns]) == 0)
            scores = response['criterion_scores']
            audit.check(f'All fourteen original criteria {judge}/{cid}',
                        len(scores) == 14 and {r['criterion_id'] for r in scores} == criterion_ids)
            applied = [r['score'] for r in scores if r['applicability'] == 'APPLICABLE']
            audit.check(f'Applicable scores are original integer 0-4 {judge}/{cid}',
                        all(type(value) is int and 0 <= value <= 4 for value in applied))
            score = Decimal(sum(applied))/Decimal(4*len(applied))
            flags = sum(bool(r['critical_error']) for r in scores)
            grade_rows.append({'candidate_id': cid, 'case': row['case'], 'architecture': row['architecture'],
                               'judge': judge, 'score': float(score), 'score_numerator': sum(applied),
                               'score_denominator': 4*len(applied), 'critical_flags': flags,
                               'reused': row['reused_from_baseline']})
            accepted_pin.append({'candidate_id': cid, 'judge': judge,
                                 'response_path': str(path.with_name('parsed_response.json')),
                                 'response_sha256': sha(path.with_name('parsed_response.json'))})
            if row['reused_from_baseline']:
                original = baseline/'judgments'/judge/cid/path.relative_to(folder).with_name('parsed_response.json')
                audit.check(f'Byte-identical baseline accepted grade {judge}/{cid}',
                            sha(path.with_name('parsed_response.json')) == sha(original))
    audit.check('All nine baseline grades present unchanged', sum(r['reused'] for r in grade_rows) == 9)
    report = read(campaign/'report/summary.json')
    for row in grade_rows:
        published = next(r for r in report['results']
                         if r['candidate_id'] == row['candidate_id'] and r['judge'] == row['judge'])
        audit.check(f'Independent grade arithmetic {row["judge"]}/{row["candidate_id"]}',
                    abs(published['score']-row['score']) < 1e-12 and published['critical_flags'] == row['critical_flags'])
    audit.check('Report includes all and only accepted judgments', len(report['results']) == len(grade_rows))
    primary = panel_summary(grade_rows, tuple(manifest['judges']))
    secondary = panel_summary(grade_rows, ('openai', 'claude'))
    audit.check('Fixed secondary panel never selected by outcome', manifest['secondary']['judges'] == ['openai', 'claude'])
    for label, independently in [('primary', primary), ('predeclared_secondary', secondary)]:
        published = report[label]
        audit.check('Independent completeness '+label, published['complete'] == independently['complete'])
        for arm in ('One-shot', 'FlowPilot'):
            wanted, actual = independently['groups'][arm], published['groups'][arm]
            same_mean = actual['mean'] is None if wanted['mean'] is None else abs(actual['mean']-wanted['mean']) < 1e-12
            audit.check('Independent panel arithmetic '+label+'/'+arm,
                        same_mean and actual['n_judgments'] == wanted['n_judgments']
                        and actual['critical_flags'] == wanted['critical_flags'])
    ledger = lines(campaign/'paid_usage.jsonl')
    reserved, settled, completed = {}, {}, []
    paid = Decimal(0)
    largest_committed = Decimal(0)
    for entry in ledger:
        rid = entry['request_id']
        if entry['event'] == 'reserved':
            audit.check('Unique spending reservation '+rid, rid not in reserved)
            reserved[rid] = entry
            committed = paid+sum(Decimal(str(r['reserved_usd'])) for key, r in reserved.items() if key not in settled)
            largest_committed = max(largest_committed, committed)
            audit.check('Reservation respected $5 cap '+rid, committed <= Decimal(5)+Decimal('1e-10'))
        elif entry['event'] in ('completed', 'failed'):
            audit.check('Each settled call has exactly one reservation '+rid, rid in reserved and rid not in settled)
            settled[rid] = entry
            if entry['event'] == 'completed':
                actual = independent_cost(entry['provider'], entry['usage'], manifest['price_basis'])
                audit.check('Independent returned-token cost '+rid, abs(actual-Decimal(str(entry['usd']))) < Decimal('1e-10'))
                expected_model = 'claude-sonnet-4-6' if entry['provider'] == 'anthropic' else 'gpt-5.4-2026-03-05'
                audit.check('Requested and returned paid model identity '+rid,
                            entry['requested_model'] == expected_model and entry['returned_model'] == expected_model)
                paid += actual
                completed.append(entry)
    pending = sorted(set(reserved)-set(settled))
    audit.check('No unresolved paid calls at completion', not pending, pending)
    billed_events = Counter(usage_key(e['provider'], e['requested_model'], e['usage']) for e in completed)
    audit.check('Paid ledger reconciles with all new generation/judge telemetry', billed_events == telemetry_paid,
                {'ledger_calls': sum(billed_events.values()), 'telemetry_calls': sum(telemetry_paid.values()),
                 'unmatched_ledger': list((billed_events-telemetry_paid).elements()),
                 'unmatched_telemetry': list((telemetry_paid-billed_events).elements())})
    audit.check('Report cost matches independent accounting',
                abs(float(paid)-report['usage']['estimated_paid_usd']) < 1e-10
                and report['usage']['recorded_paid_calls'] == len(completed))
    context_checks = lines(campaign/'local_context.jsonl')
    audit.check('Local judge evidence never truncated',
                all(r.get('truncated') is False for r in context_checks))
    audit.check('Local judge output allowance and safety margin unchanged',
                all(r.get('output_allowance') == 7500 and r.get('safety_margin') == 512
                    for r in context_checks))
    audit.check('Local context guard arithmetic correct',
                all(r.get('full_prompt_fits') ==
                    (r['input_tokens']+r['output_allowance']+r['safety_margin'] <= r['context_limit'])
                    for r in context_checks))
    # Pin current judgment artifacts without changing the campaign. Future audits
    # can verify these independent hashes against raw responses and statuses.
    judging_hashes = {str(p.relative_to(campaign)): sha(p)
                      for p in (campaign/'judgments').rglob('*') if p.is_file()}
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'campaign': str(campaign),
            'integrity_passed': all(c['passed'] for c in audit.checks),
            'checks': audit.checks, 'failed_checks': [c for c in audit.checks if not c['passed']],
            'primary_complete': primary['complete'], 'accepted_judgments': len(grade_rows),
            'missing': missing, 'primary': primary, 'secondary': secondary,
            'grade_arithmetic': grade_rows, 'generation': generation_summary, 'attempts': attempts,
            'local_context': {'checks': context_checks,
                              'rejected_checks': sum(not r['full_prompt_fits'] for r in context_checks),
                              'new_judge_attempts_rejected_before_inference': sum(
                                  r['context_rejected_before_inference'] and not r['reused'] for r in attempts),
                              'new_qwen_inference_calls_with_returned_usage': sum(
                                  r['provider_calls_with_returned_usage'] for r in attempts
                                  if r['judge'] == 'qwen' and not r['reused'])},
            'cost': {'estimated_paid_usd': float(paid), 'completed_paid_calls': len(completed),
                     'failed_transport_calls': sum(e['event'] == 'failed' for e in settled.values()),
                     'pending': pending, 'largest_reserved_plus_paid_usd': float(largest_committed),
                     'local_hardware_cost': 'not estimated',
                     'caveat': 'Unknown provider charges for failed calls are not represented by returned-token usage.'},
            'accepted_pin': accepted_pin, 'judgment_artifact_hashes': judging_hashes,
            'scope': 'Integrity, input/evaluator parity, arithmetic and accounting. No scientific regrading or new inference.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    report = verify(campaign)
    target = args.output or Path(__file__).with_name('verification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    target = target.resolve()
    assert not target.is_relative_to(campaign), 'Audit output must remain outside the frozen campaign'
    assert not target.exists(), 'Independent audit reports are immutable; use a fresh output path'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('integrity_passed', 'failed_checks', 'primary_complete',
                                                  'accepted_judgments', 'missing', 'primary', 'secondary', 'cost')}, indent=2))
    print('AUDIT_REPORT', target)
    raise SystemExit(0 if report['integrity_passed'] else 1)


if __name__ == '__main__':
    main()
