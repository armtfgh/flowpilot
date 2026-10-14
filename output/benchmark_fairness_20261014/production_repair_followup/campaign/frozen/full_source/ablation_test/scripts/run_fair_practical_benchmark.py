#!/usr/bin/env python3
"""Frozen historical-evaluator bridge and matched-format practical benchmark.

No valid grade is retried and no completed generation is replaced. Run stages
explicitly; stage A requires no fresh generation. Historical packets are copied
byte-for-byte and all source archives remain immutable.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
from types import ModuleType
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ablation_test.src import runner
NORMAL_ONE_SHOT = runner._general_one_shot
# A configure/reload cycle must never capture our own adapter as the native pipeline.
NORMAL_PRODUCTION = getattr(runner._production_full, "_fair_native_production", runner._production_full)
from ablation_test.scripts import run_practical_input_pilot as base
from ablation_test.src.holistic_audit import build_case_context
from ablation_test.src.llm_judge import parse_json_object
from ablation_test.src.newgen_2_outcome import candidate_id, stable_seed, validate_response as audit_response
from ablation_test.src.providers import activate_bundle

campaign = base.campaign
llm = base.llm
read, write, sha, now = base.read, base.write, base.sha, base.now
DEFAULT = ROOT / 'ablation_results/practical_input/claude_fair_bridge_20261014'
HISTORICAL = ROOT / 'ablation_results/manuscript_benchmark/manuscript_three_model_three_repeat_postfix_20260820'
CURRENT = ROOT / 'ablation_results/practical_input/claude_practical_repair_20261014'
PREVIOUS = ROOT / 'ablation_results/practical_input/claude_practical_pilot_20261014'
ARCHIVES = (HISTORICAL, CURRENT, PREVIOUS)
SPEND_LIMIT = Decimal('10')
JUDGES = {
    'qwen': {'provider': 'ollama', 'model': '/models/Qwen3.6-27B', 'base_url': 'http://10.13.24.169:8000/v1', 'upstream_mode': 'always', 'family': 'qwen'},
    'openai': {'provider': 'openai', 'model': 'gpt-5.4-2026-03-05', 'upstream_mode': 'never', 'family': 'openai'},
    'claude': {'provider': 'anthropic', 'model': 'claude-sonnet-4-6', 'upstream_mode': 'never', 'family': 'anthropic'},
}
GENERATOR = {'provider': 'anthropic', 'model': 'claude-sonnet-4-6', 'upstream_mode': 'never', 'family': 'anthropic', 'display': 'Claude Sonnet 4.6'}
ARMS = {'one_shot': ('general_one_shot', 'One-shot'), 'flowpilot': ('full', 'FlowPilot')}
OUTPUT_FIELDS = NORMAL_ONE_SHOT.__globals__['_flow_schema_instruction']().split('proposal_json must be a JSON-encoded string whose decoded object uses these fields:', 1)[1]
SHARED_OUTPUT = ('Required final output: one JSON object with recommended_disposition, disposition_rationale, and proposal. '
                 'The proposal must be a JSON object, not a JSON-encoded string. The proposal uses these fields:\n' + OUTPUT_FIELDS)
# Final serialization belongs to the external delivery boundary, while these
# scientific requirements must also reach intermediate agents unchanged.
_SCIENTIFIC_START = 'Distinguish inlet/STP gas flow from\n'
assert SHARED_OUTPUT.count(_SCIENTIFIC_START) == 1
SERIALIZATION_PREFIX, _scientific_tail = SHARED_OUTPUT.split(_SCIENTIFIC_START, 1)
SCIENTIFIC_OUTPUT_GUIDANCE = _SCIENTIFIC_START + _scientific_tail
FLOWPILOT_INTERNAL_OBJECTIVE = base.OBJECTIVE + '\n\n' + SCIENTIFIC_OUTPUT_GUIDANCE


def _production_with_scoped_output(case, recorder, candidate_budget, temperature, seed, execution_config=None):
    """Keep final-envelope syntax out of native specialist schema instructions."""
    expected = base.OBJECTIVE + '\n\n' + SHARED_OUTPUT
    if case.objective != expected:
        raise ValueError('Only the exact fair-benchmark objective may be routed')
    recorder.write_json('objective_routing.json', {
        'public_objective': case.objective,
        'internal_objective': FLOWPILOT_INTERNAL_OBJECTIVE,
        'removed_serialization': SERIALIZATION_PREFIX,
        'retained_guidance_exact': SCIENTIFIC_OUTPUT_GUIDANCE,
        'scope': 'Intermediate production objective only; public case and final delivery contract unchanged.',
    })
    internal_case = replace(case, objective=FLOWPILOT_INTERNAL_OBJECTIVE)
    return NORMAL_PRODUCTION(internal_case, recorder, candidate_budget, temperature, seed, execution_config)


_production_with_scoped_output._fair_native_production = NORMAL_PRODUCTION


def historical_module(output=None):
    path = ((output / 'frozen/historical_source/newgen_2_outcome.py') if output and (output / 'frozen/historical_source/newgen_2_outcome.py').exists()
            else HISTORICAL / 'frozen/source_code/newgen_2_outcome.py')
    module = ModuleType('fair_historical_' + sha(path)[:12])
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def fair_cases():
    return [(label, replace(case, objective=base.OBJECTIVE + '\n\n' + SHARED_OUTPUT,
                           suite_id='fair_practical_matched_format_v1')) for label, case in base.practical_cases()]


def configure(output=None):
    runner._general_one_shot = NORMAL_ONE_SHOT
    runner._production_full = _production_with_scoped_output
    campaign.MODELS = {'claude': GENERATOR}
    campaign.REPEAT_IDS = ('repeat_01',)
    campaign.JUDGES = tuple(JUDGES)
    campaign.TEMPERATURE = .2
    campaign.CANDIDATE_BUDGET = 2
    campaign.CANDIDATE_BUDGETS = {'one_shot': 1, 'flowpilot': 2}
    campaign.ARCHITECTURES = ARMS
    campaign.EXECUTION_CONFIGS = {}
    campaign.SEED_NAMESPACE = 'fair-practical-bridge-20261014-v1'
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = True
    campaign.selected_cases = fair_cases


def archive_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for directory in ARCHIVES for p in sorted(directory.rglob('*')) if p.is_file()}


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in base.source_files()}


def verify(output):
    frozen_sources = read(output / 'frozen/full_source_manifest.json')
    assert source_hashes() == frozen_sources, 'Sources changed after freeze'
    for relative, digest in frozen_sources.items():
        assert sha(output / 'frozen/full_source' / relative) == digest, 'Frozen source copy changed: ' + relative
    assert archive_hashes() == read(output / 'frozen/archive_manifest.json'), 'A source archive changed'
    for relative, digest in read(output / 'frozen/input_manifest.json').items():
        assert sha(output / relative) == digest, relative
    expected = json.loads(json.dumps([{'case_label': label, **case.manifest_payload()} for label, case in fair_cases()]))
    assert expected == read(output / 'frozen/case_manifest.json')['cases'], 'Case facts changed'
    assert campaign.campaign_cells(fair_cases()) == read(output / 'frozen/execution_plan.json')['cells']
    for path in output.glob('prepared_manifest_*.json'):
        for relative, digest in read(path).items():
            assert sha(output / relative) == digest, relative


def select_saved():
    historical = read(HISTORICAL / 'frozen/candidate_key_confidential.json')['candidates']
    current = read(CURRENT / 'frozen/candidate_key_confidential.json')['candidates']
    rows = [(HISTORICAL, 'historical_manuscript', r) for r in historical
            if r['generator_model'] == 'Claude Sonnet 4.6' and r['repeat_id'] == 'repeat_01']
    rows += [(CURRENT, 'current_practical', r) for r in current if r['cohort'] == 'fresh']
    assert len(rows) == 15 and len({r['candidate_id'] for _, _, r in rows}) == 15
    return rows


def initialize(output):
    if output.exists():
        raise RuntimeError('Use a new immutable campaign directory')
    frozen = output / 'frozen'
    frozen.mkdir(parents=True)
    sources = source_hashes()
    for relative in sources:
        dest = frozen / 'full_source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    write(frozen / 'full_source_manifest.json', sources)
    write(frozen / 'archive_manifest.json', archive_hashes())
    historical_source = HISTORICAL / 'frozen/source_code'
    for path in historical_source.glob('*.py'):
        dest = frozen / 'historical_source' / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
    shutil.copy2(HISTORICAL / 'frozen/outcome_rubric.json', frozen / 'outcome_rubric.json')
    assert sha(frozen / 'outcome_rubric.json') == sha(CURRENT / 'frozen/outcome_rubric.json')
    cases = fair_cases()
    write(frozen / 'case_manifest.json', {'cases': [{'case_label': label, **case.manifest_payload()} for label, case in cases]})
    write(frozen / 'execution_plan.json', {'cells': campaign.campaign_cells(cases)})
    (frozen / 'shared_output_contract.txt').write_text(SHARED_OUTPUT + '\n')
    (frozen / 'flowpilot_internal_objective.txt').write_text(FLOWPILOT_INTERNAL_OBJECTIVE + '\n')
    for label, case in cases:
        folder = frozen / 'practical_inputs' / campaign.safe_name(label)
        write(folder / 'public_input.json', case.public_payload())
        (folder / 'request.txt').write_text(case.design_input_text)
        oracle = campaign.ORACLES[label]
        dest = frozen / 'held_out_oracles' / f'{campaign.safe_name(label)}.json'
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(oracle, dest)
    keys = []
    for source, cohort, row in select_saved():
        entry = copy.deepcopy(row)
        packet = source / 'packets' / f"{row['candidate_id']}.json"
        result = Path(row['run_directory']) / 'result.json'
        dest = output / 'saved_packets' / packet.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(packet, dest)
        assert dest.read_bytes() == packet.read_bytes()
        entry.update(stage='A', cohort=cohort, source_packet=str(packet.relative_to(ROOT)), packet_sha256=sha(packet),
                     source_result=str(result.relative_to(ROOT)), result_sha256=sha(result))
        keys.append(entry)
    random.Random(202610141).shuffle(keys)
    write(frozen / 'stage_A_key.json', {'candidates': keys})
    write(frozen / 'campaign_manifest.json', {
        'created_at': now(), 'name': 'Historical evaluator bridge and matched-format practical experiment',
        'stage_A_outcomes': 15, 'stage_A_judgments': 45, 'stage_B_generation_outcomes': 6, 'stage_B_judgments': 18,
        'total_judgments': 63, 'historical_source': str(HISTORICAL.relative_to(ROOT)),
        'rubric_sha256': sha(frozen / 'outcome_rubric.json'), 'judges': JUDGES, 'generator': GENERATOR,
        'judge_runtime_temperature': 0, 'judge_actual_temperature': {'qwen': 0, 'claude': 0, 'openai': 'omitted to reproduce archived GPT-5.4 transport'}, 'judge_max_tokens': 7500, 'generation_temperature': .2,
        'one_shot_max_tokens': 16384, 'flowpilot_candidate_budget': 2, 'repeat_count': 1,
        'primary': 'Equal three-judge, applicable-criterion and case means. Critical findings separate.',
        'model_policy': 'Judge model identities are based on archived call telemetry, not mutable/default labels.',
        'packet_policy_A': 'Byte-identical saved packets; restored historical prompt/validator and actual historical panel. This isolates instructions/panel with packet content fixed, not a full historical adapter replay.',
        'packet_policy_B': 'Complete raw-JSON one-shot and canonical delivered FlowPilot content under the same adapter. No arithmetic correction. Shared phase-basis-aware delivered-arithmetic worksheet for explicit values in either arm; no hidden calculations or internal verdicts.',
        'shared_output_contract': True, 'one_shot_native_schema_repeated': True,
        'flowpilot_output_routing': 'Public output contract unchanged; only its final JSON envelope and field-list prefix are excluded from intermediate objectives. Exact gas/time/assumptions/disposition guidance is retained.',
        'historical_worksheet_caveat': 'Historical copied packets retain legacy Q_liquid*tau worksheet even for gas/STP time. Report this limitation; do not silently repair the historical record.',
        'retry_policy': 'One generation attempt per cell. At most three judge format/transport attempts; never retry a valid historical-validator grade. Stricter current citation checks are audit only.',
        'stopping_rule': 'Fixed fifteen saved outcomes then six fresh outcomes, sixty-three grades. No source changes, outcome selection, score-based retries or extensions. Stop at resource failure and retain incompleteness.',
        'stage_order': 'Stage A packets are prepared first; fixed Stage B may generate while A judging runs. No score-dependent adaptation; shared spending reservations apply.',
        'pricing': {'openai': {'model': JUDGES['openai']['model'], 'input_per_million': 2.5, 'cached_input_per_million': .25, 'output_per_million': 15,
                              'source': 'https://developers.openai.com/api/docs/models/gpt-5.4'},
                    'anthropic': {'input_per_million': 3, 'output_per_million': 15, 'cache_read_per_million': .3,
                                  'cache_write_5m_per_million': 3.75, 'cache_write_1h_per_million': 6,
                                  'source': base.PRICING_URL}},
        'spending_limit_usd': 10, 'independent_holdout': False, 'code_repairs_after_freeze': False,
    })
    (frozen / 'PREREGISTRATION.txt').write_text(
        'Fixed before judging: fifteen copied saved outcomes followed by six new matched-format outcomes. '
        'Fourteen original criteria and original historical judge prompt/validator; three equally weighted judges. '
        'All outcomes, errors, grades and paid calls retained. No favorable filtering or extension. '
        'One generation per arm/case cannot establish repeatability. A null or reversed effect remains valid. '
        'Copied historical packets retain their known gas-time worksheet limitation; current/fresh packets use their declared adapter.\n')
    write(output / 'source_git_state.json', {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                          'status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)})
    paths = [p for p in frozen.rglob('*') if p.is_file() and 'full_source' not in p.parts and p.name != 'input_manifest.json']
    paths += list((output / 'saved_packets').glob('*.json'))
    write(frozen / 'input_manifest.json', {str(p.relative_to(output)): sha(p) for p in paths})
    prepare(output, 'A')
    print(json.dumps({'initialized': str(output), 'stage_A_packets': 15, 'total_judgments': 63}), flush=True)


def keys(output, stage='all'):
    rows = []
    for letter in ('A', 'B'):
        if stage not in ('all', letter): continue
        path = output / f'candidate_key_{letter}.json'
        if path.exists(): rows += read(path)['candidates']
    return rows


def accepted(output, judge, row, legacy=None):
    legacy = legacy or historical_module(output)
    folder = output / 'judgments' / judge / row['candidate_id']
    rubric = read(output / 'frozen/outcome_rubric.json')
    packet = read(output / 'packets' / f"{row['candidate_id']}.json")
    valid = []
    for p in sorted(folder.rglob('status.json')):
        parsed = p.with_name('parsed_response.json')
        if read(p).get('status') == 'valid' and parsed.exists() and not legacy.validate_response(read(parsed), rubric, packet):
            valid.append((p, read(parsed)))
    assert len(valid) <= 1, 'Multiple accepted grades for one candidate/judge'
    return valid[0] if valid else None


def require_stage_A_complete(output):
    rows = keys(output, 'A')
    assert len(rows) == 15, 'Stage A not prepared'
    assert all(accepted(output, j, row) for row in rows for j in JUDGES), 'Finish all Stage A grading before generation'


def generate(output, index):
    verify(output)
    assert len(keys(output, 'A')) == 15, 'Prepare the frozen Stage A before generation'
    cases = fair_cases()
    row = campaign.campaign_cells(cases)[index]
    run = campaign.run_directory(output, row)
    if (run / 'run_summary.json').exists() or (run / 'fair_attempt_started.json').exists():
        print('Retaining existing generation attempt', index, flush=True)
        return
    case = next(c for _, c in cases if c.case_id == row['case_id'])
    write(run / 'fair_attempt_started.json', {'at': now(), 'cell': row})
    status = runner.execute_cell(case=case, variant=row['variant'], bundle_name=row['condition_id'], bundle=GENERATOR,
                               run_dir=run, candidate_budget=row['candidate_budget'], temperature=.2, seed=row['seed'])
    print('GENERATION', index, row['case_label'], row['architecture'], status.get('status'), flush=True)
    verify(output)


def generation_artifact(run, variant):
    complete = campaign.completed_run(run)
    present = (run / 'result.json').exists()
    result = read(run / 'result.json') if present else {
        'variant': variant, 'proposal': {}, 'generation_status': 'failed',
        'reported_disposition': 'GENERATION_FAILED',
        'disposition_rationale': 'No final design artifact was recorded in the fixed generation attempt.'}
    return result, present, complete


def prepare(output, stage):
    verify(output)
    if (output / f'prepared_manifest_{stage}.json').exists():
        print('Retaining prepared stage', stage, flush=True)
        return
    packet_dir = output / 'packets'
    packet_dir.mkdir(exist_ok=True)
    if stage == 'A':
        rows = read(output / 'frozen/stage_A_key.json')['candidates']
        for row in rows:
            source = output / 'saved_packets' / f"{row['candidate_id']}.json"
            shutil.copy2(source, packet_dir / source.name)
    else:
        from ablation_test.src.fair_practical_packet import build_fair_packet
        rows = []
        for row in campaign.campaign_cells(fair_cases()):
            run = campaign.run_directory(output, row)
            if not (run / 'run_summary.json').exists() and not (run / 'fair_attempt_started.json').exists():
                raise RuntimeError(f'Generation was never attempted: {run}')
            label, case = next((label, case) for label, case in fair_cases() if case.case_id == row['case_id'])
            result, artifact_present, complete = generation_artifact(run, row['variant'])
            context = build_case_context(case_label=label, public_input=case.public_payload(), inventory=case.inventory,
                                        oracle=read(output / 'frozen/held_out_oracles' / f'{campaign.safe_name(label)}.json'))
            cid = candidate_id('fair_bridge/' + str(run.relative_to(output)))
            packet = build_fair_packet(candidate=cid, case_context=context, result=result, deterministic_verification={},
                                       has_gas=bool(case.expected_features.get('gas_required')),
                                       is_multistage='multistage' in case.category or 'multistep' in case.category)
            assert not historical_module(output).packet_gate(packet)
            write(packet_dir / f'{cid}.json', packet)
            rows.append({**row, 'candidate_id': cid, 'case': label, 'stage': 'B', 'cohort': 'fresh_matched_format',
                         'run_directory': str(run), 'generation_status': 'completed' if complete else 'failed_or_interrupted',
                         'delivered_result_present': artifact_present, 'postprocessing_complete': complete,
                         'result_sha256': sha(run / 'result.json') if (run / 'result.json').exists() else None})
        random.Random(202610142).shuffle(rows)
    write(output / f'candidate_key_{stage}.json', {'candidates': rows})
    paths = [output / f'candidate_key_{stage}.json'] + [packet_dir / f"{r['candidate_id']}.json" for r in rows]
    if stage == 'B': paths += [p for p in (output / 'generation').rglob('*') if p.is_file()]
    write(output / f'prepared_manifest_{stage}.json', {str(p.relative_to(output)): sha(p) for p in paths})
    verify(output)


def usage_cost(provider, usage):
    if provider == 'anthropic':
        return Decimal(str(base.usage_cost(provider, usage)))
    inputs = Decimal(usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0)
    outputs = Decimal(usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0)
    details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
    cached = Decimal(details.get('cached_tokens', 0) or 0)
    return ((inputs-cached)*Decimal('2.5') + cached*Decimal('.25') + outputs*Decimal('15')) / Decimal(10**6)


def cost(output):
    rows = base.ledger_rows(output)
    done = {r['request_id'] for r in rows if r['event'] in ('completed', 'failed')}
    paid = sum((Decimal(str(r['usd'])) for r in rows if r['event'] == 'completed'), Decimal(0))
    pending = sum((Decimal(str(r['reserved_usd'])) for r in rows if r['event'] == 'reserved' and r['request_id'] not in done), Decimal(0))
    return {'estimated_paid_usd': float(paid), 'pending_reserved_usd': float(pending),
            'recorded_paid_calls': sum(r['event'] == 'completed' for r in rows),
            'local_hardware_cost': 'not estimated', 'limit_usd': 10,
            'billing_limit': 'Returned token usage only; charged work without returned usage after transport failures is unknown. Failed reservations are released, so this is not a provider billing guarantee.'}


def guarded_create(output, provider, original, kwargs):
    allowed = 'claude-sonnet-4-6' if provider == 'anthropic' else 'gpt-5.4-2026-03-05'
    if kwargs.get('model') != allowed: raise RuntimeError('Unexpected paid model')
    kwargs = copy.deepcopy(kwargs)
    delta = {}
    if provider == 'openai':
        if 'max_tokens' in kwargs:
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            delta['token_limit_parameter'] = 'max_tokens -> max_completion_tokens'
        if 'temperature' in kwargs:
            kwargs.pop('temperature')
            delta['temperature'] = 'omitted to match archived GPT-5.4 transport'
        kwargs.pop('reasoning_effort', None)
        delta['reasoning_effort'] = 'omitted, matching archived transport/default'
    input_bound = len(json.dumps(kwargs, ensure_ascii=False, default=str).encode()) + 4096
    output_bound = max(int(kwargs.get(key, 0) or 0) for key in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'))
    if output_bound <= 0: raise RuntimeError('Missing explicit output-token bound')
    # Include worst supported Anthropic cache-write rate rather than underreserve.
    input_rate = Decimal('6') if provider == 'anthropic' else Decimal('2.5')
    reserve = (Decimal(input_bound)*input_rate + Decimal(output_bound)*15) / Decimal(10**6)
    rid = uuid.uuid4().hex
    with base.ledger_lock(output):
        used = cost(output)
        if Decimal(str(used['estimated_paid_usd'])) + Decimal(str(used['pending_reserved_usd'])) + reserve > SPEND_LIMIT:
            raise RuntimeError('Paid reservation would exceed shared $10 guard')
        campaign.append_jsonl(output / 'paid_usage.jsonl', {'event': 'reserved', 'at': now(), 'request_id': rid,
            'provider': provider, 'model': allowed, 'reserved_usd': float(reserve), 'max_output_tokens': output_bound,
            'request_sha256': hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest(),
            'request_settings': {k: kwargs[k] for k in ('model', 'temperature', 'seed', 'reasoning_effort', 'max_completion_tokens', 'max_tokens') if k in kwargs},
            'transport_delta': delta})
    try:
        response = original(**kwargs)
    except Exception as exc:
        with base.ledger_lock(output):
            campaign.append_jsonl(output / 'paid_usage.jsonl', {'event': 'failed', 'at': now(), 'request_id': rid, 'provider': provider, 'error_type': type(exc).__name__})
        raise
    usage = response.usage.model_dump()
    with base.ledger_lock(output):
        campaign.append_jsonl(output / 'paid_usage.jsonl', {'event': 'completed', 'at': now(), 'request_id': rid,
            'provider': provider, 'requested_model': allowed, 'returned_model': response.model, 'usage': usage,
            'usd': float(usage_cost(provider, usage))})
    return response


def install_spending_guard(output):
    for provider, resource in [('anthropic', llm._get_anthropic_client().messages), ('openai', llm._get_openai_client().chat.completions)]:
        original = resource.create
        resource.create = lambda _provider=provider, _original=original, **kwargs: guarded_create(output, _provider, _original, kwargs)


def call_judge(output, name, row):
    legacy = historical_module(output)
    existing = accepted(output, name, row, legacy)
    if existing: return read(existing[0])
    folder = output / 'judgments' / name / row['candidate_id']
    statuses = list(folder.rglob('status.json'))
    if len(statuses) >= 3: raise RuntimeError('Fixed judge attempt budget exhausted')
    target = folder if not statuses else folder / 'attempts' / f'attempt_{len(statuses)+1}'
    rubric = read(output / 'frozen/outcome_rubric.json')
    packet = read(output / 'packets' / f"{row['candidate_id']}.json")
    system, user = legacy.judge_prompt(rubric, packet)
    schema = base.judging.schema_for_provider(legacy.response_schema(), name)
    seed = stable_seed('newgen-2.0', name, row['candidate_id'], 'base' if not statuses else f'attempt_{len(statuses)+1}')
    # No score-dependent or prior-grade feedback is appended to the historical prompt.
    write(target / 'prompt.json', {'system': system, 'user': user, 'json_schema': schema, 'seed': seed,
                                 'max_tokens': 7500, 'provider': JUDGES[name]['provider'], 'model': JUDGES[name]['model']})
    events = []
    llm.set_llm_observer(events.append)
    llm.set_llm_runtime_overrides(temperature=0., seed=seed, json_mode=True, json_schema=schema, capture_content=False)
    start = time.perf_counter()
    try:
        with activate_bundle(JUDGES[name]):
            result = llm.call_model_text(model=JUDGES[name]['model'], provider=JUDGES[name]['provider'], api_name='fair_historical_judge',
                                        max_tokens=7500, system=system, user_content=user)
        write(target / 'raw_response.json', asdict(result))
        write(target / 'telemetry.json', events)
        try:
            parsed = parse_json_object(result.text)
            errors = legacy.validate_response(parsed, rubric, packet)
            write(target / 'parsed_response.json', parsed)
            try: audits = audit_response(parsed, rubric, packet)
            except Exception as exc: audits = [f'Audit failed: {type(exc).__name__}']
            write(target / 'current_validator_audit.json', {'affects_acceptance': False, 'issues': audits})
        except Exception as exc:
            errors = [f'{type(exc).__name__}: {exc}']
        status = {'status': 'invalid' if errors else 'valid', 'validation_errors': errors}
    except Exception as exc:
        status = {'status': 'request_failed', 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        llm.clear_llm_observer()
        llm.clear_llm_runtime_overrides()
    status.update(duration_seconds=time.perf_counter()-start, provider=JUDGES[name]['provider'], model=JUDGES[name]['model'])
    write(target / 'status.json', status)
    return status


def judge(output, name, stage='A', shard=0, shards=1):
    verify(output)
    for letter in (('A','B') if stage == 'all' else (stage,)):
        assert (output / f'prepared_manifest_{letter}.json').exists(), 'Prepare and freeze stage before judging: ' + letter
    if name == 'qwen':
        llm._bounded_local_messages = lambda system, messages, max_tokens: base.exact_local_context(system, messages, max_tokens=max_tokens, audit_path=output/f'local_context_{stage}_{shard}.jsonl')
    for row in keys(output, stage)[shard::shards]:
        if accepted(output, name, row): continue
        for _ in range(3):
            status = call_judge(output, name, row)
            print('JUDGE', name, row['stage'], row['case'], row['architecture'], status['status'], flush=True)
            if status['status'] == 'valid': break
            if status.get('error') and 'reservation' in status['error'].lower(): break
        if status['status'] != 'valid': raise RuntimeError('Incomplete judge cell: ' + row['candidate_id'])
    verify(output)


def report(output, stage='all'):
    verify(output)
    result_rows, accepted_rows, missing = [], [], []
    if stage in ('B', 'all') and not (output / 'candidate_key_B.json').exists():
        for planned in campaign.campaign_cells(fair_cases()):
            for name in JUDGES:
                missing.append({'stage': 'B', 'case': planned['case_label'], 'architecture': planned['architecture'], 'judge': name, 'reason': 'Stage B packets not prepared'})
    for row in keys(output, stage):
        for name in JUDGES:
            item = accepted(output, name, row)
            if not item:
                missing.append({'candidate_id': row['candidate_id'], 'judge': name})
                continue
            path, response = item
            scores = [r['score'] for r in response['criterion_scores'] if r['applicability'] == 'APPLICABLE']
            result_rows.append({'candidate_id': row['candidate_id'], 'stage': row['stage'], 'cohort': row['cohort'], 'case': row['case'],
                'architecture': row['architecture'], 'judge': name, 'score': statistics.fmean(scores)/4,
                'critical_flags': sum(bool(r['critical_error']) for r in response['criterion_scores']), 'criteria': response['criterion_scores']})
            accepted_rows.append({'candidate_id': row['candidate_id'], 'judge': name, 'path': str(path.relative_to(output)), 'response_sha256': sha(path.with_name('parsed_response.json'))})
    groups = {}
    for cohort in sorted({r['cohort'] for r in result_rows}):
        groups[cohort] = {}
        for arm in sorted({r['architecture'] for r in result_rows if r['cohort'] == cohort}):
            rr = [r for r in result_rows if r['cohort'] == cohort and r['architecture'] == arm]
            by = {j: [r for r in rr if r['judge'] == j] for j in JUDGES}
            complete = all(len(v) == 3 for v in by.values())
            groups[cohort][arm] = {'complete': complete, 'n_judgments': len(rr),
                'mean': statistics.fmean(r['score'] for r in rr) if complete else None,
                'critical_flags': sum(r['critical_flags'] for r in rr),
                'by_judge': {j: {'mean': statistics.fmean(r['score'] for r in v) if len(v)==3 else None, 'n': len(v)} for j,v in by.items()},
                'generator_family_excluded_mean': statistics.fmean(r['score'] for r in rr if r['judge'] != 'claude') if complete else None}
    summary = {'created_at': now(), 'stage': stage, 'accepted_judgments': len(result_rows), 'missing': missing,
               'groups': groups, 'results': result_rows, 'usage': cost(output), 'original_rubric_equal_judges': True,
               'complete': not missing, 'expected_judgments': {'A':45,'B':18,'all':63}[stage]}
    write(output / 'report' / f'summary_{stage}.json', summary)
    write(output / f'accepted_judgments_{stage}.json', accepted_rows)
    print(json.dumps({k: summary[k] for k in ('stage','accepted_judgments','missing','groups','usage')}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--phase', choices=['init','generate','prepare','judge','report'], required=True)
    parser.add_argument('--stage', choices=['A','B','all'], default='A')
    parser.add_argument('--item', default='0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    configure(output)
    if args.phase in ('generate','judge'): install_spending_guard(output)
    if args.phase == 'init': initialize(output)
    elif args.phase == 'generate': generate(output, int(args.item))
    elif args.phase == 'prepare': prepare(output, args.stage)
    elif args.phase == 'judge': judge(output, args.item, args.stage, args.shard, args.shards)
    else: report(output, args.stage)

if __name__ == '__main__': main()
