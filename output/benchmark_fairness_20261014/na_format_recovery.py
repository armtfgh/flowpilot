#!/usr/bin/env python3
"""Serialization-only supplement. Never modifies campaign files or calls a model."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / 'ablation_results/practical_input/claude_fair_bridge_20261014'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'format_recovery'
JUDGES = ('qwen', 'openai', 'claude')


def digest(raw): return hashlib.sha256(raw).hexdigest()
def sha(path): return digest(path.read_bytes())
def json_bytes(value): return (json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


class Reader:
    """Track immutable inputs, permitting unrelated campaign calls to continue."""
    def __init__(self): self.hashes = {}
    def read(self, path):
        raw = path.read_bytes()
        key = str(path.resolve())
        if key in self.hashes:
            assert self.hashes[key] == digest(raw), 'Immutable input changed between reads: ' + key
        self.hashes[key] = digest(raw)
        return json.loads(raw)
    def text(self, path):
        raw = path.read_bytes()
        key = str(path.resolve())
        if key in self.hashes:
            assert self.hashes[key] == digest(raw), 'Immutable input changed between reads: ' + key
        self.hashes[key] = digest(raw)
        return raw.decode()
    def verify(self):
        assert all(sha(Path(path)) == value for path, value in self.hashes.items()), 'An immutable input changed during recovery'


def load_legacy(campaign, reader):
    path = campaign / 'frozen/historical_source/newgen_2_outcome.py'
    module = ModuleType('frozen_na_recovery_validator')
    module.__file__ = str(path)
    exec(compile(reader.text(path), str(path), 'exec'), module.__dict__)
    return module


def recover_na_only(response, rubric, packet, validator):
    """Return a derived response and exact score-only diff, or a rejection reason."""
    if not isinstance(response, dict): return None, [], 'response is not an object'
    rows = response.get('criterion_scores')
    if not isinstance(rows, list) or len(rows) != len(rubric['criteria']):
        return None, [], 'incomplete criterion set'
    expected = {r['criterion_id'] for r in rubric['criteria']}
    if any(not isinstance(r, dict) for r in rows): return None, [], 'non-object criterion row'
    ids = [r.get('criterion_id') for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != expected: return None, [], 'criterion set mismatch'
    flags = packet['case_context']['applicability']
    authorized = set()
    if flags['gas_process'] is False: authorized.add('UO-08')
    if flags['multistage_process'] is False: authorized.add('UO-09')
    derived = deepcopy(response)
    changes = []
    for i, row in enumerate(rows):
        cid = row['criterion_id']
        correct = 'NOT_APPLICABLE' if cid in authorized else 'APPLICABLE'
        if row.get('applicability') != correct:
            return None, [], 'applicability differs from frozen packet'
        if correct != 'NOT_APPLICABLE': continue
        if row.get('critical_error') is not False or row.get('severity') != 'NONE':
            return None, [], 'N/A critical/severity fields require a forbidden change'
        if row.get('score') is None: continue
        if type(row.get('score')) is not int or not 0 <= row['score'] <= 4:
            return None, [], 'N/A score is not an integer from zero to four'
        derived['criterion_scores'][i]['score'] = None
        changes.append({'path': f'$.criterion_scores[{i}].score', 'criterion_id': cid, 'before': row['score'], 'after': None})
    if not changes: return None, [], 'no qualifying serialization change'
    errors = validator(derived, rubric, packet)
    if errors: return None, [], 'other historical validation errors remain: ' + '; '.join(errors)
    assert [r for r in response['criterion_scores'] if r['applicability']=='APPLICABLE'] == [r for r in derived['criterion_scores'] if r['applicability']=='APPLICABLE']
    expected_copy = deepcopy(response)
    for i, row in enumerate(expected_copy['criterion_scores']):
        if any(d['path'] == f'$.criterion_scores[{i}].score' for d in changes): row['score'] = None
    assert derived == expected_copy
    return derived, changes, None


def attempt_paths(folder):
    paths = []
    if (folder/'status.json').exists(): paths.append(folder/'status.json')
    attempts = []
    for p in (folder/'attempts').glob('attempt_*/status.json'):
        match = re.fullmatch(r'attempt_(\d+)', p.parent.name)
        if match: attempts.append((int(match.group(1)), p))
    return paths + [p for _,p in sorted(attempts)]


def parse_raw_object(raw):
    if raw.get('finish_reason') in ('length','max_tokens') or raw.get('stop_reason') in ('length','max_tokens'):
        raise ValueError('explicitly truncated response')
    text = (raw.get('text') or '').strip()
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.I).strip()
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence: text = fence.group(1).strip()
    try: obj = json.loads(text)
    except json.JSONDecodeError:
        first,last = text.find('{'), text.rfind('}')
        if first < 0 or last <= first: raise
        obj = json.loads(text[first:last+1])
    if not isinstance(obj, dict): raise ValueError('raw response is not an object')
    return obj


def resolve_cell(campaign, out, row, judge, rubric, packet, legacy, reader):
    folder = campaign / 'judgments' / judge / row['candidate_id']
    attempts = attempt_paths(folder)
    strict = []
    for path in attempts:
        status = reader.read(path)
        parsed_path = path.with_name('parsed_response.json')
        if status.get('status') == 'valid' and parsed_path.exists():
            parsed = reader.read(parsed_path)
            if not legacy.validate_response(parsed, rubric, packet): strict.append((path, parsed))
    if len(strict) > 1: raise RuntimeError('Multiple original valid grades for one cell')
    if strict:
        path, parsed = strict[0]
        return {'kind':'original_valid', 'response':parsed, 'status_path':str(path),
                'response_path':str(path.with_name('parsed_response.json')), 'response_sha256':sha(path.with_name('parsed_response.json'))}, []
    rejections = []
    for path in attempts:
        status = reader.read(path)
        raw_path = path.with_name('raw_response.json')
        parsed_path = path.with_name('parsed_response.json')
        if status.get('status') != 'invalid' or not raw_path.exists():
            rejections.append({'status_path':str(path),'reason':'no completed invalid response with raw artifact'})
            continue
        raw = reader.read(raw_path)
        try: parsed = parse_raw_object(raw)
        except Exception as exc:
            rejections.append({'status_path':str(path),'reason':'raw parse rejected: '+str(exc)})
            continue
        if parsed_path.exists() and reader.read(parsed_path) != parsed:
            rejections.append({'status_path':str(path),'reason':'parsed artifact differs from complete raw response'})
            continue
        derived, changes, reason = recover_na_only(parsed, rubric, packet, legacy.validate_response)
        if reason:
            rejections.append({'status_path':str(path),'reason':reason})
            continue
        # Earliest qualifying attempt wins. Applicable scores never guide selection.
        dest = out/'cells'/judge/row['candidate_id']
        write(dest/'normalized_response.json', derived)
        record = {'schema_version':'na_score_null_recovery_v1','candidate_id':row['candidate_id'],'judge':judge,
            'source_attempt':str(path.parent),'source_status_sha256':sha(path),'raw_response_sha256':sha(raw_path),
            'source_parsed_sha256':sha(parsed_path) if parsed_path.exists() else None,
            'derived_response_sha256':sha(dest/'normalized_response.json'),'packet_sha256':sha(campaign/'packets'/f"{row['candidate_id']}.json"),
            'rule_declaration_sha256':sha(out/'rule_declaration.json'),'selection':'earliest qualifying unaccepted attempt',
            'diff':changes,'applicable_rows_unchanged':True,'historical_validation_errors':[],
            'original_files_unchanged':True,'api_calls':0,'source_attempts_considered_before_selection':rejections}
        write(dest/'acceptance.json', record)
        return {'kind':'recovered_na_serialization','response':derived,'status_path':str(dest/'acceptance.json'),
                'response_path':str(dest/'normalized_response.json'),'response_sha256':sha(dest/'normalized_response.json'),
                'recovery':record}, rejections
    return None, rejections


def live_usage(campaign):
    """Usage is a live read, not part of immutable-input verification while calls run."""
    p = campaign/'paid_usage.jsonl'
    rows=[]
    if p.exists():
        for line in p.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass  # an in-progress final append is not a settled call
    completed=[r for r in rows if r.get('event')=='completed']
    settled={r['request_id'] for r in rows if r.get('event') in ('completed','failed')}
    return {'estimated_paid_usd':float(sum((Decimal(str(r['usd'])) for r in completed),Decimal(0))),
            'recorded_paid_calls':len(completed),
            'pending_reserved_usd':float(sum((Decimal(str(r['reserved_usd'])) for r in rows if r.get('event')=='reserved' and r['request_id'] not in settled),Decimal(0))),
            'live_snapshot':True,'limit_usd':10,'billing_limit':'Unreported charged work following transport errors is unknown.'}


def run(campaign=DEFAULT_CAMPAIGN, out=DEFAULT_OUTPUT):
    campaign,out=campaign.resolve(),out.resolve()
    if out.is_relative_to(campaign): raise ValueError('Recovery output must stay outside campaign archive')
    reader=Reader()
    declaration=reader.read(out/'rule_declaration.json')
    assert declaration['schema_version']=='fair_bridge_na_serialization_supplement_v1'
    legacy=load_legacy(campaign,reader)
    rubric=reader.read(campaign/'frozen/outcome_rubric.json')
    candidates=[]
    for stage in ('A','B'):
        key=campaign/f'candidate_key_{stage}.json'
        if key.exists(): candidates+=reader.read(key)['candidates']
    result_rows=[];accepted=[];strict_missing=[];recovered=[];missing=[];rejections=[]
    for row in candidates:
        p=campaign/'packets'/f"{row['candidate_id']}.json"
        packet=reader.read(p)
        for judge in JUDGES:
            resolution,rejected=resolve_cell(campaign,out,row,judge,rubric,packet,legacy,reader)
            cell={'candidate_id':row['candidate_id'],'judge':judge,'stage':row['stage'],'case':row['case'],'architecture':row['architecture']}
            if rejected: rejections.append({**cell,'attempts':rejected})
            if not resolution or resolution['kind']!='original_valid': strict_missing.append(cell)
            if not resolution:
                missing.append(cell)
                continue
            response=resolution['response']
            if resolution['kind']!='original_valid': recovered.append({**cell,**{k:v for k,v in resolution.items() if k!='response'}})
            scores=[r['score'] for r in response['criterion_scores'] if r['applicability']=='APPLICABLE']
            result_rows.append({**cell,'cohort':row['cohort'],'score':statistics.fmean(scores)/4,
                'critical_flags':sum(bool(r['critical_error']) for r in response['criterion_scores']),
                'criteria':response['criterion_scores'],'acceptance_kind':resolution['kind']})
            accepted.append({**cell,**{k:v for k,v in resolution.items() if k not in ('response','recovery')}})
    if len(candidates)<21:
        missing.append({'stage':'B','reason':'Not all twenty-one fixed candidate packets have been prepared','prepared_candidate_count':len(candidates)})
    groups={}
    for cohort in sorted({r['cohort'] for r in result_rows}):
        groups[cohort]={}
        for arm in sorted({r['architecture'] for r in result_rows if r['cohort']==cohort}):
            rr=[r for r in result_rows if r['cohort']==cohort and r['architecture']==arm]
            by={j:[r for r in rr if r['judge']==j] for j in JUDGES}
            complete=all(len(v)==3 for v in by.values())
            groups[cohort][arm]={'complete':complete,'n_judgments':len(rr),
                'mean':statistics.fmean(r['score'] for r in rr) if complete else None,
                'critical_flags':sum(r['critical_flags'] for r in rr),
                'by_judge':{j:{'mean':statistics.fmean(r['score'] for r in v) if len(v)==3 else None,'n':len(v)} for j,v in by.items()},
                'generator_family_excluded_mean':statistics.fmean(r['score'] for r in rr if r['judge']!='claude') if complete else None}
    reader.verify()
    summary={'created_at':datetime.now(timezone.utc).isoformat(),'stage':'all','kind':'Supplemental serialization-recovery overlay; original strict report unchanged',
        'expected_judgments':63,'accepted_judgments':len(result_rows),'complete':len(result_rows)==63 and not missing,
        'missing':missing,'strict_missing_cells':strict_missing,'strict_valid_judgments':len(result_rows)-len(recovered),
        'recovered_cells':recovered,'groups':groups,'results':result_rows,'usage':live_usage(campaign),
        'original_rubric_equal_judges':True,'rule_declaration_sha256':sha(out/'rule_declaration.json'),
        'api_calls':0,'applicable_scores_unchanged':True,'original_inputs_unchanged':True,
        'source_hashes_before_and_after':reader.hashes,'helper_sha256':sha(Path(__file__))}
    write(out/'summary_all.json',summary)
    write(out/'accepted_judgments_overlay.json',accepted)
    write(out/'unrecoverable_attempts.json',rejections)
    reader.verify()
    write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps({'accepted':len(result_rows),'strict_valid':summary['strict_valid_judgments'],'recovered':len(recovered),'missing':len(missing),'complete':summary['complete']},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    run(args.campaign,args.output)
