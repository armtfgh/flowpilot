"""Read-only checks of schema-routing control and raw strict grades. No API calls."""
from pathlib import Path
import hashlib
import importlib.util
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIMARY = ROOT / 'ablation_results/practical_input/claude_fair_bridge_20261014'
FOLLOW = OUT / 'schema_routing_followup/campaign'
JUDGES = ('qwen', 'openai', 'claude')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    checks = {}
    for manifest, relative_root in (
        (PRIMARY / 'frozen/full_source_manifest.json', ROOT),
        (PRIMARY / 'frozen/input_manifest.json', PRIMARY),
        (PRIMARY / 'frozen/archive_manifest.json', ROOT),
        (PRIMARY / 'prepared_manifest_A.json', PRIMARY),
        (PRIMARY / 'prepared_manifest_B.json', PRIMARY),
        (FOLLOW / 'frozen/source_manifest.json', ROOT),
        (FOLLOW / 'frozen/helper_manifest.json', ROOT),
        (FOLLOW / 'frozen/input_manifest.json', ROOT),
        (FOLLOW / 'prepared_manifest_B.json', ROOT),
    ):
        assert manifest.exists(), f'Not ready: {manifest}'
        hashes = read(manifest)
        for filename, digest in hashes.items():
            path = relative_root / filename
            assert sha(path) == digest, str(path)
        checks[str(manifest.relative_to(ROOT))] = len(hashes)

    assert read(FOLLOW / 'frozen/case_manifest.json') == read(PRIMARY / 'frozen/case_manifest.json')
    assert (FOLLOW / 'frozen/outcome_rubric.json').read_bytes() == (PRIMARY / 'frozen/outcome_rubric.json').read_bytes()
    for p in (FOLLOW / 'frozen/historical_source').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            assert p.read_bytes() == (PRIMARY / 'frozen/historical_source' / p.relative_to(FOLLOW / 'frozen/historical_source')).read_bytes()
    for p in (FOLLOW / 'frozen/held_out_oracles').glob('*.json'):
        assert p.read_bytes() == (PRIMARY / 'frozen/held_out_oracles' / p.name).read_bytes()

    pin = read(OUT / 'schema_routing_followup/baseline_grade_manifest.json')
    assert len(pin['cells']) == 9
    for filename, digest in pin['hashes'].items():
        assert sha(Path(filename)) == digest, filename
    baseline_cells = {(r['candidate_id'], r['judge']): r for r in pin['cells']}
    rows = read(FOLLOW / 'candidate_key_B.json')['candidates']
    originals = read(PRIMARY / 'candidate_key_B.json')['candidates']
    assert len(rows) == 6 and len({r['candidate_id'] for r in rows}) == 6
    assert sum(r['reused_from_primary'] for r in rows) == 3
    routing = read(FOLLOW / 'frozen/routing.json')
    assert routing['public_objective'].endswith(routing['removed_serialization'] + routing['retained_guidance_exact'])
    original_prefix = routing['public_objective'][:-(len(routing['removed_serialization']) + len(routing['retained_guidance_exact']))]
    assert routing['internal_objective'] == original_prefix + routing['retained_guidance_exact']

    spec = importlib.util.spec_from_file_location('followup_frozen_validator', FOLLOW / 'frozen/historical_source/newgen_2_outcome.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    rubric = read(FOLLOW / 'frozen/outcome_rubric.json')
    criteria_ids = {r['criterion_id'] for r in rubric['criteria']}
    scores, missing, attempts, hashes = [], [], [], []
    for row in rows:
        cid = row['candidate_id']
        packet_path = FOLLOW / 'packets' / f'{cid}.json'
        packet = read(packet_path)
        original_same_case = [r for r in originals if r['case'] == row['case']]
        assert len(original_same_case) == 2
        for old in original_same_case:
            assert packet['case_context'] == read(PRIMARY / 'packets' / f"{old['candidate_id']}.json")['case_context']
        if row['reused_from_primary']:
            assert packet_path.read_bytes() == (PRIMARY / 'packets' / f'{cid}.json').read_bytes()
            source = PRIMARY
        else:
            source = FOLLOW
            old = next(r for r in original_same_case if r['architecture_key'] == 'flowpilot')
            new_run = Path(row['run_directory'])
            old_run = Path(old['run_directory'])
            assert (new_run / 'input_public.json').read_bytes() == (old_run / 'input_public.json').read_bytes()
            assert (new_run / 'input_inventory.json').read_bytes() == (old_run / 'input_inventory.json').read_bytes()
            assert read(new_run / 'schema_routing.json') == routing
            for key in ('seed', 'candidate_budget', 'model_id', 'execution_config'):
                assert row[key] == old[key], (row['case'], key)
            meta = read(new_run / 'metadata.json')['metadata']
            old_meta = read(old_run / 'metadata.json')['metadata']
            for key in ('provider', 'model', 'candidate_budget', 'temperature', 'seed', 'protocol_sha256', 'design_input_sha256'):
                assert meta[key] == old_meta[key], (row['case'], key)
        for judge in JUDGES:
            valid = []
            for status_path in sorted((source / 'judgments' / judge / cid).rglob('status.json')):
                status = read(status_path)
                attempts.append({'candidate_id': cid, 'judge': judge, 'status': status['status'], 'status_path': str(status_path), 'validation_errors': status.get('validation_errors', [])})
                if status['status'] == 'valid':
                    response = read(status_path.with_name('parsed_response.json'))
                    errors = validator.validate_response(response, rubric, packet)
                    assert not errors, (cid, judge, errors)
                    valid.append((status_path, response))
            assert len(valid) <= 1, (cid, judge, 'Multiple accepted grades')
            if not valid:
                n = sum(r['candidate_id'] == cid and r['judge'] == judge for r in attempts)
                missing.append({'candidate_id': cid, 'judge': judge, 'attempt_count': n, 'exhausted': n >= 3})
                continue
            status_path, response = valid[0]
            if row['reused_from_primary']:
                pinned = baseline_cells[(cid, judge)]
                assert str(status_path) == pinned['source_files']['status.json']
                assert sha(status_path.with_name('parsed_response.json')) == pinned['response_sha256']
            criteria = response['criterion_scores']
            assert len(criteria) == 14 and {r['criterion_id'] for r in criteria} == criteria_ids
            applicable = [r for r in criteria if r['applicability'] == 'APPLICABLE']
            assert all(type(r['score']) is int and 0 <= r['score'] <= 4 for r in applicable)
            expected = 12 + int(packet['case_context']['applicability']['gas_process']) + int(packet['case_context']['applicability']['multistage_process'])
            assert len(applicable) == expected
            scores.append({'candidate_id': cid, 'case': row['case'], 'architecture': row['architecture'], 'judge': judge, 'score': sum(r['score'] for r in applicable)/(4*len(applicable)), 'critical_flags': sum(r['critical_error'] is True for r in criteria), 'reused_from_primary': row['reused_from_primary']})
            hashes.append({'candidate_id': cid, 'judge': judge, 'response_path': str(status_path.with_name('parsed_response.json')), 'sha256': sha(status_path.with_name('parsed_response.json'))})
    groups = {}
    for arm in ('One-shot', 'FlowPilot'):
        items = [r for r in scores if r['architecture'] == arm]
        complete = len(items) == 9
        groups[arm] = {'complete': complete, 'n_judgments': len(items), 'mean': statistics.mean(r['score'] for r in items) if complete else None, 'critical_flags': sum(r['critical_flags'] for r in items), 'by_judge': {j: statistics.mean(r['score'] for r in items if r['judge'] == j) if sum(r['judge'] == j for r in items) == 3 else None for j in JUDGES}, 'by_case': {c: statistics.mean(r['score'] for r in items if r['case'] == c) if sum(r['case'] == c for r in items) == 3 else None for c in sorted({r['case'] for r in rows})}}
    official = FOLLOW / 'report/summary_B.json'
    summary_matches = False
    if official.exists():
        report = read(official)
        assert report['accepted_judgments'] == len(scores)
        for r in scores:
            actual = next(s for s in report['results'] if (s['candidate_id'], s['judge']) == (r['candidate_id'], r['judge']))
            assert abs(actual['score'] - r['score']) < 1e-12 and actual['critical_flags'] == r['critical_flags']
        for arm, group in groups.items():
            actual = report['groups']['schema_routing_control'][arm]
            assert actual['complete'] == group['complete'] and actual['critical_flags'] == group['critical_flags']
            if group['complete']:
                assert abs(actual['mean'] - group['mean']) < 1e-12
        summary_matches = True
    result = {'complete': len(scores) == 18 and not missing, 'strict_accepted_judgments': len(scores), 'expected_judgments': 18, 'missing': missing, 'groups': groups, 'raw_scores': scores, 'raw_grade_hashes': hashes, 'attempts': attempts, 'integrity_files_checked': checks, 'case_context_equal_both_arms': True, 'public_inputs_inventory_budgets_seeds_unchanged': True, 'exact_nine_baseline_grades_unchanged': True, 'only_final_schema_serialization_routed_out_of_internal_objective': True, 'official_summary_matches': summary_matches, 'api_calls': 0, 'source_or_archive_edits': 0, 'interpretation': 'Developmental schema-routing control, with three original one-shot designs and their nine strict grades reused. No N/A recovery or missing-cell imputation in this strict verification.'}
    (OUT / 'schema_routing_followup/independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('complete', 'strict_accepted_judgments', 'missing', 'groups', 'official_summary_matches')}, indent=2))


if __name__ == '__main__':
    verify()
