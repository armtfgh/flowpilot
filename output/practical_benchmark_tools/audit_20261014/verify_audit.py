"""Read-only evidence checks; writes diagnostic outputs beside this script."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PILOT = ROOT / 'ablation_results/practical_input/claude_practical_pilot_20261014'
ASTRA = ROOT / 'ablation_results/manuscript_benchmark/astra_verified_repairs_dev04_20261012'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

identity_re = re.compile(r'flow\s*pilot|flow\s*agent|general[_ -]?one[_ -]?shot|\bqwen\b|\bgpt(?:-[\w.]+)?\b|\bopenai\b|\bclaude\b|\banthropic\b', re.I)
rows = []
for row in read(PILOT / 'frozen/candidate_key_confidential.json')['candidates']:
    packet_path = PILOT / 'packets' / (row['candidate_id'] + '.json')
    outcome = read(packet_path)['delivered_final_outcome']
    check = {'candidate_id': row['candidate_id'], 'case': row['case'], 'architecture': row['architecture'], 'packet_sha256': digest(packet_path)}
    if row['architecture'] == 'One-shot':
        raw = read(Path(row['run_directory']) / 'raw_response.json')['text']
        prose = outcome['chemistry_and_operations']['chemistry_notes']
        check['full_prose_preserved_after_identity_scrub'] = prose == identity_re.sub('[generator withheld]', raw)
        assert check['full_prose_preserved_after_identity_scrub']
        check['prose_characters'] = len(prose)
    else:
        check['status'] = outcome['delivery_status']
        check['blocked_empty_topology_preserved'] = outcome['topology']['generation_status'] == 'blocked' and not outcome['topology']['unit_operations']
        check['no_final_operating_procedure'] = outcome['chemistry_and_operations']['operating_procedure'] is None
        check['fallback_chemistry_present'] = bool(outcome['chemistry_and_operations']['chemistry_identity'])
        check['fallback_pre_reactor_steps_present'] = bool(outcome['chemistry_and_operations']['pre_reactor_steps'])
        check['fallback_post_reactor_steps_present'] = bool(outcome['chemistry_and_operations']['post_reactor_steps'])
        check['fallback_provenance_explicitly_labelled'] = False
        assert check['blocked_empty_topology_preserved'] and check['no_final_operating_procedure']
    rows.append(check)
write('packet_audit.json', {'checks': rows, 'interpretation': 'Full one-shot evidence is present. Blocked outputs retain provisional chemistry/proposal fallback, which needs an explicit provenance label; this cannot supply a final executable topology or procedure.'})

sources = [PILOT / name for name in ['frozen/input_changes.json','frozen/input_token_comparison.json','report/summary.json','accepted_judgments.json','frozen/campaign_manifest.json','evaluation_format_amendment.json']]
sources += list((PILOT / 'packets').glob('*.json'))
for judgment in read(PILOT / 'accepted_judgments.json'):
    sources.append((PILOT / judgment['accepted_status']).with_name('parsed_response.json'))
sources += list((PILOT / 'readable_outputs').glob('*/one_shot.txt'))
sources += [ASTRA / name for name in ['report/COMPLETED_REPAIR_AND_BENCHMARK_REPORT.txt','report/INTERPRETATION_AND_INTEGRITY.json','frozen/campaign_manifest.json']]
for relative in ['ablation_test/src/newgen_2_outcome.py','ablation_test/src/runner.py','ablation_test/scripts/run_practical_input_pilot.py','flora_translate/executable_artifacts.py']:
    frozen = PILOT / 'frozen/full_source' / relative
    if frozen.exists(): sources.append(frozen)
write('evidence_manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))})

comparisons=[]
for relative in ['flora_translate/executable_artifacts.py','ablation_test/src/newgen_2_outcome.py','ablation_test/src/runner.py']:
    earlier=ASTRA/'frozen/full_source'/relative
    current=PILOT/'frozen/full_source'/relative
    comparisons.append({'relative_source':relative,'astra_frozen_sha256':digest(earlier),'practical_frozen_sha256':digest(current),'identical':earlier.read_bytes()==current.read_bytes()})
write('frozen_source_comparison.json', comparisons)
write('audit_artifact_manifest.json', {p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'audit_artifact_manifest.json'})
print(json.dumps({'packet_checks':len(rows),'full_prose_preservation':'passed','blocked_contract_preservation':'passed','frozen_source_comparison':comparisons},indent=2))
