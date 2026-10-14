"""Render the manually reviewed records; never create scores or call a model."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
audit = json.loads((base/'fresh_content_audit.json').read_text())
lines = ['Independent content audit — fresh matched-JSON campaign', '', audit['scope'], '',
         'Campaign: '+audit['campaign'],
         f"Status: {audit['status']}; {len(audit['reviewed_outcomes'])}/{audit['expected_outcomes']} outcomes reviewed.", '']
if audit.get('interpretation_caution'):
    lines += ['Interpretation: '+audit['interpretation_caution'], '']
for record in audit['reviewed_outcomes']:
    lines += [record['case']+' / '+record['arm'], 'Source: '+record['result_path'],
              'SHA256: '+record['result_sha256'], 'Boundary: '+record['reviewed_boundary'], '']
    for check in record['positive_checks']:
        lines += ['Verified/supporting check — '+check['description']]
        if check.get('quote'): lines += ['Quote: '+check['quote']]
        lines += [check['check'], '']
    for finding in record['findings']:
        lines += [finding['id']+' — '+finding['classification'],
                  'Fields: '+', '.join(finding['source_paths'])]
        lines += ['Quote: '+quote for quote in finding.get('quotes', [])]
        lines += [finding['check'], '']
    if record.get('serializer_differences'):
        lines += ['Raw-versus-normalized serialization differences (raw answer remains authoritative):',
                  json.dumps(record['serializer_differences'], ensure_ascii=False), '']
    lines += ['-'*72, '']
(base/'fresh_content_audit.txt').write_text('\n'.join(lines)+'\n')
print(f"Rendered {len(audit['reviewed_outcomes'])} outcomes")
