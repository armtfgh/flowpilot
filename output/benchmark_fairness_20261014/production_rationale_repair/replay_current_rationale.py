"""Unscored offline replay of the rationale compiler; does not call a model."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from flora_translate.final_rationale import build_final_rationale

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'output/benchmark_fairness_20261014/schema_routing_followup/campaign/generation'
OUT = Path(__file__).parent
records = []
for path in sorted(BASE.glob('*/claude_flowpilot/repeat_01/result.json')):
    raw = path.read_bytes()
    result = json.loads(raw)
    original = deepcopy(result)
    final = result['final_design']
    current, history = build_final_rationale(result['proposal'], final['parameters'], final['streams'], final['stages'])
    source = result['proposal']['decision_rationale']
    restored = {**{key: value['source_record'] for key, value in history.items()},
                **{key: {**value['source_record'], 'text': value['text']}
                   for key, value in current.items() if 'source_record' in value}}
    assert restored == source, 'Every original source record must remain recoverable.'
    assert result == original
    assert path.read_bytes() == raw
    case = path.parts[-4]
    artifact = {'kind': 'offline unscored current-rationale replay',
                'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha256(raw).hexdigest(),
                'decision_rationale': current, 'rationale_history': history}
    (OUT / f'{case}_rationale.json').write_text(json.dumps(artifact, indent=2) + '\n')
    audit = current['final_quantitative_basis']
    records.append({'case': case, 'source_sha256': sha256(raw).hexdigest(), 'source_unchanged': True,
                    'source_record_count': len(source), 'source_records_exactly_preserved': restored == source,
                    'arithmetic_status': audit['status'], 'time_indices': audit['time_indices'],
                    'gas_streams': audit['gas_streams']})
summary = {'kind': 'offline unscored replay; no generation, judging, or setpoint change', 'model_calls': 0, 'records': records}
(OUT / 'replay_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps([{k: v for k, v in row.items() if k not in ('time_indices', 'gas_streams')} for row in records], indent=2))
