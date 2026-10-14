"""Offline post-benchmark compiler check; no model calls or archive mutation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from flora_translate.executable_artifacts import _stream_components, _stream_preparation_instruction
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.schemas import BatchRecord, ChemistryPlan


def digest(value):
    return hashlib.sha256(value).hexdigest()


source = ROOT / 'ablation_results/practical_input/claude_practical_repair_20261014/generation/hydrogenolysis/claude_repaired_flowpilot/repeat_01/result.json'
raw = source.read_bytes()
result = json.loads(raw)
unchanged = deepcopy(result)
stream = next(item for item in result['proposal']['streams'] if item['stream_label'] == 'A')
original_recipe = next(item['instruction'] for item in result['final_design']['operating_procedure'] if item['step_id'] == 'PREP-A')
components = _stream_components(result, result['proposal']['streams'])
feed = [item for item in components if item.stream_label == 'A']
new_recipe = _stream_preparation_instruction(stream, feed, parameters=result['final_design']['parameters'])
plan = ChemistryPlan.model_validate(result['chemistry_plan'])
batch = BatchRecord.model_validate(result['batch_record'])
new_plan, events = reconcile_chemistry_plan(batch, plan)

assert original_recipe.count('1.26 mmol') == 2
assert new_recipe.count('1.26 mmol') == 1
assert sum(item.concentration_M or 0 for item in feed if item.role.startswith('substrate')) == .126
assert result == unchanged
assert source.read_bytes() == raw
assert plan.model_dump() == ChemistryPlan.model_validate(unchanged['chemistry_plan']).model_dump()

record = {
    'scope': 'Post-benchmark offline compiler repair; UNSCORED. No generation, judging, experimental validation, or frozen result edits.',
    'source_path': str(source.relative_to(ROOT)),
    'source_sha256_before_and_after': digest(raw),
    'raw_proposal_sha256_before_and_after': digest(json.dumps(result['proposal'], sort_keys=True).encode()),
    'original_preparation_instruction': original_recipe,
    'recompiled_preparation_instruction': new_recipe,
    'stock_volume_mL': 10,
    'literal_source_identical_substrate_mmol_before': 2.52,
    'literal_source_identical_substrate_mmol_after': 1.26,
    'intended_substrate_concentration_M': .126,
    'canonical_components': [item.model_dump() for item in feed],
    'reconciled_plan_liquid_feeds': [item.model_dump() for item in new_plan.stream_logic if item.phase == 'liquid'],
    'reconciliation_report': events,
    'archive_and_raw_proposal_unchanged': True,
}
output = Path(__file__).with_name('saved_hydrogenolysis_recipe_proof.json')
output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'proof': str(output.relative_to(ROOT)), 'before_mmol': 2.52, 'after_mmol': 1.26, 'unchanged_source_sha256': digest(raw)}))
