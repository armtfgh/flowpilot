"""Fresh, append-only Figure 5 runs with confirmed oxygen and council physics."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_khu_six_revised import cases, save

HISTORY = (
    "Qualitative KHU observation reported on 15 September 2026: liquid solution "
    "flow was 0.02 mL/min and the gas actually used was pure oxygen at a reported "
    "0.43 mL/min. Gas reverse flow toward Stage 1 was observed. The earlier design "
    "had specified air, but the experiment used oxygen instead. The reference "
    "conditions of that reported experimental gas reading were not independently "
    "verified. No yield, gas supply pressure, startup sequence, or pressure traces "
    "were supplied. This is qualitative operability evidence, not a kinetic data "
    "point or a calibrated backflow model. Reducing gas flow alone does not prove "
    "that reverse flow is prevented. Stage 1 must remain oxygen-free."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    from openpyxl import load_workbook
    from flora_translate.inventory_profiles import load_inventory_profile, InventoryProfile
    source = ROOT / 'inventory_khu/source_review_20260915/source_extraction.json'
    inventory = ROOT / 'inventory_khu/KHU_inventory_20260915_v6.json'
    workbook = ROOT / 'inventory_khu/Inventory_final.xlsx'
    profile = load_inventory_profile(inventory).model_dump()
    book = load_workbook(workbook, data_only=True, read_only=True)
    increment = float(book['MFC']['D3'].value)
    book.close()
    assert increment == .01, 'Review changed MFC worksheet before running.'
    device = next(d for d in profile['lab_inventory']['gas_hardware']
                  if d['equipment_id'] == 'mfc_ffc00')
    device['flow_rate_increment_sccm'] = increment
    device['notes'] += ' Inlet/STP setting increment 0.01 mL/min, Inventory_final.xlsx MFC!D3.'
    profile.update(version=7, profile_id='khu_laboratory_inventory_20260918_mfc_grid',
                   updated_at_utc=datetime.now(timezone.utc).isoformat())
    profile['extraction_metadata']['campaign_amendment'] = {
        'source_profile': str(inventory.relative_to(ROOT)), 'source_profile_sha256': sha(inventory),
        'workbook_sha256': sha(workbook), 'source_cell': 'MFC!D3',
        'only_hardware_change': 'Typed MFC flow setting increment; no additional hardware assumed.'}
    profile = InventoryProfile.model_validate(profile)
    profile_path = output / 'inventory_profile_v7.json'
    if profile_path.exists():
        profile = load_inventory_profile(profile_path)
    else:
        save(profile_path, profile.model_dump())
    physics = json.loads((ROOT / 'docs/examples/transient_assumptions_illustrative.json').read_text())
    runtime = dict(design_policy='scientific_v2', candidate_budget=12,
                   upstream_model='claude-opus-4-6', downstream_model='claude-sonnet-4-6',
                   council_backflow_review=True, council_physics_profile=physics)
    manifest = dict(runtime=runtime, source_extraction_sha256=sha(source),
                    inventory_profile_sha256=sha(profile_path),
                    protocol_source='KHU revised final PPT; original Figure 5 response sets',
                    amendments=['Pure O2 required at Stage 2 by user', HISTORY,
                                'MFC increment read from the workbook and enforced before selection'],
                    interpretation='Numerical/inventory closure is not safety clearance. Physics assumptions are not lab settings.',
                    rerun_policy='Retain all attempts; no cherry-picking by predicted yield.')
    if not (output / 'manifest.json').exists():
        save(output / 'manifest.json', manifest)
    return cases(source)[:3], profile, runtime


def package_for(case, profile):
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.inventory_resolution import bind_inventory
    answers = [dict(question_id=k, answer=v, source=f'KHU final PPT slide {case["source_slide"]}')
               for k, v in case['answers'].items() if k != 'Q-GAS-001']
    answers += [
        dict(question_id='Q-GAS-001', answer='Pure oxygen (O2), 100 mol% O2. Require pure oxygen at Stage 2; do not use air.', source='User-confirmed rerun requirement'),
        dict(question_id='Q-HIST-001', answer=HISTORY, source='KHU email, 2026-09-15'),
        dict(question_id='Q-PREF-001', answer='Report each feed component stock concentration and molar flow, assigned equipment, stage-specific temperature, volume, liquid flow, oxygen flow in mL/min at inlet/STP, and V/(liquid + inlet/STP gas) apparent residence time. Include conditional backflow findings and missing measurements. Do not predict measured yield or claim safety approval.', source='User reporting requirements')]
    package = bind_inventory(IntakeAgent().analyze(case['protocol'], answers=answers, use_llm=False), profile)
    assert package.ready_for_design, package.missing_question_ids
    gas = package.engineering_requirements['gas']
    assert gas.get('identity_source') == 'chemist_answer', gas
    assert gas.get('species') == 'O2' and gas.get('reagent_mole_fraction') == 1.0, gas
    assert gas.get('introduction_stage') == 2 and gas.get('minimum_equiv_inlet_stp') == 2.0, gas
    return package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/figure5_pure_oxygen_physics_20260918')
    parser.add_argument('--case', choices=['figure5_set1', 'figure5_set2', 'figure5_set3'])
    parser.add_argument('--attempt', default='attempt_01')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected, profile, runtime = prepare(args.output)
    from flora_translate.intake_agent import intake_context_block
    from flora_translate.engine.llm_agents import set_llm_observer, set_llm_runtime_overrides
    from flora_translate.main import translate
    from flora_translate.gui_autosave import autosave_gui_result
    logging.basicConfig(level=logging.INFO)
    set_llm_runtime_overrides(capture_content=True)
    for case in selected:
        if args.case and case['id'] != args.case:
            continue
        package = package_for(case, profile)
        if args.prepare_only:
            print(case['id'], 'READY', package.engineering_requirements.get('gas'), flush=True)
            continue
        folder = args.output / case['id'] / args.attempt
        folder.mkdir(parents=True, exist_ok=False)
        save(folder / 'provided_input.json', case)
        save(folder / 'inventory_profile.json', profile.model_dump())
        save(folder / 'intake_package.json', package.model_dump())
        save(folder / 'request.json', dict(intake_package=package.model_dump(), runtime_options=runtime))
        (folder / 'prompt.txt').write_text(intake_context_block(package), encoding='utf-8')
        hashes = {}
        paths = [*ROOT.glob('flora_translate/*.py'),
                 *ROOT.glob('flora_translate/engine/council_v4/*.py'), Path(__file__),
                 ROOT / 'scripts/run_khu_six_revised.py']
        for path in paths:
            target = folder / 'source_snapshot' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(path.relative_to(ROOT))] = sha(path)
        save(folder / 'source_hashes.json', hashes)
        handler = logging.FileHandler(folder / 'run.log')
        logging.getLogger().addHandler(handler)
        def observe(event):
            with (folder / 'llm_calls.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, default=str, ensure_ascii=False) + '\n')
        set_llm_observer(observe)
        started = time.monotonic()
        summary = dict(case=case['id'], status='running')
        save(folder / 'summary.json', summary)
        before = {x for base in ['scientific_pipeline', 'scientific_council']
                  for x in (ROOT / 'outputs' / base).glob('*')}
        print('START', case['id'], args.attempt, flush=True)
        try:
            with (folder / 'console.log').open('w') as console, redirect_stdout(console), redirect_stderr(console):
                result = translate(case['protocol'], intake_package=package.model_dump(), runtime_options=runtime)
                save(folder / 'result.json', result)
                archive = Path(autosave_gui_result(result, intake_package=package.model_dump(),
                               source='figure5_pure_oxygen_physics', user_input=case['protocol']))
                shutil.copytree(archive, folder / 'gui_export')
                summary.update(status=result.get('final_design', {}).get('status', 'unknown'), archive=str(archive))
        except Exception as exc:
            summary.update(status='failed', error=str(exc))
            (folder / 'failure.txt').write_text(traceback.format_exc())
        finally:
            after = {x for base in ['scientific_pipeline', 'scientific_council']
                     for x in (ROOT / 'outputs' / base).glob('*')}
            for path in sorted(after - before):
                shutil.copytree(path, folder / 'model_artifacts' / path.parent.name / path.name)
            summary['elapsed_seconds'] = time.monotonic() - started
            save(folder / 'summary.json', summary)
            set_llm_observer(None)
            logging.getLogger().removeHandler(handler)
            handler.close()
            print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
