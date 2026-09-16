"""Package the clarified-inventory reruns without changing earlier campaigns."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'outputs/khu_revised_six_20260915'
OUT=CAMPAIGN/'presentation'
INVENTORY=ROOT/'inventory_khu/KHU_inventory_20260915_v6.json'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/package_khu_six_revised.py'),str(CAMPAIGN),
        '--date','15 September 2026','--inventory',str(INVENTORY),'--refresh-titles'],check=True,cwd=ROOT)
    selected=json.loads((OUT/'selected_results.json').read_text())
    assert len(selected)==6 and all(r['passed'] for r in selected), 'Do not package failed designs as successful'
    for stem in ['stage_summary','component_feeds']:
        rows=[]
        for case in selected:
            with (OUT/case['case']/f'{stem}.csv').open() as f:
                rows.extend(csv.DictReader(f))
        write_csv(OUT/f'all_{stem}.csv',rows)
    checks=[];comparison=[]
    old=ROOT/'outputs/khu_revised_six_20260914/presentation'
    for case in selected:
        review=json.loads((OUT/case['case']/'independent_checks.json').read_text())
        checks.extend(dict(case=case['case'],attempt=case['attempt'],check=k,passed=v) for k,v in review['checks'].items())
        previous=json.loads((old/case['case']/'independent_checks.json').read_text())
        for after in review['stage_rows']:
            before=next(s for s in previous['stage_rows'] if s['stage']==after['stage'])
            for key in ['volume_mL','temperature_C','liquid_flow_mL_min','gas_inlet_STP_mL_min','nominal_inlet_residence_min','reactor','module','light','BPR_bar']:
                comparison.append(dict(case=case['case'],stage=after['stage'],parameter=key,previous=before[key],updated=after[key]))
    write_csv(OUT/'all_independent_checks.csv',checks)
    write_csv(OUT/'previous_vs_updated.csv',comparison)
    sources=OUT/'sources';sources.mkdir(exist_ok=True)
    for name in ['Inventory_final.xlsx','Figure 5, 6 batch protocol_results (KHU revised V2)_final.pptx','Vapourtec Peristaltic pump reagent list.pdf']:
        shutil.copy2(ROOT/'inventory_khu'/name,sources/name)
    shutil.copytree(ROOT/'inventory_khu/source_review_20260915',sources/'source_review',dirs_exist_ok=True)
    shutil.copy2(CAMPAIGN/'regression_tests_final.log',OUT/'regression_tests.log')
    code=OUT/'reproduction_scripts';code.mkdir(exist_ok=True)
    for path in ROOT.glob('scripts/*khu*.py'):
        shutil.copy2(path,code/path.name)
    history=[]
    for attempt in sorted(CAMPAIGN.glob('figure*/attempt*')):
        history.append(dict(case=attempt.parent.name,attempt=attempt.name,
            summary=json.loads((attempt/'summary.json').read_text()),
            selected=any(r['case']==attempt.parent.name and r['attempt']==attempt.name for r in selected)))
    (OUT/'attempt_history.json').write_text(json.dumps(history,indent=2))
    lines=['# KHU clarified inventory and six reruns: 15 September 2026','',
        '## Scope and outcome','',
        'Two batch chemistries, each with three exact response sets: six requests, not six independent chemistries. All attempts are retained; selected results must pass the independent checks. These are initial screening designs, not measured yield improvements or authorization for experimental operation.','',
        '| Request | Attempt | Volume S1 / S2 (mL) | Nominal time S1 / S2 (min) | Temperature S1 / S2 (C) |',
        '|---|---|---|---|---|']
    for row in selected:
        value=lambda key:' / '.join(f'{s[key]:.5g}' for s in row['stages'])
        lines.append(f'| {row["case"]} | {row["attempt"]} | {value("volume_mL")} | {value("nominal_inlet_residence_min")} | {value("temperature_C")} |')
    lines += ['', '## Source corrections','',
        '- Checked all nine workbook sheets against the previous archived workbook. The coil sheet adds explicit module compatibility in K3:K13 and moves fabrication remarks to column L. Other sheets are unchanged.',
        '- Every listed coil is declared compatible with UV-150 and manual photoreactors, but physical module capacity still applies: a 20 mL coil is not admitted into the 10 mL UV-150. No additional dedicated reactor or double-counted coil stock is invented.',
        '- UV-150 requires an E-series or R-series platform. Its lights remain attached to that module; manual modules use their own lights. Compatible tubing installed inside UV-150 is not a detached-light arrangement.',
        '- Mixed pump systems are allowed but penalized during selection. Chemical exclusions, pressure, shared channels and setting-grid feasibility are checked first. A mixed selection carries a laboratory-confirmation flag.',
        '- Pump minimum flow and setting increment are separate typed fields. E/R/SF10 increments are 0.01 mL/min. Syringe increments come from each syringe row. STP MFC resolution is unspecified and is not invented.',
        '- Joint scale quantization preserves Q_i = k(equiv_i/C_i). It does not round feeds independently or change specified stock concentrations. MFC-limited rescaling and final allocation also check pump increments.',
        '- Independent review rejected an invented 0.1 M strength for an unnamed pH 9 buffer with a trailing aqueous annotation. Medium classification now recognizes that phase annotation and removes only software-generated medium-strength assumptions; named reactive buffer reagents remain subject to stoichiometric checks. The original result and failed check are retained.',
        '- A setting grid anchored at zero is the explicit numerical interpretation of an increment; no independent offset is supplied in the workbook.',
        '- Oxygen is introduced only into Stage 2 for all Figure 5 sets; the inventory requires the MFC outlet check valve. The generic Pump symbol and the uploaded check-valve PNG are used by the shared renderer.',
        '- The same revised PPT response slides are used: 5, 8, 11 and 16, 19, 22. Prior FlowPilot output slides are not treated as measured experimental evidence.','',
        '## Reproducibility and evaluation','',
        'Models: Claude Opus 4.6 upstream, Claude Sonnet 4.6 downstream/council; scientific design policy, 12 connected-process candidates. Every original run has fresh generation and council calls. Any technical retry or replay is separately marked in its provenance. Prompts, responses, usage, frozen intake, inventory and source hashes remain in each raw attempt.',
        'The final test log is included. Early precision regression failures were retained and corrected; no validation gate was removed. Independent checks cover flow arithmetic, feed concentrations and molar stoichiometry, pump limits/increments, module membership/capacity, shared resources, gas placement, the check valve, and consistency of topology labels with final values.',
        'Runtime and token counts are in each case usage.json. Similar selections between different objectives are retained, not rerun to manufacture a preferred trend. The old-versus-new CSV is a descriptive comparison: stochastic generation and inventory changes both affect the result, so it is not a controlled causal estimate.',
        'The production translation, autosave and shared topology renderer are exercised. This campaign does not claim an interactive browser GUI smoke test. A separately saved GUI profile is named KHU Laboratory Inventory - confirmed modules and pump increments; the prior profiles are retained.','',
        'Presentation topology exports use full wrapped titles rendered from the unchanged final topology. Original images are retained alongside them; topology_export_provenance.json records this display-only refresh. No numerical design or council decision is changed.','',
        '## Timing and laboratory checks','',
        'Gas feed is referenced to 273.15 K and 1.01325 bar in mL/min. Air contains 21 mol% O2. Gas-fed stage time is the requested nominal V/(Q_liquid + Q_gas,STP), not a measured hydrodynamic contact time. Liquid-only stage time is V/Q_liquid.',
        'Before experiments, confirm full mixture compatibility, stock solution solubility, buffer formulation, heating arrangements, pressurized gas supply and pressure reference conventions. The check valve\'s 1 bar cracking differential is NOT its working-pressure rating; its maximum rating remains unconfirmed. MFC quantity is represented as one listed instrument and needs confirmation. Software closure does not resolve these missing physical facts.',
        'Fabricatable tubing, unspecified-wavelength lights and the integrated H-Cube are preserved in the source catalogue but are not silently enumerated as installed independent devices.','',
        '## Files','',
        '- KHU_revised_six_designs.pptx: source responses, topologies, stage tables, component feed tables, rationale and preparation.',
        '- Each case folder: selected complete result/intake/inventory/request, topology PNG/SVG, component and stage CSVs, checks and usage.',
        '- previous_vs_updated.csv: comparison with the archived 14 September results.',
        '- Parent campaign: every raw attempt, full model logs and source snapshots.',
        '- sources: workbook, compatibility PDF, response deck, cell-by-cell extraction and user-supplied clarification text.','',
        '## Reproduction','', '```bash',
        '.venv-flowpilot/bin/python scripts/run_khu_six_revised.py --output outputs/khu_revised_six_NEW --inventory inventory_khu/KHU_inventory_20260915_v6.json --source-review inventory_khu/source_review_20260915/source_extraction.json',
        '```', '', 'This makes paid model calls. Exact source inputs are reproducible; model outputs are not guaranteed identical.']
    (OUT/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(OUT/'RUN_REPORT.md',CAMPAIGN/'RUN_REPORT.md')
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='sha256_manifest.json'}
    (OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Packaged',len(selected),'selected runs;',len(checks),'independent checks')


if __name__=='__main__':
    main()
