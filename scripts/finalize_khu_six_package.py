"""Archive selected results and provenance; fail rather than package failed designs."""
import hashlib
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'outputs/khu_revised_six_20260914'


def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/package_khu_six_revised.py'),str(CAMPAIGN)],check=True,cwd=ROOT)
    out=CAMPAIGN/'presentation'
    selected=json.loads((out/'selected_results.json').read_text())
    assert len(selected)==6 and all(x['passed'] for x in selected), 'Not all six results passed'
    for stem in ['stage_summary','component_feeds']:
        rows=[]
        for case in selected:
            with (out/case['case']/f'{stem}.csv').open(newline='') as handle:
                rows.extend(csv.DictReader(handle))
        with (out/f'all_{stem}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checks=[]
    for case in selected:
        review=json.loads((out/case['case']/'independent_checks.json').read_text())
        checks.extend(dict(case=case['case'],attempt=case['attempt'],check=k,passed=v) for k,v in review['checks'].items())
    with (out/'all_independent_checks.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case','attempt','check','passed']);writer.writeheader();writer.writerows(checks)
    sources=out/'sources';sources.mkdir(exist_ok=True)
    for name in ['Inventory_final.xlsx','Figure 5, 6 batch protocol_results (KHU revised V2)_final.pptx','Vapourtec Peristaltic pump reagent list.pdf']:
        shutil.copy2(ROOT/'inventory_khu'/name,sources/name)
    shutil.copytree(ROOT/'inventory_khu/source_review_20260914',sources/'source_review',dirs_exist_ok=True)
    shutil.copy2('/tmp/khu_final_tests_v2.log',out/'regression_tests.log')
    code=out/'reproduction_scripts';code.mkdir(exist_ok=True)
    for path in [*ROOT.glob('scripts/*khu*.py')]:shutil.copy2(path,code/path.name)
    lines=['# KHU revised six-design package', '',
        '## Outcome', '',
        'Six response sets have selected results that pass the independent software checks. '
        'These are screening proposals, not measured yields or authorization to operate equipment. '
        'The original KHU files and all previous attempts remain unchanged.', '',
        '| Case | Selected attempt | Volume, stages 1 / 2 (mL) | Nominal time, stages 1 / 2 (min) | Temperature, stages 1 / 2 (C) | Checks |',
        '|---|---|---|---|---|---|']
    for row in selected:
        stages=row['stages'];review=json.loads((out/row['case']/'independent_checks.json').read_text())
        values=lambda key:' / '.join(f'{s[key]:.5g}' for s in stages)
        lines.append(f'| {row["case"]} | {row["attempt"]} | {values("volume_mL")} | {values("nominal_inlet_residence_min")} | {values("temperature_C")} | {sum(review["checks"].values())}/{len(review["checks"])} |')
    lines.extend(['', '## Contents', '',
        '- `KHU_revised_six_designs.pptx`: editable slides, exact revised responses, topology, stage conditions, component concentrations/molar flows, preparation and limitations.',
        '- Each case folder: selected result, frozen intake, request, inventory, topology PNG/SVG, stage/component CSVs, review and usage information.',
        '- `all_stage_summary.csv`, `all_component_feeds.csv`, `all_independent_checks.csv`: combined comparison tables for all six selected results.',
        '- `sources/`: original XLSX/PPT/PDF, extracted cell/slide provenance and implementation report.',
        '- `reproduction_scripts/`: campaign-specific scripts; these run within the FlowPilot repository, not as a standalone installation.',
        '- Full raw attempts, model messages, council artifacts and logs: parent campaign directory.', '',
        '## Source and model configuration', '',
        'Batch protocols are from slides 1 and 12. Revised response sets are from slides 5, 8, 11 and 16, 19, 22. '
        'Previous slides containing FlowPilot designs are not treated as wet-lab evidence. '
        'The profile uses all nine workbook sheets and the attached compatibility guide; the no-inline-degasser restriction is separately attributed to the prior KHU instruction.', '',
        'Upstream: Claude Opus 4.6. Downstream/council: Claude Sonnet 4.6. '
        'Each run realizes 12 connected-process candidates under the scientific design policy. '
        'OpenAI embeddings in retrieval are not an additional design agent.', '',
        '## Corrections and audit trail', '',
        'Generic Pump icons replace the syringe-specific icon, while labels retain the assigned equipment name. '
        'Shared pump channels and photoreactor modules, compatible lights, stream concentrations and inventory-required gas check valves are enforced in shared backend code. '
        'See the source-review implementation report for detailed changes and inventory limitations.', '',
        'Independent review found two parser/validator faults in Figure 5: an invented buffer molarity, and air described as supplying O2 being mistaken for pure O2. '
        'Both were corrected with regression tests. No failing gate was disabled. '
        'The final two corrected reviews reuse saved upstream chemistry, retrieval and initial proposals for identical chemist inputs and normalized physical inventory; '
        'their 12-candidate realization, final checks and rendering are fresh. '
        'Figure 5 set 1 repeats all six council calls. Figure 5 set 2 additionally reuses four completed specialist reviews only after exact system/request equality checks; '
        'its skeptic and chief are fresh calls. Their usage files explicitly count only fresh calls. '
        'The original generation and specialist logs remain in the referenced attempts.', '',
        'Figure 6 display-verified attempts preserve their original numerical designs and original council decisions. '
        'They rebuild the final contract and topology with corrected component concentration labels, without additional model calls. '
        'The serialization-only failed retry and provider-truncated skeptic response are retained as well. '
        'A bounded JSON retry with a larger output budget now records both attempts without fabricating missing judgments or bypassing semantic validation. '
        'Results are not selected according to predicted yield; unsuccessful attempts remain available.', '',
        '## Verification and interpretation', '',
        '127 focused regression tests passed. Independent checks recompute cumulative flow, nominal stage time, component molar flow and stoichiometry; '
        'check stock geometry, operating temperature, shared resources, module compatibility, gas destination and the check-valve assignment; '
        'and compare diagram concentrations/times to the final proposal. These checks supplement, rather than replace, chemical and equipment review.', '',
        'Gas flow is reported at 273.15 K and 1.01325 bar. For air, the O2 feed is 21% of the total gas molar feed. '
        'Gas-fed stage time is explicitly nominal V/(Q_liquid + Q_gas,STP), not a measured physical gas-liquid residence time. '
        'Liquid-only stage time is V/Q_liquid. Neither quantity demonstrates conversion.', '',
        'Different objectives may select the same conditions. Where a design has shorter nominal processing time, this does not by itself demonstrate higher throughput or yield. '
        'No measured kinetic data or flow yields were provided for these revised runs.', '',
        '**Objective-alignment limitation:** Figure 5 set 3 selects a larger Stage 2 reactor and longer nominal time at the same substrate feed rate as sets 1/2; '
        'this is not a demonstrated throughput improvement. Figure 6 set 3 reduces nominal total time and reactor volume but also has a lower substrate molar feed rate than sets 1/2. '
        'These trade-offs remain visible rather than rerunning until a preferred trend appears. '
        'Council statements such as "ensures adequate conversion" or "chemically compatible" are model assessments, not experimental findings or complete mixture-compatibility certifications.', '',
        '## Laboratory confirmation before experiments', '',
        '- Confirm the CV-3301 maximum working pressure and chemical compatibility. Its 1 bar cracking differential is not a maximum pressure rating.',
        '- Confirm MFC quantity, gas calibration and pressure reference conventions; one listed MFC is assumed.',
        '- Confirm the proposed pressurized, MFC-metered air delivery. The council identifies this change from ambient batch exposure as requiring chemist review.',
        '- Confirm buffer identity/strength, complete feed compatibility with pump tubing, stock solubility and mixing stability.',
        '- Confirm the heating arrangement and pressure reference/rating for the complete connected system.',
        '- Fabricatable tubing, unspecified-wavelength sources and the integrated H-Cube are preserved in the source catalogue but not silently treated as installed selectable modules.', '',
        'The API/production translation and renderer were exercised. This campaign does not claim an interactive browser GUI smoke test.', '',
        '## Reproduction', '',
        'Run from the project root with the configured model credentials and environment:', '',
        '```bash',
        '.venv-flowpilot/bin/python scripts/run_khu_six_revised.py --output outputs/khu_revised_six_NEW --attempt attempt_01',
        '.venv-flowpilot/bin/python scripts/package_khu_six_revised.py outputs/khu_revised_six_NEW',
        '```', '',
        'The first command makes new API calls and incurs provider charges. Exact answers and inventory snapshots are archived per attempt; LLM output is not guaranteed bitwise reproducible.'])
    (out/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(out/'RUN_REPORT.md',CAMPAIGN/'RUN_REPORT.md')
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='sha256_manifest.json'}
    (out/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FINAL_PACKAGE_CHECKED',len(selected))


if __name__=='__main__':main()
