"""Record final review and archive selected exports without removing raw attempts."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'outputs/khu_revised_six_20260914'
OUT=CAMPAIGN/'presentation'


def main():
    selected=json.loads((OUT/'selected_results.json').read_text())
    render=json.loads((OUT/'visual_review/render_audit.json').read_text())
    assert len(selected)==6 and all(row['passed'] for row in selected)
    assert render['slides']==40 and not render['out_of_page_text']
    history=[]
    for attempt in sorted(CAMPAIGN.glob('figure*_set*/attempt*')):
        summary=json.loads((attempt/'summary.json').read_text())
        checks=json.loads((attempt/'independent_checks.json').read_text()) if (attempt/'independent_checks.json').exists() else {}
        history.append(dict(case=attempt.parent.name,attempt=attempt.name,summary=summary,
            independent_pass=checks.get('all_passed'),failed_checks=checks.get('failed'),
            selected=any(x['case']==attempt.parent.name and x['attempt']==attempt.name for x in selected),
            reused_generation=(attempt/'reused_generation_provenance.json').exists(),
            reused_council=(attempt/'reused_council_provenance.json').exists(),
            display_only=(attempt/'display_revision_provenance.json').exists()))
    (OUT/'attempt_history.json').write_text(json.dumps(history,indent=2))
    (OUT/'visual_review/manual_review.md').write_text(
        '# Visual review\n\nAll 40 slides were reviewed in contact sheets; dense review/input/table slides and topology exports were inspected at larger size. '
        'No out-of-page text was detected. Pump icons are generic; actual equipment names remain below them. '
        'Stage conditions and feed concentrations are present, and the six selected topology files match the checked proposal tables. '
        'Original source files and prior attempts were not altered. This is an exported-slide/diagram review, not an interactive GUI browser test.\n')
    for path in ROOT.glob('scripts/*khu*.py'):
        shutil.copy2(path,OUT/'reproduction_scripts'/path.name)
    # Report is generated separately; append the final rendering record once.
    report=OUT/'RUN_REPORT.md'
    text=report.read_text()
    if '## Slide review' not in text:
        report.write_text(text+'\n## Slide review\n\n40-slide PowerPoint and PDF exported. All pages rendered for inspection; zero out-of-page text spans. '
            'The visual_review directory contains page renders, contact sheets and the check record.\n')
    shutil.copy2(report,CAMPAIGN/'RUN_REPORT.md')
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='sha256_manifest.json'}
    (OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=shutil.make_archive(str(CAMPAIGN/'KHU_selected_results_package'),'zip',root_dir=OUT)
    complete=CAMPAIGN.parent/'KHU_revised_six_complete_20260914.zip'
    with zipfile.ZipFile(complete,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(CAMPAIGN.rglob('*')):
            if path.is_file() and path.suffix!='.zip':
                z.write(path,str(path.relative_to(CAMPAIGN.parent)))
    with zipfile.ZipFile(complete) as z:assert z.testzip() is None
    print(json.dumps({'archive':archive,'size_MB':Path(archive).stat().st_size/1e6,
        'complete_archive':str(complete),'complete_size_MB':complete.stat().st_size/1e6,
        'selected':len(selected),'attempts_retained':len(history)}))


if __name__=='__main__':main()
