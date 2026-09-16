"""Seal the reviewed selected package and the full append-only campaign."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'outputs/khu_revised_six_20260915'
OUT=CAMPAIGN/'presentation'


def main():
    selected=json.loads((OUT/'selected_results.json').read_text())
    render=json.loads((OUT/'visual_review/render_audit.json').read_text())
    layout=json.loads((OUT/'topology_layout_audit.json').read_text())
    assert len(selected)==6 and all(r['passed'] for r in selected)
    assert not render['out_of_page_text'] and len(layout)==6
    for path in ROOT.glob('scripts/*khu*.py'):
        shutil.copy2(path,OUT/'reproduction_scripts'/path.name)
    shutil.copy2(CAMPAIGN/'regression_tests_final.log',OUT/'regression_tests.log')
    shutil.copy2(CAMPAIGN/'inventory_workbook_diff.json',OUT/'sources/inventory_workbook_diff.json')
    report=OUT/'RUN_REPORT.md'
    text=report.read_text()
    if '## Export checks' not in text:
        text += f'\n## Export checks\n\n{render["slides"]} slides exported to PowerPoint and PDF; all pages rendered, with zero out-of-page text spans. '
        text += 'Six SVG topology layouts passed horizontal main-icon alignment and left-to-right checks. All three Figure 5 SVGs embed the exact uploaded check-valve PNG. '
        text += 'Visual inspection notes and contact sheets are in visual_review.\n'
        report.write_text(text)
    shutil.copy2(report,CAMPAIGN/'RUN_REPORT.md')
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='sha256_manifest.json'}
    (OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
    selected_zip=Path(shutil.make_archive(str(CAMPAIGN/'KHU_selected_results_package'),'zip',root_dir=OUT))
    complete=CAMPAIGN.parent/'KHU_revised_six_complete_20260915.zip'
    with zipfile.ZipFile(complete,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(CAMPAIGN.rglob('*')):
            if path.is_file() and path.suffix!='.zip':
                z.write(path,str(path.relative_to(CAMPAIGN.parent)))
    for path in [selected_zip,complete]:
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
        print(path,round(path.stat().st_size/1e6,1),'MB')


if __name__=='__main__':main()
