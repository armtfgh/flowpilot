"""Rebuild Word copies, converge the cached contents, and run final verification."""
from pathlib import Path
import subprocess
import sys
import json
import fitz

from refresh_revision_toc_20260922 import refresh, normalized
from verify_submission_20260929 import update_toc

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript/Submission';OUT=BASE/'revision_20260929'
PY=sys.executable

def command(args):
    subprocess.run(args,cwd=ROOT,check=True)

def render(paths):
    command(['libreoffice','-env:UserInstallation=file:///tmp/flowpilot_submission_final_20260929','--headless','--convert-to','pdf','--outdir',str(OUT/'rendered'),*[str(p) for p in paths]])

def main():
    command([PY,'scripts/revise_submission_20260929.py'])
    clean=[BASE/f'{s}_submission_revised_20260929.docx' for s in ('manuscript','esi')]
    render(clean)
    for cycle in range(3):
        update_toc()
        cached=json.loads((OUT/'contents_page_audit.json').read_text())['clean']
        render([clean[1]])
        d=fitz.open(OUT/'rendered/esi_submission_revised_20260929.pdf')
        mismatches=[r for r in cached if normalized(r['heading']) not in normalized(d[r['page']-1].get_text())]
        if not mismatches:
            print(f'Contents verified against rendered page numbers after {cycle+1} pass(es).',flush=True)
            break
    else:raise AssertionError(('Unstable contents',mismatches))
    # Marking does not change content; render the marked copies to test that too.
    render([BASE/f'{s}_submission_revised_20260929_marked.docx' for s in ('manuscript','esi')])
    command([PY,'scripts/verify_submission_20260929.py'])
    check=json.loads((OUT/'verification.json').read_text())
    assert check['passed']==check['total'],check
    command([PY,'scripts/report_submission_revision_20260929.py'])
    render([BASE/'Submission_revision_notes_20260929.docx'])
    print('Final document package completed.',flush=True)

if __name__=='__main__':main()
