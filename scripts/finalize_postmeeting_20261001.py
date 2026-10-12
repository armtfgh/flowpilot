"""Render, refresh contents and verify the postmeeting reading/review copies."""
import json
from pathlib import Path
import subprocess
import sys
import fitz
from refresh_revision_toc_20260922 import refresh

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript'
OUT=BASE/'october1_postmeeting_work'


def render(paths):
    subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/flowpilot-postmeeting-final-render',
        '--headless','--convert-to','pdf','--outdir',str(OUT/'rendered'),*[str(p) for p in paths]],
        check=True,timeout=240)


def main():
    main=BASE/'FlowPilot_manuscript_october1_postmeeting.docx'
    esi=BASE/'FlowPilot_ESI_october1_postmeeting.docx'
    render([main,esi])
    previous=None
    audits=[]
    for _ in range(4):
        with fitz.open(OUT/'rendered'/esi.with_suffix('.pdf').name) as pdf:
            audit=refresh(esi,pdf)
            refresh(OUT/'marked'/'FlowPilot_ESI_october1_postmeeting_marked.docx',pdf)
        audits.append(audit)
        render([esi])
        if audit==previous:break
        previous=audit
    else:
        raise RuntimeError('Contents page numbers did not stabilize')
    (OUT/'toc_audit.json').write_text(json.dumps(audits,indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_postmeeting_20261001.py')],check=True)


if __name__=='__main__':main()
