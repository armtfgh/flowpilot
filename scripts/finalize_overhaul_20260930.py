"""Converge the contents page numbers and render both revision variants."""
import json
import subprocess
import sys

import fitz

from overhaul_submission_20260930 import ROOT, BASE, OUT
from refresh_revision_toc_20260922 import refresh, normalized


def render(paths):
    subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/flowpilot_overhaul_20260930',
                    '--headless','--convert-to','pdf','--outdir',str(OUT/'rendered'),
                    *map(str,paths)],cwd=ROOT,check=True)


def main():
    clean=BASE/'esi_submission_overhauled_20260930.docx'
    marked=clean.with_stem(clean.stem+'_marked')
    for cycle in range(4):
        pdf=fitz.open(OUT/'rendered'/clean.with_suffix('.pdf').name)
        rows=refresh(clean,pdf)
        refresh(marked,pdf)
        (OUT/'contents_page_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
        pdf.close()
        render([clean])
        with fitz.open(OUT/'rendered'/clean.with_suffix('.pdf').name) as document:
            missing=[r for r in rows if normalized(r['heading']) not in normalized(document[r['page']-1].get_text())]
        if not missing:
            print('Contents page audit converged after',cycle+1,'pass(es).',flush=True)
            break
    else:
        raise AssertionError(missing)
    render([BASE/f'{stem}_submission_overhauled_20260930_marked.docx' for stem in ['manuscript','esi']])
    subprocess.run([sys.executable,'scripts/verify_overhaul_20260930.py'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
