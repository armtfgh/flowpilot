"""Read-only extraction and source hashes for the second submission review."""
from pathlib import Path
from hashlib import sha256
import json
from lxml import etree as E
from revise_manuscript_cases_20260922 import Package,W,NS,text
from overhaul_submission_20260930 import with_cites

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/october1_postmeeting_round2_work'


def main():
    OUT.mkdir(exist_ok=True)
    manifest={}
    for name in ['manuscript','ESI']:
        path=ROOT/f'manuscript/FlowPilot_{name}_october1_postmeeting.docx'
        pkg=Package(path)
        manifest[str(path.relative_to(ROOT))]=sha256(path.read_bytes()).hexdigest()
        records=[]
        for i,n in enumerate(pkg.body):
            row={'body':i,'kind':E.QName(n).localname,'text':with_cites(n) if n.tag==W+'p' else text(n),
                 'style':n.xpath('./w:pPr/w:pStyle/@w:val',namespaces=NS),
                 'image_ids':n.xpath('.//a:blip/@r:embed',namespaces=NS)}
            if n.tag==W+'tbl':row['rows']=[[text(c) for c in r.findall(W+'tc')] for r in n.findall(W+'tr')]
            records.append(row)
        (OUT/f'{name}_source.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
        (OUT/f'{name}_source.txt').write_text('\n\n'.join(f"[{r['body']}] {r['style']} {r['text']}"+('\n'+'\n'.join(' | '.join(x) for x in r['rows']) if 'rows' in r else '') for r in records))
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
