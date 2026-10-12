"""Inventory the latest meeting documents without modifying their sources."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manuscript'
OUT = BASE / 'october2_final_review'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS = {'w': W[1:-1], 'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'v': 'urn:schemas-microsoft-com:vml'}

def text(n):
    return ''.join(n.xpath('.//w:t/text()', namespaces=NS))

def inspect(path, label):
    with ZipFile(path) as z:
        d = E.fromstring(z.read('word/document.xml'))
        rels = {n.get('Id'): n.get('Target') for n in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        lines, items = [], []
        for i, n in enumerate(d.find(W+'body')):
            kind = E.QName(n).localname
            style = n.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
            images = [rels.get(r, r) for r in n.xpath('.//a:blip/@r:embed | .//v:imagedata/@r:id', namespaces=NS)]
            if kind == 'tbl':
                rows = [[text(c) for c in tr.findall(W+'tc')] for tr in n.findall(W+'tr')]
                value = '\n'.join(' | '.join(r) for r in rows)
            else:
                value = text(n)
            items.append({'index':i, 'kind':kind, 'style':style, 'images':images,'text':value})
            lines.append(f'[{i:04d}] {kind} {style} {images}\n{value}\n')
        (OUT/f'{label}_body.txt').write_text('\n'.join(lines))
        (OUT/f'{label}_body.json').write_text(json.dumps(items,indent=2,ensure_ascii=False))
        return {'file':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'body_items':len(items), 'tables':sum(x['kind']=='tbl' for x in items),
                'image_paragraphs':sum(bool(x['images']) for x in items)}

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    sources = {'main':'FlowPilot_manuscript_october2_notation_corrected.docx',
               'esi':'FlowPilot_ESI_october2_notation_corrected.docx',
               'experimental':'Supplementary Information (KRICT)_final (1).docx'}
    result = {label:inspect(BASE/name,label) for label,name in sources.items()}
    (OUT/'sources.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
