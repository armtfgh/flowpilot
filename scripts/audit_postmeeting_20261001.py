"""Extract the supplied October 1 documents for a non-destructive editorial audit."""
from hashlib import sha256
from pathlib import Path
import json

from lxml import etree as E
from revise_manuscript_cases_20260922 import Package, W, NS, text
from overhaul_submission_20260930 import with_cites

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/october1_postmeeting_work'


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for name in ['manuscript', 'ESI']:
        source = ROOT / f'manuscript/FlowPilot_{name}__20261001.docx'
        pkg = Package(source)
        media = OUT / 'source_media' / name
        media.mkdir(parents=True, exist_ok=True)
        rels = {r.get('Id'): r.get('Target') for r in pkg.rels}
        records, lines = [], []
        pi = 0
        for bi, node in enumerate(pkg.body):
            kind = E.QName(node).localname
            value = with_cites(node) if kind == 'p' else text(node)
            entry = {'body': bi, 'paragraph': pi if kind == 'p' else None,
                     'kind': kind, 'text': value, 'images': []}
            for rid in node.xpath('.//a:blip/@r:embed', namespaces=NS):
                target = rels[rid]
                part = target.lstrip('/') if target.startswith('/') else 'word/' + target
                path = media / Path(part).name
                path.write_bytes(pkg.parts[part])
                entry['images'].append(str(path.relative_to(ROOT)))
            if kind == 'p':
                entry['style'] = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
                entry['bold'] = [text(r) for r in node.xpath('./w:r[w:rPr/w:b[not(@w:val="0" or @w:val="false")]]', namespaces=NS)]
                pi += 1
            elif kind == 'tbl':
                entry['rows'] = [[text(c) for c in r.findall(W + 'tc')] for r in node.findall(W + 'tr')]
            lines.append(f"[body {bi} | p {entry['paragraph']} | {kind}] {value}")
            if entry['images']:
                lines.append('IMAGES: ' + ', '.join(entry['images']))
            if kind == 'tbl':
                lines.extend(' | '.join(row) for row in entry['rows'])
            records.append(entry)
        (OUT / f'{name}_source.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
        (OUT / f'{name}_source.txt').write_text('\n\n'.join(lines))
        manifest[str(source.relative_to(ROOT))] = sha256(source.read_bytes()).hexdigest()
        print(name, len(records), 'body blocks;', pi, 'paragraphs;', len(list(media.iterdir())), 'media files')
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
