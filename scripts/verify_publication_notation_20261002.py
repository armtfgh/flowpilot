#!/usr/bin/env python3
"""Audit values, references, images and layout after publication typesetting."""
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

import fitz
from lxml import etree as E
from PIL import Image, ImageDraw

from typeset_publication_notation_20261002 import (
    ROOT, BASE, OUT, W, M, NS, SCI_NOTE, QUOTE_NOTE,
    plain, sha, dump, digits, typeset_paragraph, upright_math_descriptors,
)
from refresh_revision_toc_20260922 import refresh


def render(paths):
    subprocess.run([
        'libreoffice', '-env:UserInstallation=file:///tmp/flowpilot-notation-20261002',
        '--headless', '--convert-to', 'pdf', '--outdir', str(OUT / 'rendered'),
        *map(str, paths),
    ], check=True, timeout=240)


def package(path):
    with ZipFile(path) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    return parts, E.fromstring(parts['word/document.xml'])


def body_paragraphs(doc):
    return [p for p in doc.iter(W + 'p') if not p.xpath('ancestor::w:sdt', namespaces=NS)]


def captions(doc):
    return [(m[1], int(m[2])) for p in body_paragraphs(doc)
            if (m := re.match(r'^(Figure|Table) (S?\d+)\.', plain(p))) and m[2].isdigit()]


def main():
    reports = json.loads((OUT / 'manifest.json').read_text())
    paths = [ROOT / r['output'] for r in reports]
    render(paths)
    esi = paths[1]
    previous = None
    for _ in range(4):
        with fitz.open(OUT / 'rendered' / esi.with_suffix('.pdf').name) as pdf:
            toc = refresh(esi, pdf)
        render([esi])
        if toc == previous:
            break
        previous = toc
    else:
        raise ValueError('Contents pagination did not stabilize')
    dump(OUT / 'contents_page_audit.json', toc)
    checks, pages = [], []

    def check(name, condition, detail=None):
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})

    for report in reports:
        oldparts, old = package(ROOT / report['source'])
        parts, doc = package(ROOT / report['output'])
        label = Path(report['output']).stem
        check(label + ': original source unchanged', sha(ROOT / report['source']) == report['source_sha256'])
        check(label + ': image bytes unchanged', {k:v for k,v in parts.items() if k.startswith('word/media/')} == {k:v for k,v in oldparts.items() if k.startswith('word/media/')})
        check(label + ': image geometry unchanged', [E.tostring(x) for x in old.iter(W+'drawing')] == [E.tostring(x) for x in doc.iter(W+'drawing')])
        for name in ['word/styles.xml','word/numbering.xml','word/fontTable.xml','word/_rels/document.xml.rels']:
            check(label + ': preserved ' + name, parts[name] == oldparts[name])
        check(label + ': page settings unchanged', [E.tostring(x) for x in old.iter(W+'sectPr')] == [E.tostring(x) for x in doc.iter(W+'sectPr')])
        check(label + ': table count unchanged', len(list(old.iter(W+'tbl'))) == len(list(doc.iter(W+'tbl'))))
        table_shape = lambda d: [[len(tr.findall(W+'tc')) for tr in tb.findall(W+'tr')] for tb in d.iter(W+'tbl')]
        check(label + ': table row/column layout unchanged', table_shape(old) == table_shape(doc))
        geometry = lambda d: [(n.tag, sorted(n.attrib.items())) for n in d.iter() if n.tag in {W+'gridCol', W+'tcW', W+'tblW', W+'trHeight'}]
        check(label + ': table dimensions unchanged', geometry(old) == geometry(doc))
        oldps = body_paragraphs(old)
        newps = [p for p in body_paragraphs(doc) if not plain(p).startswith(('Notation convention.', 'Typography of reproduced reports.'))]
        check(label + ': original paragraph count preserved', len(oldps) == len(newps))
        mismatch, badnumbers = [], []
        for i, (a,b) in enumerate(zip(oldps,newps)):
            original = plain(a)
            simulated = deepcopy(a)
            typeset_paragraph(simulated)
            if plain(simulated) != plain(b):
                mismatch.append(i)
            if digits(original) != digits(plain(b)):
                badnumbers.append(i)
        check(label + ': text changes limited to specified notation', not mismatch, mismatch)
        check(label + ': every original digit/value retained', not badnumbers, badnumbers)
        old_captions = [re.match(r'^(Figure|Table) (S?\d+)\.', plain(p)).group() for p in oldps if re.match(r'^(Figure|Table) (S?\d+)\.', plain(p))]
        new_captions = [re.match(r'^(Figure|Table) (S?\d+)\.', plain(p)).group() for p in newps if re.match(r'^(Figure|Table) (S?\d+)\.', plain(p))]
        check(label + ': caption numbering unchanged', old_captions == new_captions)
        # Equation content remains identical. Only upright descriptive subscripts
        # may differ, not the mathematical structure or numerical values.
        oldmath = list(old.iter(M+'oMath')); newmath = list(doc.iter(M+'oMath'))
        check(label + ': equation count preserved', len(oldmath) == len(newmath))
        check(label + ': equation text preserved', [x.xpath('.//m:t/text()',namespaces=NS) for x in oldmath] == [x.xpath('.//m:t/text()',namespaces=NS) for x in newmath])
        remaining = []
        for i,p in enumerate(body_paragraphs(doc)):
            for token in re.findall(r'(?<!\w)(?:Q|V|tau|\u03c4|T|P|C|A)_[A-Za-z0-9_,]+', plain(p)):
                if token != 'Q_gas_sccm': remaining.append((i,token))
        check(label + ': no residual scientific underscore notation', not remaining, remaining)
        for name,value in parts.items():
            if name.endswith(('.xml','.rels')): E.fromstring(value)
        check(label + ': all package XML parses', True)

        pdfpath = OUT/'rendered'/Path(report['output']).with_suffix('.pdf').name
        visual = OUT/'visual_check'; visual.mkdir(exist_ok=True)
        with fitz.open(pdfpath) as pdf:
            selected = []
            clipped = []
            for i,page in enumerate(pdf):
                t = page.get_text()
                if any(v in t for v in ['Notation convention.', 'Typography of reproduced reports.',
                                       'Figure 5. Photoredox', 'Figure 6. DPDTC',
                                       'Liquid component molar delivery', '(F5-S1): selected',
                                       '(F6-S1): selected', 'General materials, instrumentation',
                                       'LRMS (ESI) m/z:']):
                    selected.append(i)
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for s in line['spans']:
                            if not s['text'].strip(): continue
                            x0,y0,x1,y1=s['bbox']
                            if min(x0,y0) < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                                clipped.append({'page':i+1,'text':s['text']})
            check(label + ': no off-page text', not clipped, clipped)
            tiles = []
            for i in selected:
                path = visual/f'{label}_page_{i+1:03d}.png'
                pdf[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(path)
                with Image.open(path) as im:
                    thumb=im.copy(); thumb.thumbnail((330,460))
                    tile=Image.new('RGB',(350,490),'#dddddd')
                    tile.paste(thumb,((350-thumb.width)//2,10))
                    ImageDraw.Draw(tile).text((10,470),f'Page {i+1}',fill='black')
                    tiles.append(tile)
            if tiles:
                sheet=Image.new('RGB',(1050,490*((len(tiles)+2)//3)),'white')
                for j,tile in enumerate(tiles): sheet.paste(tile,((j%3)*350,(j//3)*490))
                sheet.save(visual/f'{label}_contact.png')
            pages.append({'file':report['output'],'pages':len(pdf),'visual_check_pages':[i+1 for i in selected]})
    dump(OUT/'verification.json',{'checks':checks,'documents':pages})
    failed=[c for c in checks if not c['passed']]
    assert not failed, failed
    print(json.dumps({'passed':len(checks),'failed':len(failed),'documents':pages},indent=2))


if __name__ == '__main__':
    main()
