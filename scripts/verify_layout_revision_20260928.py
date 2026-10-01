"""Verify table consistency, editorial scope, typography and figure geometry."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

import fitz
from lxml import etree as E

import revise_layout_discussion_20260928 as revision
import revise_prior_work_20260928 as prior
from refresh_revision_toc_20260922 import refresh, normalized
from verify_introduction_20260923 import no_highlight_structure

OUT, BASE, ROOT = revision.OUT, revision.BASE, revision.ROOT
W, NS = revision.W, revision.NS
checks = []


def check(name, passed, detail=''):
    checks.append({'check':name,'passed':bool(passed),'detail':detail})


def read(path):
    with ZipFile(path) as z:
        parts = {n:z.read(n) for n in z.namelist()}
    return parts,E.fromstring(parts['word/document.xml'])


def unmarked_structure(doc):
    doc = deepcopy(doc)
    for h in list(doc.iter(W+'highlight')):
        h.getparent().remove(h)
    # Marking a previously unformatted run creates rPr; empty rPr has no effect.
    for props in list(doc.iter(W+'rPr')):
        if not len(props) and not props.attrib and not props.text:
            props.getparent().remove(props)
    return no_highlight_structure(doc)


def refresh_contents():
    results = {}
    for suffix in ['', '_marked']:
        path = BASE/f'esi_revised_20260928_v2{suffix}.docx'
        with fitz.open(OUT/'rendered'/path.with_suffix('.pdf').name) as pdf:
            results[suffix or 'clean'] = refresh(path,pdf)
        parts,doc = read(path)
        prior.bold_references(doc)
        prior.intro_tools.write_package(parts,doc,path)
    (OUT/'contents_page_audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Refreshed ESI contents in both copies; rerender PDFs before validation.')


def node_texts(doc):
    return [prior.text(n) for n in doc.find(W+'body') if n.tag != W+'sdt']


def check_tables(doc, original):
    tables = doc.findall('.//'+W+'tbl')
    before = original.findall('.//'+W+'tbl')
    check('All 39 table contents preserved',len(tables)==len(before)==39 and [prior.text(t) for t in tables]==[prior.text(t) for t in before])
    failures = []
    fonts = []
    header_wrap = []
    font = fitz.Font('tibo')
    for index,t in enumerate(tables,1):
        props = t.find(W+'tblPr')
        border = [(E.QName(n).localname, n.get(W+'val'),n.get(W+'sz'),n.get(W+'color')) for n in props.find(W+'tblBorders')]
        expected = [(edge,'single','4','B8B8B8') for edge in ['top','left','bottom','right','insideH','insideV']]
        if border != expected:
            failures.append((index,'borders'))
        for row_index,row in enumerate(t.findall(W+'tr')):
            for cell in row.findall(W+'tc'):
                fill = cell.find(W+'tcPr').find(W+'shd').get(W+'fill')
                if fill != ('E6E6E6' if row_index==0 else 'FFFFFF'):
                    failures.append((index,'fill',row_index,fill))
                for run in cell.iter(W+'r'):
                    if not prior.text(run).strip():
                        continue
                    if not run.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]',namespaces=NS) or not run.xpath('./w:rPr/w:sz[@w:val="22"]',namespaces=NS):
                        fonts.append((index,prior.text(run)))
                    if row_index==0 and not run.xpath('./w:rPr/w:b[@w:val="1"]',namespaces=NS):
                        failures.append((index,'header_not_bold'))
                if row_index==0:
                    width=int(cell.find(W+'tcPr').find(W+'tcW').get(W+'w'))/20-8
                    too_wide=[s for s in re.findall(r'[A-Za-z]+',prior.text(cell)) if font.text_length(s,fontsize=11)>width]
                    header_wrap.extend((index,s) for s in too_wide)
        if not t.find(W+'tr').xpath('./w:trPr/w:tblHeader',namespaces=NS):
            failures.append((index,'no_repeat_header'))
    check('Consistent gray headers, white bodies, borders and repeat headers',not failures,failures)
    check('All table text is Times New Roman 11 pt',not fonts,fonts)
    check('Header words fit without mid-word breaks',not header_wrap,header_wrap)


def bold_check(doc, stem):
    count, failures = 0, []
    for p in doc.iter(W+'p'):
        value,flags='',[]
        for t in prior.paragraph_text_nodes(p):
            text=t.text or ''
            bold=t.getparent().find(W+'rPr/'+W+'b')
            value += text
            flags.extend([bold is not None and bold.get(W+'val','1') not in {'0','false','off'}]*len(text))
        for match in prior.LABEL.finditer(value):
            count += 1
            if not all(flags[match.start():match.end()]):
                failures.append(match[0])
    check(stem+': figure/table labels remain bold',not failures,{'labels':count,'unbolded':failures})


def main():
    audit=json.loads((OUT/'document_revision_audit.json').read_text())
    for path,h in audit['source_hashes'].items():
        check('Source unchanged: '+path,sha256((ROOT/path).read_bytes()).hexdigest()==h)
    for stem in ['manuscript','esi']:
        source_parts,source=read(BASE/f'{stem}_revised_20260928.docx')
        parts,doc=read(BASE/f'{stem}_revised_20260928_v2.docx')
        marked_parts,marked=read(BASE/f'{stem}_revised_20260928_v2_marked.docx')
        allowed={'word/document.xml'}
        if stem=='esi':allowed.update('word/'+x for x in audit['esi']['replaced_media'])
        changed={n for n in source_parts if source_parts[n]!=parts.get(n)}
        check(stem+': only authorized package parts changed',set(source_parts)==set(parts) and changed<=allowed,sorted(changed))
        check(stem+': marked and clean differ only in highlights',unmarked_structure(doc)==unmarked_structure(marked))
        check(stem+': originals and replacements keep image assignments',source.xpath('.//a:blip/@r:embed',namespaces=NS)==doc.xpath('.//a:blip/@r:embed',namespaces=NS))
        bold_check(doc,stem)
        if stem=='manuscript':
            old,new=node_texts(source),node_texts(doc)
            a,b=old.index('DISCUSSION'),old.index('CONCLUSION')
            c,d=new.index('DISCUSSION'),new.index('CONCLUSION')
            check('Main: every other section and bibliography unchanged',old[:a+1]==new[:c+1] and old[b:]==new[d:])
            check('Main: five continuous Discussion paragraphs',d-c-1==5 and not any(s in '\n'.join(new[c+1:d]) for s in ['Limitations and scope boundaries','Positioning relative to prior AI systems','Outlook']))
            check('Main: Discussion cut by more than half',audit['manuscript']['new_discussion_words'] < audit['manuscript']['old_discussion_words']*0.5,audit['manuscript'])
            ps=doc.find(W+'body').findall(W+'p')
            ref_index=next(i for i,p in enumerate(ps) if prior.text(p)=='REFERENCES')
            refs=[p for p in ps[ref_index+1:] if prior.text(p).strip()]
            ids=[]
            for p in ps[:ref_index]:
                for _,values in prior.intro_tools.citation_runs(p):
                    ids.extend(x for x in values if x not in ids)
            check('Main: all 62 references still cited in order',ids==list(range(1,len(refs)+1)) and len(refs)==62)
            cited={'Figure':set(),'Table':set()}
            for match in prior.LABEL.finditer(prior.text(doc)):
                value=match[0]
                kind='Table' if value.lower().startswith('table') else 'Figure'
                for a,b in re.findall(r'S(\d+)\s*[-\u2013]\s*S?(\d+)',value):
                    cited[kind].update(range(int(a),int(b)+1))
                cited[kind].update(map(int,re.findall(r'S(\d+)',value)))
            check('Main: all ESI figures S1-S22 remain cited',set(range(1,23))<=cited['Figure'])
            check('Main: all ESI tables S1-S36 remain cited',set(range(1,37))<=cited['Table'])
        else:
            check_tables(doc,source)
            old,new=node_texts(source),node_texts(doc)
            allowed_prefixes=('Figure S13.','Figure S17 reopens','Figure S17.','The companion evidence package','(a) Actual saved GUI topology','(b) Actual saved GUI topology','(c) Actual saved GUI topology')
            diffs=[(a,b) for a,b in zip(old,new) if a!=b]
            check('ESI: only identified diagram captions/provenance changed',len(old)==len(new) and all(a.startswith(allowed_prefixes) for a,b in diffs),{'changed_paragraphs':len(diffs)})
        for suffix in ['', '_marked']:
            pdf=fitz.open(OUT/'rendered'/f'{stem}_revised_20260928_v2{suffix}.pdf')
            clipped,empty=[],[]
            for i,page in enumerate(pdf):
                if not page.get_text().strip() and not page.get_images():empty.append(i+1)
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line['spans']:
                            box=fitz.Rect(span['bbox'])
                            if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width+1 or box.y1 > page.rect.height+1:clipped.append((i+1,span['text']))
            check(f'{stem}{suffix}: no out-of-page text',not clipped,clipped or {'pages':len(pdf)})
            check(f'{stem}{suffix}: no blank pages',not empty,empty)
            pdf.close()
    diagrams=json.loads((OUT/'figure_layout_audit.json').read_text())
    check('Eight diagram panels regenerated',len(diagrams)==8)
    for row in diagrams:
        check(row['figure']+': archived JSON unchanged',sha256((ROOT/row['source_json']).read_bytes()).hexdigest()==row['source_json_sha256'])
        ys=[row['node_positions'][k][1] for k in row['main_path']]
        check(row['figure']+': horizontal main process path',max(ys)-min(ys)<1e-6)
        orthogonal=all(abs(a[0]-b[0])<1e-6 or abs(a[1]-b[1])<1e-6 for edge in row['edges'] for a,b in zip(edge['points'],edge['points'][1:]))
        check(row['figure']+': orthogonal routes and separated labels',orthogonal and not row['label_box_overlaps'])
    toc=json.loads((OUT/'contents_page_audit.json').read_text())
    for suffix,rows in toc.items():
        ending='' if suffix=='clean' else suffix
        with fitz.open(OUT/'rendered'/f'esi_revised_20260928_v2{ending}.pdf') as pdf:
            bad=[r for r in rows if normalized(r['heading']) not in normalized(pdf[r['page']-1].get_text())]
            check('ESI contents matches final pagination '+suffix,not bad,bad)
    result={'passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks}
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[c for c in checks if not c['passed']]},indent=2))
    assert all(c['passed'] for c in checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh-toc',action='store_true')
    args=parser.parse_args()
    refresh_contents() if args.refresh_toc else main()
