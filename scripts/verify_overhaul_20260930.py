"""Cross-check the shortened submission against source records and rendered pages."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import re
import shutil

import fitz
from lxml import etree as E

import overhaul_submission_20260930 as rev
import verify_professional_review_20260930 as evidence
import verify_submission_20260929 as layout
from verify_manuscript_20260930 import xml, images
from refresh_revision_toc_20260922 import normalized
from revise_manuscript_cases_20260922 import expanded_cites

OUT, BASE, ROOT = rev.OUT, rev.BASE, rev.ROOT
W, NS, text = rev.W, rev.NS, rev.text
checks = []


def check(name, passed, detail=None):
    checks.append({'check': name, 'passed': bool(passed), 'detail': detail})


def labeled(value, supplemental=True):
    marker = 'S' if supplemental else ''
    pattern = re.compile(r'\b(Figures?|Tables?)\s+(' + marker + r'\d+[a-z]?(?:\s*(?:[–-]|,\s*(?:and\s+)?|\band\b)\s*' + marker + r'?\d+[a-z]?)*)')
    found = []
    for m in pattern.finditer(value):
        for bit in re.split(r'\s*(?:,\s*(?:and\s+)?|and)\s*', m[2]):
            nums = [int(n) for n in re.findall(r'\d+', bit)]
            for number in (range(nums[0], nums[1]+1) if len(nums) == 2 else nums):
                found.append(('Figure' if m[1].startswith('Figure') else 'Table', number))
    return found


def crossrefs(main, esi):
    definitions = {}
    for p in esi.body:
        m = re.match(r'^(Figure|Table) S(\d+)\.', text(p))
        if m:
            key = (m[1], int(m[2]))
            check('Unique caption: '+m[0], key not in definitions)
            definitions[key] = text(p)
    report = []
    first = {'Figure': [], 'Table': []}
    for i, p in enumerate(main.body):
        for key in labeled(text(p)):
            if key[1] not in first[key[0]]:
                first[key[0]].append(key[1])
                report.append({'label': f'{key[0]} S{key[1]}', 'main_body_index': i,
                               'first_citation': text(p), 'caption': definitions.get(key, 'MISSING')})
    for kind, count in [('Figure', 23), ('Table', 25)]:
        check(kind+' captions sequential in ESI', [n for k,n in definitions if k==kind] == list(range(1,count+1)))
        check(kind+' first citations sequential in manuscript', first[kind] == list(range(1,count+1)), first[kind])
        esi_first=[]
        for p in esi.body:
            if p.tag==W+'sdt':continue
            for k,n in labeled(text(p)):
                if k==kind and n not in esi_first:esi_first.append(n)
        check(kind+' first citations sequential within ESI',esi_first==list(range(1,count+1)),esi_first)
    missing = [(tag, text(p)[:150], key) for tag, pkg in [('main',main),('esi',esi)]
               for p in pkg.body if p.tag != W+'sdt' for key in labeled(text(p)) if key not in definitions]
    check('Every main/ESI supplementary figure/table reference resolves', not missing, missing)
    for tag,pkg in [('main',main),('esi',esi)]:
        stray=[text(p)[:200] for p in pkg.body if p.tag!=W+'sdt' and any(int(n)>25 for n in re.findall(r'\bS(\d+)\b',text(p)))]
        check(tag+': no stray obsolete supplementary numbers',not stray,stray)
    first_main = []
    for p in main.body:
        for kind, number in labeled(text(p), False):
            if kind=='Figure' and number not in first_main:
                first_main.append(number)
    check('Main figures first cited in order 1-6', first_main == list(range(1,7)), first_main)
    for pkg, name in [(main,'main'),(esi,'esi')]:
        refs = [m[1] for p in pkg.body if p.tag!=W+'sdt' for m in re.finditer(r'\bSections? (\d+(?:\.\d+)?)',text(p))]
        check(name+': all section references use revised range', all(1<=int(v.split('.')[0])<=10 for v in refs), sorted(set(refs)))
    with (OUT/'citation_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)


def package_checks():
    packages={}
    for stem, source in [('manuscript',rev.MAIN),('esi',rev.ESI)]:
        original=rev.Package(source)
        pkg=rev.Package(BASE/f'{stem}_submission_overhauled_20260930.docx')
        marked=rev.Package(BASE/f'{stem}_submission_overhauled_20260930_marked.docx')
        packages[stem]=pkg
        check(stem+': original styles, numbering and fonts preserved', all(pkg.parts.get(k)==original.parts.get(k) for k in ['word/styles.xml','word/numbering.xml','word/fontTable.xml']))
        check(stem+': page geometry preserved', xml(pkg.body[-1])==xml(original.body[-1]))
        check(stem+': prior embedded media retained', all(pkg.parts.get(k)==v for k,v in original.parts.items() if k.startswith('word/media/')))
        check(stem+': clean and marked text identical',text(pkg.doc)==text(marked.doc))
        check(stem+': yellow revision highlighting present',bool(marked.doc.xpath('.//w:highlight[@w:val="yellow"]',namespaces=NS)))
        bad=[]
        for run in marked.doc.xpath('.//w:r[w:rPr/w:highlight[@w:val="yellow"]]',namespaces=NS):
            if text(run).strip() and (not run.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]',namespaces=NS) or not run.xpath('./w:rPr/w:sz[@w:val="22"]',namespaces=NS)):
                bad.append(text(run))
        check(stem+': revisions use Times New Roman 11',not bad,bad)
        rels={r.get('Id'):r.get('Target') for r in pkg.rels}
        missing=[rid for rid in pkg.doc.xpath('.//a:blip/@r:embed',namespaces=NS) if 'word/'+rels.get(rid,'MISSING') not in pkg.parts]
        check(stem+': image relationships resolve',not missing,missing)
        check(stem+': no placeholder names or yield markers',not any(v in text(pkg.doc) for v in ['[[','XXX','YYY','[Name to be added]']))
        format_view=deepcopy(pkg.doc)
        if stem=='esi':
            # Caption bodies and the contents are deliberately not bold references.
            for node in list(format_view.find(W+'body')):
                if node.tag==W+'sdt' or re.match(r'^(Figure|Table) S\d+(?: \(continued\))?\.',text(node)):
                    node.getparent().remove(node)
        layout.fonts_tables(format_view,stem)
    m,s=packages['manuscript'],packages['esi']
    check('Author lines agree',text(m.body[1])==text(s.body[1]))
    for email in ['boyoungy.park@khu.ac.kr','gnahn@krict.re.kr']:
        check('Both corresponding-author addresses retained: '+email,email in text(m.doc) and email in text(s.doc))
    for number in range(1,5):
        source=rev.Package(BASE/'manuscript_revised_20260928.docx' if number==4 else rev.MAIN)
        p,_=rev.old.base.image_before_caption(m,number)
        old,_=rev.old.base.image_before_caption(source,number)
        check(f'Main Figure {number}: original artwork and dimensions preserved',xml(p)==xml(old) and images(m,p)==images(source,old))
    for number in (5,6):
        p,c=rev.old.base.image_before_caption(m,number)
        values=images(m,p)
        check(f'Main Figure {number}: redesigned artwork embedded',sha256((OUT/f'figures/figure{number}_redesigned.png').read_bytes()).hexdigest() in values)
        check(f'Main Figure {number}: NMR measurement explicitly defined', 'NMR' in text(c) and 'isolated yield' not in text(c))
    check('No main-text comparison table',not list(m.doc.iter(W+'tbl')))
    check('Grouped laboratory results not treated as independent repeats', 'not independent repeats' in text(m.doc) and 'share one supplied' in text(m.doc))
    check('Negative and mixed benchmark evidence retained',all(v in text(s.doc) for v in ['evaluator false positive','98','0.31 ± 0.03','0.48 ± 0.23']))
    check('Source-isolation caveat retained', 'not proof of absence from pretraining' in text(s.doc))
    check('No isolated-yield comparison in main', 'isolated yield' not in text(m.doc))
    check('Conclusion expanded to substantive synthesis',len(' '.join(text(p) for p in rev.old.base.section(m,'CONCLUSION','METHODS')).split())>=250)
    caption_errors=[]
    for p in s.body:
        match=re.match(r'^(Figure|Table) S\d+(?: \(continued\))?\.',text(p))
        if not match:
            continue
        position=0
        bold_prefix=False
        for run in p.iter(W+'r'):
            value=text(run)
            prop=run.find(W+'rPr')
            b=prop.find(W+'b') if prop is not None else None
            bold=b is not None and b.get(W+'val','1') not in ['0','false','off']
            if position<match.end() and value.strip():
                bold_prefix=bold_prefix or bold
            elif value.strip() and bold:
                caption_errors.append(text(p)[:90])
            position+=len(value)
        if not bold_prefix:
            caption_errors.append('Missing bold label: '+text(p)[:50])
    check('ESI caption label bold; explanatory text regular',not caption_errors,caption_errors)
    crossrefs(m,s)
    # Numeric superscripts are literature citations, except isotope labels.
    heading=rev.old.base.find(m,'REFERENCES')
    order=[]
    for p in list(m.body)[:list(m.body).index(heading)]:
        for match in re.finditer(r'\[\[(\d+(?:[,–-]\d+)*)\]\]',rev.with_cites(p)):
            for number in expanded_cites(match[1]):
                if number not in order:
                    order.append(number)
    check('Main bibliography first appearances ordered 1-61',order==list(range(1,62)),order)
    refs=[text(p) for p in list(m.body)[list(m.body).index(heading)+1:] if text(p).strip()]
    check('Main has 61 retained references',len(refs)==61)
    eheading=next(p for p in s.body if text(p)=='References')
    eindex=list(s.body).index(eheading)
    eorder=[]
    for p in list(s.body)[:eindex]:
        for match in re.finditer(r'\[([0-9]+(?:[,–-][0-9]+)*)\]|\breference ([0-9]+)\b',text(p)):
            for number in expanded_cites(match[1] or match[2]):
                if number not in eorder:eorder.append(number)
    check('ESI bibliography first appearances ordered 1-12',eorder==list(range(1,13)),eorder)
    toc=json.loads((OUT/'contents_page_audit.json').read_text())
    sections={row['section'] for row in toc}
    for tag,pkg in [('main',m),('esi',s)]:
        unresolved=[]
        for p in pkg.body:
            if p.tag==W+'sdt':
                continue
            for match in re.finditer(r'\bSections? (\d+(?:\.\d+)?)',text(p)):
                if match[1] not in sections:
                    unresolved.append((match[1],text(p)[:120]))
        check(tag+': section references exist in refreshed contents',not unresolved,unresolved)
    return packages


def scientific_checks():
    evidence.checks=checks
    evidence.evidence()
    with (OUT/'yield_comparison.csv').open() as f:
        data=list(csv.DictReader(f))
    check('Figure 5 yields match verified NMR sources',[int(r['yield_pct']) for r in data if r['figure']=='5']==[90,97,98])
    check('Figure 6 yields match verified NMR sources',[int(r['yield_pct']) for r in data if r['figure']=='6']==[98,86,68])
    check('Literature batch/flow distinction preserved',data[0]['label']=='Literature flow' and data[3]['label']=='Literature batch')
    for name in ['figure5_redesigned','figure6_redesigned','figureS17_irradiation_sources']:
        audit=json.loads((OUT/f'{name}_layout_audit.json').read_text())
        check(name+': no text-text overlaps or out-of-canvas labels',not audit['issues'],audit['issues'])
    check('Original ESI remains in companion archive',sha256((OUT/'companion_archive/esi_before_streamlining.docx').read_bytes()).hexdigest()==sha256(rev.ESI.read_bytes()).hexdigest())
    for path,digest in json.loads((OUT/'source_manifest.json').read_text()).items():
        check('Source unchanged: '+Path(path).name,sha256((ROOT/path).read_bytes()).hexdigest()==digest)


def rendered():
    maps={}
    for stem in ['manuscript','esi']:
        path=OUT/f'rendered/{stem}_submission_overhauled_20260930.pdf'
        marked_path=path.with_stem(path.stem+'_marked')
        maps[stem]=layout.rendered_pdf(path,stem)
        clean,marked=fitz.open(path),fitz.open(marked_path)
        check(stem+': clean/marked page counts agree',len(clean)==len(marked))
        check(stem+': clean/marked rendered text agrees',[''.join(p.get_text().split()) for p in clean]==[''.join(p.get_text().split()) for p in marked])
        detached=[]
        for i,page in enumerate(clean):
            captions=re.findall(r'^Figure S?\d+(?: \(continued\))?\.',page.get_text(),re.MULTILINE)
            if captions and not page.get_image_info():detached.append({'page':i+1,'captions':captions})
        check(stem+': figure captions share a page with artwork',not detached,detached)
        for i,page in enumerate(clean):
            if any(needle in page.get_text() for needle in ['Figure 5.', 'Figure 6.', 'Table S1.', 'Figure S17.', 'CONCLUSION']):
                page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/f'visual_review/{stem}_detail_{i+1:03d}.png')
        shutil.copy2(path,BASE/path.name)
    doc=fitz.open(OUT/'rendered/esi_submission_overhauled_20260930.pdf')
    toc=json.loads((OUT/'contents_page_audit.json').read_text())
    missing=[r for r in toc if normalized(r['heading']) not in normalized(doc[r['page']-1].get_text())]
    check('All contents entries point to rendered headings',not missing,missing)
    (OUT/'rendered_page_map.json').write_text(json.dumps(maps,indent=2)+'\n')


def main():
    layout.OUT,layout.checks=OUT,checks
    package_checks();scientific_checks();rendered()
    result={'passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[c for c in checks if not c['passed']]},indent=2))
    assert result['passed']==result['total']


if __name__=='__main__':
    main()
