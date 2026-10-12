"""Audit the meeting revision without changing source data or documents."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import csv
import json
import posixpath
import re
import shutil

from lxml import etree as E
import fitz

from revise_manuscript_cases_20260922 import Package,W,NS,text
from review_october2_final import inspect
from verify_overhaul_20260930 import labeled
from typeset_publication_notation_20261002 import typeset_paragraph

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript'
OUT=BASE/'october2_final_review'
CHECKS=[]

def check(name,passed,detail=None):
    CHECKS.append({'check':name,'passed':bool(passed),'detail':detail})

def media(pkg,node):
    rels={r.get('Id'):r.get('Target') for r in pkg.rels}
    ids=node.xpath('.//a:blip/@r:embed | .//@r:id',namespaces=NS)
    return [sha256(pkg.parts[posixpath.normpath('word/'+rels[r])]).hexdigest() for r in ids if r in rels and posixpath.normpath('word/'+rels[r]) in pkg.parts]

def graphic(pkg,number):
    items=list(pkg.body)
    i=next(i for i,n in enumerate(items) if text(n).startswith(f'Figure {number}.'))
    return next(n for n in reversed(items[:i]) if n.xpath('.//a:blip',namespaces=NS))

def crossrefs(main,esi):
    defs={}
    for node in esi.body:
        m=re.match(r'^(Figure|Table) S(\d+)\.',text(node))
        if m:
            key=m[1],int(m[2]);check('Unique caption '+m[0],key not in defs)
            defs[key]=text(node)
    report=[]
    for doc,pkg in [('main',main),('esi',esi)]:
        first={'Figure':[],'Table':[]}
        missing=[]
        for i,node in enumerate(pkg.body):
            if node.tag==W+'sdt':continue
            for key in labeled(text(node)):
                if key not in defs:missing.append((i,key,text(node)[:100]))
                if key[1] not in first[key[0]]:
                    first[key[0]].append(key[1]);report.append({'document':doc,'label':f'{key[0]} S{key[1]}','body_index':i,'first_citation':text(node),'caption':defs.get(key,'MISSING')})
        check(doc+': every supplementary reference resolves',not missing,missing)
        for kind,count in [('Figure',27),('Table',12)]:
            check(doc+': '+kind+' first citations sequential',first[kind]==list(range(1,count+1)),first[kind])
    for kind,count in [('Figure',27),('Table',12)]:
        check(kind+' captions numbered sequentially', [n for k,n in defs if k==kind]==list(range(1,count+1)))
    first=[]
    for node in main.body:
        for k,n in labeled(text(node),False):
            if k=='Figure' and n not in first:first.append(n)
    check('Main Figures 1-6 sequentially cited',first==list(range(1,7)),first)
    with (OUT/'citation_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=report[0]);writer.writeheader();writer.writerows(report)

def main():
    packages={}
    sources=json.loads((OUT/'sources.json').read_text())
    for label,source_label in [('manuscript','main'),('ESI','esi')]:
        src=BASE/f'FlowPilot_{label}_october2_notation_corrected.docx'
        path=BASE/f'FlowPilot_{label}_october2_final_real1.docx'
        orig=Package(src);pkg=Package(path);packages[label]=pkg
        inspect(path,'final_'+source_label)
        check(label+': source untouched',sha256(src.read_bytes()).hexdigest()==sources[source_label]['sha256'])
        for part in ['word/styles.xml','word/numbering.xml','word/fontTable.xml']:
            check(label+': retained '+part,orig.parts.get(part)==pkg.parts.get(part))
        check(label+': page geometry unchanged',E.tostring(orig.body[-1])==E.tostring(pkg.body[-1]))
        rels={r.get('Id'):r for r in pkg.rels}
        bad=[]
        for rid in pkg.doc.xpath('.//@r:id | .//@r:embed',namespaces=NS):
            rel=rels.get(rid)
            if rel is None:bad.append(rid)
            elif rel.get('TargetMode')!='External' and posixpath.normpath('word/'+rel.get('Target')) not in pkg.parts:bad.append(rid)
        check(label+': all internal relationships resolve',not bad,bad)
        prose='\n'.join(text(n) for n in pkg.body if n.tag!=W+'sdt')
        check(label+': institutional shorthand removed except correct email',not re.search(r'\bkhu\b',prose.replace('boyoungy.park@khu.ac.kr',''),re.I))
        alt=' '.join(v for node in pkg.doc.iter() for k,v in node.attrib.items() if E.QName(k).localname in {'descr','title'})
        check(label+': image descriptions contain no institutional shorthand',not re.search(r'\bkhu\b',alt,re.I))
        check(label+': no voting terminology in text',not re.search(r'\b(?:vot\w*|veto\w*)\b',prose,re.I))
        check(label+': no placeholders or broken references',not any(s in prose for s in ['[[','XXX','YYY','Error! Reference source','REF!']))
        check(label+': author line unchanged',text(orig.body[1])==text(pkg.body[1]))
        check(label+': corresponding emails retained',all(s in prose for s in ['boyoungy.park@khu.ac.kr','gnahn@krict.re.kr']))
        for name,value in pkg.parts.items():
            if name.endswith(('.xml','.rels')):E.fromstring(value)
        check(label+': all package XML parses',True)
        mns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
        check(label+': equation contents preserved', orig.doc.xpath('.//m:oMath//m:t/text()',namespaces=mns)==pkg.doc.xpath('.//m:oMath//m:t/text()',namespaces=mns))
        def bibliography(p):
            nodes=list(p.body);i=next(i for i,n in enumerate(nodes) if text(n).upper()=='REFERENCES')
            return [text(n) for n in nodes[i+1:] if text(n)]
        check(label+': bibliography unchanged',bibliography(orig)==bibliography(pkg))
        residual=[]
        for node in pkg.doc.iter(W+'p'):
            if node.xpath('ancestor::w:sdt',namespaces=NS):continue
            for token in re.findall(r'(?<!\w)(?:Q|V|tau|\u03c4|T|P|C|A)_[A-Za-z0-9_,]+',text(node)):
                if token!='Q_gas_sccm':residual.append(token)
        check(label+': no untypeset scientific underscores',not residual,residual)
        caption_errors=[]
        for node in pkg.body:
            match=re.match(r'^(Figure|Table) S?\d+\.',text(node))
            if not match:continue
            offset=0
            for r in node.findall(W+'r'):
                v=text(r);b=r.find(W+'rPr/'+W+'b')
                bold=b is not None and b.get(W+'val','1') not in ['0','false','off']
                if v.strip() and bold!=(offset<match.end()):caption_errors.append((text(node)[:35],v))
                offset+=len(v)
        check(label+': captions have bold labels only',not caption_errors,caption_errors[:10])
    m,e=packages['manuscript'],packages['ESI']
    om=Package(BASE/'FlowPilot_manuscript_october2_notation_corrected.docx')
    oe=Package(BASE/'FlowPilot_ESI_october2_notation_corrected.docx')
    for i in range(1,5):
        check(f'Main Figure {i}: unchanged artwork',media(m,graphic(m,i))==media(om,graphic(om,i)))
        check(f'Main Figure {i}: unchanged placement',E.tostring(graphic(m,i))==E.tostring(graphic(om,i)))
    crossrefs(m,e)
    items=list(e.body)
    capindex=next(i for i,n in enumerate(items) if text(n).startswith('Table S6.'))
    tb=next(n for n in items[capindex+1:] if n.tag==W+'tbl')
    rows=[[text(c) for c in r.findall(W+'tc')] for r in tb.findall(W+'tr')]
    wanted=['No council','One-shot','Without Chemistry agent','Without Fluidics agent','Without Safety agent','Without Kinetics agent','Full FlowPilot']
    check('Ablation table matches seven Figure 4d conditions', [r[0] for r in rows[1:]]==wanted,rows)
    oldtb=list(oe.body)[182]
    original={text(r.findall(W+'tc')[0]):[text(c) for c in r.findall(W+'tc')] for r in oldtb.findall(W+'tr')[1:]}
    check('Ablation table data unchanged',all(r==original[r[0]] for r in rows[1:]))
    source_data=OUT/'source_data';source_data.mkdir(exist_ok=True)
    shutil.copy2(BASE/'Submission/overhaul_20260930/source_data/tableS10_all15_module_conditions.csv',source_data/'complete_15_condition_screen.csv')
    with (source_data/'tableS6_seven_displayed_conditions.csv').open('w',newline='') as f:
        csv.writer(f).writerows(rows)
    for oldi,newnum in [(270,10),(293,11)]:
        # Locate the response tables by captions; avoid hard-coded source table offsets.
        oldnum=newnum+3
        olditems=list(oe.body)
        oi=next(i for i,n in enumerate(olditems) if text(n).startswith(f'Table S{oldnum}.'))
        ni=next(i for i,n in enumerate(items) if text(n).startswith(f'Table S{newnum}.'))
        ot=next(n for n in olditems[oi+1:] if n.tag==W+'tbl')
        nt=next(n for n in items[ni+1:] if n.tag==W+'tbl')
        check(f'Response Table S{newnum}: confirmed answer text unchanged',text(ot)==text(nt))
    original_experiment=Package(BASE/'Supplementary Information (KRICT)_final (1).docx')
    targets=[sha256(v).hexdigest() for k,v in e.parts.items() if k.startswith('word/media/meeting_original')]
    for name in ('image10.emf','image12.emf'):
        check('Original experimental vector restored: '+name,sha256(original_experiment.parts['word/media/'+name]).hexdigest() in targets)
    captions='\n'.join(text(n) for n in e.body if re.match(r'^(Figure|Table) S\d+\.',text(n)))
    for removed in ['Audit of the handbook-derived rule base','Available literature-corpus fields','Matched one-shot and FlowPilot architecture scores by generator model','Aggregate architecture-level score and critical-error summary','Proposal-to-implementation reconciliation.','Nominal component feeds from reported stock']:
        check('Requested content removed: '+removed,removed not in captions)
    check('Restored major timeline table', 'Major development milestones' in captions)
    for label in ['manuscript','ESI']:
        pdfpath=OUT/'rendered'/f'FlowPilot_{label}_october2_final_real1.pdf'
        if not pdfpath.exists():continue
        pdf=fitz.open(pdfpath);bad=[];blank=[];caption_orphans=[]
        for i,page in enumerate(pdf):
            for b in page.get_text('dict')['blocks']:
                for line in b.get('lines',[]):
                    if line['bbox'][0]<-1 or line['bbox'][2]>page.rect.width+1:bad.append((i+1,line['bbox']))
            if len(page.get_text().strip())<8 and not page.get_image_info():blank.append(i+1)
        check(label+': no horizontally off-page text',not bad,bad)
        check(label+': no blank pages',not blank,blank)
    (OUT/'quality_audit.json').write_text(json.dumps(CHECKS,indent=2,ensure_ascii=False))
    manifest=[]
    for p in [BASE/'FlowPilot_manuscript_october2_final_real1.docx',BASE/'FlowPilot_ESI_october2_final_real1.docx',*(OUT/'rendered').glob('*.pdf')]:
        manifest.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes()).hexdigest()})
    (OUT/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
    failures=[c for c in CHECKS if not c['passed']]
    print(json.dumps({'checks':len(CHECKS),'passed':len(CHECKS)-len(failures),'failures':failures},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
