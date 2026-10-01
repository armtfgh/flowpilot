"""Independent package, arithmetic, citation and rendered-layout checks."""
import argparse
import csv
import json
import re
import math
import statistics
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree as E
import fitz
from PIL import Image, ImageDraw, ImageFont

import revise_prior_work_20260928 as prior
from refresh_revision_toc_20260922 import refresh, normalized
from verify_introduction_20260923 import no_highlight_structure

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript/Submission';OUT=BASE/'revision_20260929'
W=prior.W;NS=dict(prior.NS);NS.update(a='http://schemas.openxmlformats.org/drawingml/2006/main',wp='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing')
checks=[]
def check(name,ok,detail=None):checks.append({'check':name,'passed':bool(ok),'detail':detail})
def read(path):
    with ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
    return parts,E.fromstring(parts['word/document.xml'])
def text(n):return prior.text(n)
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))

def update_toc():
    results={}
    for suffix in ('','_marked'):
        path=BASE/f'esi_submission_revised_20260929{suffix}.docx'
        pdf=fitz.open(OUT/'rendered/esi_submission_revised_20260929.pdf')
        results[suffix or 'clean']=refresh(path,pdf)
        parts,doc=read(path);prior.bold_references(doc);prior.intro_tools.write_package(parts,doc,path)
    (OUT/'contents_page_audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Updated contents; rerender both documents before validation.')

def crossrefs(main,esi):
    definitions={}
    for tag,doc in [('main',main),('esi',esi)]:
        body=doc.find(W+'body')
        for i,p in enumerate(body):
            if p.tag!=W+'p':continue
            value=text(p)
            m=re.match(r'^(Figure|Table) (S?\d+)([a-z])?(?:\.| \(continued\))',value)
            if m:definitions.setdefault((m[1],m[2]),[]).append({'document':tag,'index':i,'caption':value})
    check('Main figure captions 1-6',set(n for k,n in definitions if k=='Figure' and not n.startswith('S'))=={str(i) for i in range(1,7)})
    check('ESI figure captions S1-S31',set(n for k,n in definitions if k=='Figure' and n.startswith('S'))=={f'S{i}' for i in range(1,32)})
    check('ESI table captions S1-S39',set(n for k,n in definitions if k=='Table' and n.startswith('S'))=={f'S{i}' for i in range(1,40)})
    audit=[];missing=[];maincited=set()
    for tag,doc in [('main',main),('esi',esi)]:
        for i,p in enumerate(doc.find(W+'body')):
            if p.tag!=W+'p':continue
            value=text(p)
            if re.match(r'^(Figure|Table) S?\d',value):continue
            for match in prior.LABEL.finditer(value):
                typ='Table' if match[0].lower().startswith('table') else 'Figure'
                tokens=re.findall(r'S?\d+',match[0]);nums=[]
                for token in tokens:
                    if token.startswith('S'):prefix='S'
                    else:prefix='S' if nums and nums[-1].startswith('S') else ''
                    nums.append(token if token.startswith('S') else prefix+token)
                if len(nums)==2 and re.search(r'\d\s*[-–]\s*S?\d',match[0]):
                    pre='S' if nums[0].startswith('S') else ''
                    nums=[pre+str(j) for j in range(int(nums[0].lstrip('S')),int(nums[1].lstrip('S'))+1)]
                for n in nums:
                    key=(typ,n);found=key in definitions
                    audit.append({'document':tag,'paragraph':i,'citation':match[0],'resolved_label':typ+' '+n,'found':found,'caption':definitions.get(key,[{}])[0].get('caption',''),'context':value})
                    if not found:missing.append([tag,match[0]])
                    if tag=='main' and n.startswith('S'):maincited.add(key)
    check('Every parsed figure/table citation resolves',not missing,missing)
    uncited=[f'{k} {n}' for k,n in definitions if n.startswith('S') and (k,n) not in maincited]
    check('All ESI figures/tables have a main-text citation',not uncited,uncited)
    with (OUT/'cross_reference_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    (OUT/'caption_inventory.json').write_text(json.dumps({f'{k} {n}':v for (k,n),v in definitions.items()},indent=2)+'\n')

def fonts_tables(doc,tag):
    errors=[];refs=[]
    for p in doc.iter(W+'p'):
        flags=[];value=''
        for t in prior.paragraph_text_nodes(p):
            s=t.text or '';value+=s;r=t.getparent();bold=r.find(W+'rPr/'+W+'b')
            flags.extend([bold is not None and bold.get(W+'val','1') not in ('0','false','off')]*len(s))
        for m in prior.LABEL.finditer(value):
            if not all(flags[m.start():m.end()]):refs.append(m[0])
    for i,t in enumerate(doc.iter(W+'tbl')):
        for j,row in enumerate(t.findall(W+'tr')):
            for c in row.findall(W+'tc'):
                fills=c.xpath('./w:tcPr/w:shd/@w:fill',namespaces=NS)
                if fills!=['E6E6E6' if j==0 else 'FFFFFF']:errors.append([i,j,'shading',fills])
                for r in c.iter(W+'r'):
                    if not text(r).strip():continue
                    if not r.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]',namespaces=NS) or not r.xpath('./w:rPr/w:sz[@w:val="22"]',namespaces=NS):errors.append([i,j,'font',text(r)])
    check(tag+' tables use uniform gray headers and TNR11',not errors,errors)
    check(tag+' figure/table references are bold',not refs,refs)

def rendered_pdf(path,tag):
    d=fitz.open(path);outside=[];blank=[];thumbs=[];locations={}
    for i,p in enumerate(d):
        for b in p.get_text('blocks'):
            if len(b)>6 and b[6]!=0:continue
            if b[0]<-1 or b[1]<-1 or b[2]>p.rect.width+1 or b[3]>p.rect.height+1:outside.append([i+1,b[:4],str(b[4])[:120]])
        if len(p.get_text().strip())<8 and not p.get_image_info():blank.append(i+1)
        for label in re.findall(r'(?:Figure|Table) S?\d+[a-z]?',p.get_text()):locations.setdefault(label,[]).append(i+1)
        pix=p.get_pixmap(matrix=fitz.Matrix(.55,.55));im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        canvas=Image.new('RGB',(360,495),'#eeeeee');canvas.paste(im,((360-im.width)//2,23))
        ImageDraw.Draw(canvas).text((10,4),f'{tag} page {i+1}',fill='black');thumbs.append(canvas)
    dest=OUT/'visual_review';dest.mkdir(exist_ok=True)
    for start in range(0,len(thumbs),12):
        grid=Image.new('RGB',(1440,1485),'#999')
        for j,im in enumerate(thumbs[start:start+12]):grid.paste(im,(360*(j%4),495*(j//4)))
        grid.save(dest/f'{tag}_pages_{start+1:03d}_{min(start+12,len(d)):03d}.png')
    check(tag+' rendered text stays on page',not outside,outside)
    check(tag+' no empty rendered pages',not blank,blank)
    return {'pages':len(d),'locations':locations}

def main():
    originals=json.loads((OUT/'source_manifest.json').read_text())
    for tag,entry in originals.items():check(tag+' authoritative source unchanged',sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'])
    mainparts,main=read(BASE/'manuscript_submission_revised_20260929.docx')
    esiparts,esi=read(BASE/'esi_submission_revised_20260929.docx')
    for tag,parts,doc in [('main',mainparts,main),('esi',esiparts,esi)]:
        sourceparts,source=read(ROOT/originals[tag]['path'])
        check(tag+' styles and numbering preserved',all(parts.get(k)==sourceparts.get(k) for k in ['word/styles.xml','word/numbering.xml']))
        check(tag+' section geometry preserved',E.tostring(doc.find('.//'+W+'sectPr'))==E.tostring(source.find('.//'+W+'sectPr')))
        check(tag+' all original media retained',all(parts.get(k)==v for k,v in sourceparts.items() if k.startswith('word/media/')))
        markedparts,marked=read(BASE/f'{"manuscript" if tag=="main" else "esi"}_submission_revised_20260929_marked.docx')
        check(tag+' clean/marked have identical text',text(doc)==text(marked))
        check(tag+' marked revisions highlighted yellow',bool(marked.xpath('.//w:highlight[@w:val="yellow"]',namespaces=NS)))
        check(tag+' no placeholder author names or yield fields',not any(t in text(doc) for t in ['[Name to be added]','Dr. Myeong','XXX','YYY']))
        rels={r.get('Id'):r.get('Target') for r in E.fromstring(parts['word/_rels/document.xml.rels'])}
        missing=[rid for rid in doc.xpath('.//a:blip/@r:embed',namespaces=NS) if 'word/'+rels.get(rid,'NO') not in parts]
        check(tag+' all embedded image relationships resolve',not missing,missing)
        fonts_tables(doc,tag)
    check('Author lines identical',text(main.find(W+'body')[1])==text(esi.find(W+'body')[1]))
    for email in ('boyoungy.park@khu.ac.kr','gnahn@krict.re.kr'):check('Both documents include '+email,email in text(main) and email in text(esi))
    crossrefs(main,esi)
    # Independent arithmetic rather than checking only rounded prose.
    rows=csvrows(OUT/'source_data/khu_reported_experimental_results.csv')
    for row in rows:
        q=float(row['Q1_mL_min'])+float(row['Q2_mL_min'])+float(row['Qgas_reference_mL_min'])
        check(f"Figure {row['figure']} sets {row['sets']} stage 1 closure",math.isclose(float(row['V1_mL'])/float(row['Q1_mL_min']),float(row['t1_min']),rel_tol=1e-9))
        check(f"Figure {row['figure']} sets {row['sets']} stage 2 convention closure",math.isclose(float(row['V2_mL'])/q,float(row['t2_index_min']),rel_tol=1e-9))
    check('Figure 5 O2 equivalents at explicitly stated STP',math.isclose(.09/22.414/(.1*.02),2.0078,rel_tol=1e-4))
    for qa,qb in ((.16,.04),(.28,.07)):check(f'Amine stoichiometry QA={qa}',math.isclose(2.1*qb/(.5*qa),1.05,rel_tol=1e-12))
    check('Sulfoxide isolated mass/yield',round(42/(.2*223.29)*100)==94)
    check('Amide isolated mass/yield',round(227/270.29*100)==84)
    mr=csvrows(OUT/'source_data/tableS10_all15_module_conditions.csv')
    check('Table S10 source has 15 conditions / 45 outcomes',len(mr)==15 and sum(int(r['n_outcomes']) for r in mr)==45)
    score=csvrows(OUT/'source_data/figure4_scores.csv');flags=csvrows(OUT/'source_data/figure4_flags.csv')
    check('Figure 4 five-model cohort excludes GPT-5.4 generator',len({r['model'] for r in score})==5 and not any('5.4' in r['model'] for r in score))
    for arch,expected,total in [('One-shot',.794,98),('FlowPilot',.917,4)]:
        rs=[r for r in score if r['architecture']==arch];fs=[r for r in flags if r['architecture']==arch]
        check(arch+' reported mean matches source',round(sum(float(r['mean']) for r in rs)/len(rs),3)==expected)
        check(arch+' critical-flag sum matches source',round(sum(float(r['mean'])*3 for r in fs))==total)
    outcomes_path=ROOT/'deliverables/manuscript_benchmark_visualizations_20260825/source_data_revised/fig08_campaign_error_summary.csv'
    outcomes=csvrows(outcomes_path)
    check('Underlying cohort has 90 records',len(outcomes)==90 and len({r['candidate_id'] for r in outcomes})==90)
    (OUT/'source_data/retained_90_campaign_outcomes.csv').write_bytes(outcomes_path.read_bytes())
    for r in score:
        os=[o for o in outcomes if o['model']==r['model'] and o['architecture']==r['architecture']]
        means=[statistics.mean(float(o['benchmark_score']) for o in os if o['repeat_id']==repeat) for repeat in sorted({o['repeat_id'] for o in os})]
        check(r['model']+' '+r['architecture']+' mean/SD recompute from nine campaigns',len(os)==9 and math.isclose(statistics.mean(means),float(r['mean']),abs_tol=1e-7) and math.isclose(statistics.stdev(means),float(r['sample_sd']),abs_tol=1e-7))
    for arch,n,total in [('One-shot',18,98),('FlowPilot',43,4)]:
        os=[o for o in outcomes if o['architecture']==arch]
        check(arch+' raw flags and flag-free counts recompute',sum(int(o['judge_flag_count']) for o in os)==total and sum(int(o['judge_flag_count'])==0 for o in os)==n)
    renders={}
    for tag,stem in [('main','manuscript'),('esi','esi')]:renders[tag]=rendered_pdf(OUT/f'rendered/{stem}_submission_revised_20260929.pdf',tag)
    (OUT/'rendered_page_map.json').write_text(json.dumps(renders,indent=2)+'\n')
    (OUT/'verification.json').write_text(json.dumps({'passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks},indent=2)+'\n')
    print(json.dumps({'passed':sum(c['passed'] for c in checks),'total':len(checks),'failed':[c for c in checks if not c['passed']]},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-toc',action='store_true');args=parser.parse_args()
    if args.refresh_toc:update_toc()
    else:main()
