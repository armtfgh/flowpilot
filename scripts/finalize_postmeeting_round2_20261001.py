"""Render, inspect structural integrity and independently recheck reported numbers."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import re
import statistics
import subprocess

import fitz
from lxml import etree as E
from PIL import Image, ImageDraw
from revise_manuscript_cases_20260922 import Package, W, NS, text, expanded_cites
from overhaul_submission_20260930 import with_cites
from refresh_revision_toc_20260922 import refresh, normalized
from verify_overhaul_20260930 import labeled
from verify_postmeeting_20261001 import media
import verify_narrative_20261001 as evidence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manuscript'
OUT = BASE/'october1_postmeeting_round2_work'
CHECKS = []


def check(name, passed, detail=None):
    CHECKS.append({'check':name, 'passed':bool(passed), 'detail':detail})


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def render(paths):
    subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/flowpilot-round2-review',
                    '--headless','--convert-to','pdf','--outdir',str(OUT/'rendered'),
                    *map(str,paths)],check=True,timeout=240)


def update_contents(esi):
    last = None
    audits = []
    for _ in range(4):
        with fitz.open(OUT/'rendered'/esi.with_suffix('.pdf').name) as pdf:
            audit = refresh(esi,pdf)
            refresh(OUT/'marked'/esi.with_stem(esi.stem+'_marked').name,pdf)
        audits.append(audit)
        render([esi])
        if audit == last:break
        last = audit
    else:raise RuntimeError('Contents did not stabilize')
    (OUT/'toc_audit.json').write_text(json.dumps(audits,indent=2))
    return audit


def package_checks():
    pkgs = {}
    for name in ['manuscript','ESI']:
        before = Package(BASE/f'FlowPilot_{name}_october1_postmeeting.docx')
        after = Package(BASE/f'FlowPilot_{name}_october1_postmeeting_round2.docx')
        marked = Package(OUT/'marked'/f'FlowPilot_{name}_october1_postmeeting_round2_marked.docx')
        pkgs[name] = after
        for part in ['word/styles.xml','word/fontTable.xml','word/numbering.xml']:
            check(name+': unchanged '+part,before.parts.get(part)==after.parts.get(part))
        check(name+': unchanged page setup',E.tostring(before.body[-1])==E.tostring(after.body[-1]))
        check(name+': unchanged displayed image bytes and order',media(before)==media(after),len(media(after)))
        check(name+': unchanged figure dimensions/position/crop',
              [E.tostring(x) for x in before.doc.iter(W+'drawing')]==[E.tostring(x) for x in after.doc.iter(W+'drawing')])
        check(name+': clean and marked wording identical',text(after.doc)==text(marked.doc))
        check(name+': no highlighting in clean copy',not after.doc.findall('.//'+W+'highlight'))
        highlighted=marked.doc.xpath('.//w:r[w:rPr/w:highlight[@w:val="yellow"]]',namespaces=NS)
        check(name+': changes highlighted in review copy',bool(highlighted))
        badfonts=[text(r) for r in highlighted if text(r).strip() and
                  (not r.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]',namespaces=NS) or
                   not r.xpath('./w:rPr/w:sz[@w:val="22"]',namespaces=NS))]
        check(name+': revised text Times New Roman 11',not badfonts,badfonts)
        check(name+': unchanged OfficeMath equations',
              [E.tostring(x) for x in before.doc.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')]==
              [E.tostring(x) for x in after.doc.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')])
        check(name+': authors unchanged',[text(n) for n in list(before.body)[:5]]==[text(n) for n in list(after.body)[:5]])
        ids=[r.get('Id') for r in after.rels]
        check(name+': no duplicate relationship IDs',len(ids)==len(set(ids)))
        for key,part in after.parts.items():
            if key.endswith(('.xml','.rels')): E.fromstring(part)
        check(name+': valid package XML',True)
        start=next(i for i,n in enumerate(before.body) if text(n) in {'REFERENCES','References'})
        expected=[text(n) for n in list(before.body)[start+1:]]
        start2=next(i for i,n in enumerate(after.body) if text(n) in {'REFERENCES','References'})
        check(name+': bibliography text unchanged',expected==[text(n) for n in list(after.body)[start2+1:]])
        if name=='ESI':
            oldrows=json.loads((OUT/'ESI_source.json').read_text())
            # Preserve verbatim prompts, responses and experimental records, except
            # the explicitly audited correction of a bibliography range.
            new_paragraphs={with_cites(p) for p in after.doc.iter(W+'p')}
            for i in [102,105,138,140,158,160,178,180,265,266,295]:
                if oldrows[i]['kind']=='p':
                    check(f'ESI source paragraph {i} retained',oldrows[i]['text'] in new_paragraphs)
            oldtables=[n for n in before.body if n.tag==W+'tbl']
            newtables=[n for n in after.body if n.tag==W+'tbl']
            check('ESI table count unchanged',len(oldtables)==len(newtables)==20)
            for i in range(14,20):
                check(f'Table S{i+1}: response/experimental cells unchanged',text(oldtables[i])==text(newtables[i]))
    for path,digest in json.loads((OUT/'source_manifest.json').read_text()).items():
        check('Source not overwritten: '+path,sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    return pkgs


def citations(pkgs):
    definitions={}
    for p in pkgs['ESI'].body:
        m=re.match(r'^(Figure|Table) S(\d+)\.',text(p))
        if not m:continue
        key=(m[1],int(m[2]))
        check('Unique '+m[0],key not in definitions)
        definitions[key]=text(p)
        offset=0;bad=[]
        for r in p.iter(W+'r'):
            if offset>=m.end() and text(r).strip() and r.xpath('./w:rPr/w:b[not(@w:val="0")]',namespaces=NS):bad.append(text(r))
            offset+=len(text(r))
        check(m[0]+' caption body not bold',not bad,bad)
    audit=[]
    for name,pkg in pkgs.items():
        seen=defaultdict(list);bad=[]
        for i,node in enumerate(pkg.body):
            if node.tag==W+'sdt':continue
            for kind,num in labeled(text(node)):
                if (kind,num) not in definitions:bad.append([kind,num,i])
                if num not in seen[kind]:
                    seen[kind].append(num)
                    audit.append({'document':name,'reference':f'{kind} S{num}',
                                  'body_index':i,'first_citation':text(node),'caption':definitions.get((kind,num))})
        check(name+': all figure/table citations resolve',not bad,bad)
        if name=='manuscript':
            for kind,count in [('Figure',25),('Table',20)]:
                check(f'Main first citation order: {kind} S1-S{count}',seen[kind]==list(range(1,count+1)),seen[kind])
    with (OUT/'citation_audit.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(audit[0]));wr.writeheader();wr.writerows(audit)
    order=[]
    for node in pkgs['manuscript'].body:
        if text(node)=='REFERENCES':break
        for group in re.findall(r'\[\[(.*?)\]\]',with_cites(node)):
            for number in expanded_cites(group):
                if number not in order:order.append(number)
    check('Main references 1-61 cited in sequence',order==list(range(1,62)),order)
    esi=text(pkgs['ESI'].doc)
    check('Amide characterization points to correct references 12-13','[12-13]' in esi and '[11-12]' not in esi)
    for qid in ['Q-BATCH-001','Q-OBJ-001','Q-CHEM-001','Q-HIST-001','Q-INV-001','Q-CONSTR-001','Q-HYP-001','Q-PREF-001',
                'Q-GAS-001','Q-GAS-002','Q-GAS-003','Q-PHOTO-001','Q-MULTI-001']:
        t=list(pkgs['ESI'].body.findall(W+'tbl'))[2]
        check('Table S3 contains '+qid,qid in text(t))


def evidence_checks():
    evidence.OUT=OUT
    evidence.CHECKS=[]
    evidence.evidence_checks()
    CHECKS.extend(evidence.CHECKS)
    data=ROOT/'deliverables/manuscript_benchmark_visualizations_20260825/source_data_revised'
    register=rows(data/'fig08_judge_critical_evidence.csv')
    heatmap=rows(data/'fig08a_campaign_error_map.csv')
    check('Critical register contains exactly 102 judge flags',len(register)==102)
    check('No UO-01/UO-14 critical flags hidden by map',not any(r['criterion_id'] in ['UO-01','UO-14'] for r in register))
    check('Map has 990 cells for 90 outcomes and 11 criteria',len(heatmap)==990 and len({r['candidate_id'] for r in heatmap})==90 and len({r['criterion_id'] for r in heatmap})==11)
    check('Map preserves all 102 reported flags',sum(int(r['judge_flags']) for r in heatmap)==102)
    module=rows(ROOT/'deliverables/newgen_2_0_module_attribution_qwen38_screen3_matched_v1_20260821/tables/module_outcomes.csv')
    by_condition=defaultdict(list)
    for r in module:by_condition[r['condition']].append(float(r['consensus_score_0_1']))
    check('Module outcomes: 15 conditions x 3 cases',len(module)==45 and len(by_condition)==15 and all(len(v)==3 for v in by_condition.values()))
    summary=[]
    for k,v in by_condition.items():summary.append({'condition':k,'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'n_cases':len(v)})
    (OUT/'module_recalculation.json').write_text(json.dumps(summary,indent=2))
    check('CuAAC contradictory nominal volumes not silently reconciled',abs(math.pi*(4/2)**2*60/1000-.7539822369)<1e-9)
    manifest=json.loads((OUT/'frozen_benchmark_inputs/manifest.json').read_text())
    for row in manifest['exports']:
        check('Exact input export hash: '+row['file'],sha256((OUT/'frozen_benchmark_inputs'/row['file']).read_bytes()).hexdigest()==row['sha256'])


def layout(toc):
    for name in ['manuscript','ESI']:
        filename=f'FlowPilot_{name}_october1_postmeeting_round2'
        pdf=fitz.open(OUT/'rendered'/f'{filename}.pdf')
        marked=fitz.open(OUT/'rendered'/f'{filename}_marked.pdf')
        check(name+': clean/marked pagination identical',len(pdf)==len(marked),len(pdf))
        check(name+': clean/marked rendered text identical',
              [normalized(p.get_text()) for p in pdf]==[normalized(p.get_text()) for p in marked])
        overflow=[];blank=[];detached=[];thumbs=[];index=[]
        for i,p in enumerate(pdf):
            contents=p.get_text()
            spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
            for s in spans:
                x0,y0,x1,y1=s['bbox']
                if x0 < -1 or y0 < -1 or x1 > p.rect.width+1 or y1 > p.rect.height+1:overflow.append([i+1,s['text']])
            if len(contents.strip())<8 and not p.get_image_info():blank.append(i+1)
            if re.search(r'^Figure S?\d+\.',contents,re.M) and not p.get_image_info():detached.append(i+1)
            index.append({'page':i+1,'start':contents[:200],'figures':re.findall(r'Figure S?\d+\.',contents),
                          'tables':re.findall(r'Table S\d+\.',contents)})
            pix=p.get_pixmap(matrix=fitz.Matrix(.4,.4))
            im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            tile=Image.new('RGB',(260,350),'#e4e4e4');tile.paste(im,(8,20));ImageDraw.Draw(tile).text((8,4),f'{name} {i+1}',fill='black');thumbs.append(tile)
            if any(term in contents for term in ['Table S3.', 'Table S9.', 'Table S12.', 'Frozen inventory summary.',
                                                'Figure 4.', 'Figure 5.', 'Figure 6.', '5.3', '5.4']):
                p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/'rendered'/f'{name}_page_{i+1:02}.png')
        for start in range(0,len(thumbs),20):
            batch=thumbs[start:start+20];sheet=Image.new('RGB',(1040,350*math.ceil(len(batch)/4)),'white')
            for j,im in enumerate(batch):sheet.paste(im,((j%4)*260,(j//4)*350))
            sheet.save(OUT/'rendered'/f'{name}_contact_{start+1:02}.png')
        check(name+': no text outside pages',not overflow,overflow)
        check(name+': no blank pages',not blank,blank)
        check(name+': figure captions share a page with an image',not detached,detached)
        (OUT/f'{name}_page_index.json').write_text(json.dumps(index,indent=2))
        if name=='ESI':
            check('Every TOC anchor matches its rendered page',all(normalized(r['heading']) in normalized(pdf[r['page']-1].get_text()) for r in toc))
            check('Engineering subsections 5.1-5.6 in contents',[r['section'] for r in toc if r['section'].startswith('5.')]==['5.1','5.2','5.3','5.4','5.5','5.6'])


def main():
    mainpath=BASE/'FlowPilot_manuscript_october1_postmeeting_round2.docx'
    esipath=BASE/'FlowPilot_ESI_october1_postmeeting_round2.docx'
    render([mainpath,esipath])
    toc=update_contents(esipath)
    render(sorted((OUT/'marked').glob('*.docx')))
    pkgs=package_checks()
    citations(pkgs)
    evidence_checks()
    layout(toc)
    report={'passed':sum(c['passed'] for c in CHECKS),'total':len(CHECKS),'checks':CHECKS}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    failed=[c for c in CHECKS if not c['passed']]
    print(json.dumps({'passed':report['passed'],'total':len(CHECKS),'failed':failed},indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
