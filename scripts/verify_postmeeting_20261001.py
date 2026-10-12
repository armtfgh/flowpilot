"""Package, evidence and layout checks for the postmeeting Word revision."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import json
import re

import fitz
from lxml import etree as E
from PIL import Image, ImageDraw
from revise_manuscript_cases_20260922 import Package, W, NS, text, expanded_cites
from overhaul_submission_20260930 import with_cites
from verify_overhaul_20260930 import labeled

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manuscript'
OUT = BASE/'october1_postmeeting_work'
CHECKS=[]


def check(name, value, detail=None):
    CHECKS.append({'check':name,'passed':bool(value),'detail':detail})


def media(pkg):
    rels={r.get('Id'):r.get('Target') for r in pkg.rels}
    hashes=[]
    for p in pkg.body:
        for rid in p.xpath('.//a:blip/@r:embed',namespaces=NS):
            target=rels[rid]
            part=target.lstrip('/') if target.startswith('/') else 'word/'+target
            hashes.append(sha256(pkg.parts[part]).hexdigest())
    return hashes


def main():
    main=Package(BASE/'FlowPilot_manuscript_october1_postmeeting.docx')
    esi=Package(BASE/'FlowPilot_ESI_october1_postmeeting.docx')
    maps=json.loads((OUT/'numbering_map.json').read_text())
    for name,pkg in [('manuscript',main),('ESI',esi)]:
        source=Package(BASE/f'FlowPilot_{name}__20261001.docx')
        for part in ['word/styles.xml','word/fontTable.xml','word/numbering.xml']:
            check(f'{name}: {part} preserved',source.parts.get(part)==pkg.parts.get(part))
        check(f'{name}: original page setup retained',E.tostring(source.body[-1])==E.tostring(pkg.body[-1]))
        ids=[r.get('Id') for r in pkg.rels]
        check(f'{name}: unique image/document relationship IDs',len(ids)==len(set(ids)))
        check(f'{name}: front matter text unchanged',[text(n) for n in list(source.body)[:5]]==[text(n) for n in list(pkg.body)[:5]])
        check(f'{name}: clean reading copy has no revision highlighting',not pkg.doc.findall('.//'+W+'highlight'))
        marked=Package(OUT/'marked'/f'FlowPilot_{name}_october1_postmeeting_marked.docx')
        check(f'{name}: marked copy has identical wording',text(pkg.doc)==text(marked.doc))
        check(f'{name}: marked copy highlights revisions',bool(marked.doc.findall('.//'+W+'highlight')))
        for k,v in pkg.parts.items():
            if k.endswith('.xml') or k.endswith('.rels'): E.fromstring(v)
        check(f'{name}: all XML parts parse',True)
        if name=='manuscript':
            check('All six main-text figure images unchanged and in original order',media(source)==media(pkg))
            check('All six main-text drawings retain sizing/position/crop',[E.tostring(n) for n in source.doc.iter(W+'drawing')]==[E.tostring(n) for n in pkg.doc.iter(W+'drawing')])
        else:
            check('Accepted inventory GUI image preserved',sha256(source.parts['word/media/revision_54_figureS13_inventory_forms.png']).hexdigest() in media(pkg))
            # Experimental section remains unchanged apart from reference numbers.
            for i in [331,343,365]:
                old=list(source.body)[i]
                check(f'Experimental table source block {i} retains all cells',text(old) in [text(t) for t in pkg.body.findall(W+'tbl')])
            for x in json.loads((OUT/'ESI_source.json').read_text()):
                if x['body']>=321:
                    for img in x['images']:
                        check('Experimental image retained: '+Path(img).name,sha256((ROOT/img).read_bytes()).hexdigest() in media(pkg))
    for path,digest in json.loads((OUT/'source_manifest.json').read_text()).items():
        check('Source not overwritten: '+path,sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    definitions={}
    for p in esi.body:
        match=re.match(r'^(Figure|Table) S(\d+)\.',text(p))
        if match:
            key=(match[1],int(match[2]))
            check('Unique caption '+match[0],key not in definitions)
            definitions[key]=text(p)
            off=0; bad=[]
            for r in p.iter(W+'r'):
                if off>=match.end() and text(r).strip() and r.xpath('./w:rPr/w:b[not(@w:val="0")]',namespaces=NS):bad.append(text(r))
                off+=len(text(r))
            check(match[0]+' regular caption body',not bad,bad)
    for kind in ['Figure','Table']:
        nums=[n for k,n in definitions if k==kind]
        check(kind+' captions consecutive',nums==list(range(1,len(nums)+1)),nums)
    citations=[]
    for name,pkg in [('main',main),('ESI',esi)]:
        seen={'Figure':[],'Table':[]}
        bad=[]
        for i,p in enumerate(pkg.body):
            if p.tag==W+'sdt':continue
            for kind,n in labeled(text(p)):
                if (kind,n) not in definitions:bad.append([i,kind,n])
                if n not in seen[kind]:
                    seen[kind].append(n)
                    citations.append({'document':name,'reference':f'{kind} S{n}','body_index':i,'first_citation':text(p),'caption':definitions.get((kind,n),'MISSING')})
        check(name+': figure/table cross-references resolve',not bad,bad)
        if name=='main':
            for kind in seen:
                wanted=list(range(1,len(maps[kind])+1))
                check('Main text cites all '+kind+'s in order',seen[kind]==wanted,seen[kind])
    with (OUT/'citation_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(citations[0]));w.writeheader();w.writerows(citations)
    esi_value=text(esi.doc)
    response_evidence=json.loads((OUT/'response_table_source_audit.json').read_text())
    for case,answers in response_evidence.items():
        for row in answers:
            check(f'{case} set {row["set"]} {row["question_id"]} answer retained in table',row['answer'] in esi_value)
    check('Old standalone rule-base subsection removed','Engineering rule-base construction and use' not in esi_value)
    check('Duplicated retrieval schematic subsection removed','Plan-aware retrieval and tier transitions' not in esi_value)
    check('Correct three-panel rule image embedded',sha256((OUT/'figures/Figure_S4_rules_three_panels.png').read_bytes()).hexdigest() in media(esi))
    check('Actual annotated GUI image embedded',sha256((OUT/'gui/front_page_annotated.png').read_bytes()).hexdigest() in media(esi))
    for path in sorted((OUT/'figures').glob('*proposed_topology.png')):
        check('Restored '+path.name,sha256(path.read_bytes()).hexdigest() in media(esi))
    check('Old preprint Figure 6 preserved exactly',sha256((OUT/'figures/archived_preprint_figure6_dry_run.png').read_bytes()).hexdigest() in media(esi))
    refnums=[]
    for p in main.body:
        if text(p)=='REFERENCES':break
        for c in re.findall(r'\[\[(.*?)\]\]',with_cites(p)):
            for n in expanded_cites(c):
                if n not in refnums:refnums.append(n)
    check('Main literature citation sequence 1-61 retained',refnums==list(range(1,62)),refnums)
    # Check that stage keys reproduce V / incoming total flow on the stated basis.
    numerical=[]
    for path in sorted((OUT/'raw').glob('figure*/stage_parameters.csv')):
        for row in csv.DictReader(path.open()):
            q=float(row['liquid_flow_mL_min'])+float(row['gas_inlet_STP_mL_min'])
            calc=float(row['volume_mL'])/q
            reported=float(row['nominal_inlet_residence_min'])
            numerical.append({'case':row['case'],'stage':row['stage'],'calculated_min':calc,'reported_min':reported,'basis':'inlet/STP index' if float(row['gas_inlet_STP_mL_min']) else 'liquid'})
            check('Stage key closes: '+row['case']+' R'+row['stage'],abs(calc-reported)<.01)
    (OUT/'restored_stage_calculation_audit.json').write_text(json.dumps(numerical,indent=2))
    # Page images allow visual review; detected text overflow is checked separately.
    render=OUT/'rendered'
    for name in ['manuscript','ESI']:
        pdf=fitz.open(render/f'FlowPilot_{name}_october1_postmeeting.pdf')
        info=[]; overflow=[]; blank=[]
        thumbs=[]
        for i,page in enumerate(pdf):
            spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
            for s in spans:
                x0,y0,x1,y1=s['bbox']
                if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:overflow.append([i+1,s['text'],s['bbox']])
            if len(page.get_text().strip())<8 and not page.get_images():blank.append(i+1)
            info.append({'page':i+1,'start':page.get_text()[:180],'images':len(page.get_images())})
            pix=page.get_pixmap(matrix=fitz.Matrix(.4,.4))
            im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            tile=Image.new('RGB',(260,350),'#dddddd');tile.paste(im,(8,20));ImageDraw.Draw(tile).text((8,4),str(i+1),fill='black');thumbs.append(tile)
        for start in range(0,len(thumbs),20):
            group=thumbs[start:start+20]; sheet=Image.new('RGB',(260*4,350*((len(group)+3)//4)),'white')
            for j,im in enumerate(group):sheet.paste(im,((j%4)*260,(j//4)*350))
            sheet.save(render/f'{name}_contact_{start+1:02}.png')
        check(name+': no text outside page bounds',not overflow,overflow)
        check(name+': no blank pages',not blank,blank)
        (OUT/f'{name}_page_index.json').write_text(json.dumps(info,indent=2))
    (OUT/'verification.json').write_text(json.dumps(CHECKS,indent=2,ensure_ascii=False))
    failed=[c for c in CHECKS if not c['passed']]
    print(json.dumps({'checks':len(CHECKS),'failed':failed},indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
