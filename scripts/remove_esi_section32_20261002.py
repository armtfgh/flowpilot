"""Remove the entire retrieval subsection, not relocate or summarize it in ESI."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import posixpath
import re
import subprocess

import fitz
from lxml import etree as E
from revise_manuscript_cases_20260922 import Package, W, NS, text, expanded_cites
from overhaul_submission_20260930 import with_cites
from refresh_revision_toc_20260922 import refresh, normalized
from verify_overhaul_20260930 import labeled
from verify_postmeeting_20261001 import media
from revise_layout_discussion_20260928 import order_properties

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manuscript'
OUT = BASE/'october2_section32_removal'
REF = re.compile(r'\b(Figures?|Tables?) S(\d+)([a-e]?)(?:[\u2013\u2014-]S?(\d+)([a-e]?))?')
CHECKS = []
CHANGES = []


def check(name, value, detail=None):
    CHECKS.append({'check':name,'passed':bool(value),'detail':detail})


def patch_text(p, pattern, substitute):
    """Change only matched characters, preserving all unaffected runs and math."""
    nodes=p.xpath('.//w:t',namespaces=NS)
    value=''.join(n.text or '' for n in nodes)
    spans=[];offset=0
    for n in nodes:
        end=offset+len(n.text or '')
        spans.append((n,offset,end));offset=end
    changed=False
    for match in reversed(list(pattern.finditer(value))):
        new=substitute(match)
        if new==match[0]:continue
        affected=[(n,a,b) for n,a,b in spans if a < match.end() and b > match.start()]
        for i,(n,a,b) in enumerate(affected):
            old=n.text or ''
            left=max(0,match.start()-a);right=min(b-a,match.end()-a)
            n.text=old[:left]+(new if i==0 else '')+old[right:]
            n.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        changed=True
    return changed


def revise(pkg,p,value):
    CHANGES.append({'document':pkg.source.name,'before':with_cites(p),'after':value})
    pkg.revise(p,value)


def remove_section(pkg):
    nodes=list(pkg.body)
    start=next(i for i,n in enumerate(nodes) if text(n)=='Retrieval settings, benchmark and source-exclusion controls')
    end=next(i for i in range(start+1,len(nodes)) if nodes[i].xpath('./w:pPr/w:pStyle[@w:val="Heading1"]',namespaces=NS))
    removed=nodes[start:end]
    check('Removal includes the complete subsection and exactly two tables',sum(n.tag==W+'tbl' for n in removed)==2)
    check('Removal includes Figure S5 and Tables S5-S6',all(any(text(n).startswith(label) for n in removed) for label in ['Figure S5.','Table S5.','Table S6.']))
    rids=[rid for n in removed for rid in n.xpath('.//a:blip/@r:embed',namespaces=NS)]
    check('Removal includes exactly one displayed figure',len(rids)==1,rids)
    archive=E.Element('removed_section')
    for n in removed:archive.append(deepcopy(n));pkg.body.remove(n)
    next_heading=nodes[end]
    pp=next_heading.find(W+'pPr')
    if pp.find(W+'pageBreakBefore') is None:E.SubElement(pp,W+'pageBreakBefore')
    (OUT/'removed_section.xml').write_bytes(E.tostring(archive,xml_declaration=True,encoding='UTF-8'))
    (OUT/'removed_section.txt').write_text('\n\n'.join(text(n) for n in removed))
    deleted_parts=[]
    for rid in rids:
        if pkg.doc.xpath('.//*[@r:embed=$rid or @r:link=$rid]',namespaces=NS,rid=rid):continue
        rel=next(r for r in pkg.rels if r.get('Id')==rid)
        target=rel.get('Target')
        pkg.rels.remove(rel)
        if any(r.get('Target')==target for r in pkg.rels):continue
        part=target.lstrip('/') if target.startswith('/') else posixpath.normpath('word/'+target)
        pkg.parts.pop(part)
        deleted_parts.append(part)
    CHANGES.append({'document':pkg.source.name,'removed_body_nodes':len(removed),
                    'removed_tables':['S5','S6'],'removed_figure':'S5','removed_image_parts':deleted_parts,
                    'policy':'Nothing from this subsection is reinserted elsewhere in the ESI.'})
    title=next(n for n in pkg.body if text(n).strip()=='Literature corpus, rule base, and retrieval')
    revise(pkg,title,'Literature corpus and engineering rule base')
    return removed,rids


def main_references(pkg):
    replacements={
        '(Figure 3; Table S5)':'(Figure 3)',
        'Figure 3d illustrates the ranking change; Figure S5 and Tables S5\u2013S6 give the analyses and their scope.':'Figure 3d illustrates the ranking change.',
        'described in Methods and ESI Section 3.2.':'described in Methods.',
        ' The operational field components and weights are listed in Table S5.':'',
        ' Its 56 queries and script-specific similarity functions are documented in ESI Section 3.2 and Table S6.':'',
    }
    counts={key:0 for key in replacements}
    for p in pkg.body.findall(W+'p'):
        old=with_cites(p);new=old
        for a,b in replacements.items():
            counts[a]+=new.count(a);new=new.replace(a,b)
        if new!=old:revise(pkg,p,new)
    check('All five manuscript pointers to deleted content corrected',all(n==1 for n in counts.values()),counts)


def renumber(pkgs):
    maps={'Figure':{},'Table':{}}
    for n in pkgs['ESI'].body:
        m=re.match(r'^(Figure|Table) S(\d+)\.',text(n))
        if m:maps[m[1]][int(m[2])]=len(maps[m[1]])+1
    def replace(m):
        prefix,start,suffix,end,end_suffix=m.groups()
        mapping=maps['Figure' if prefix.startswith('Figure') else 'Table']
        numbers=list(range(int(start),int(end)+1)) if end else [int(start)]
        assert all(n in mapping for n in numbers),('Stale pointer before renumbering',m[0])
        if end:
            return prefix+' S'+str(mapping[int(start)])+suffix+'\u2013S'+str(mapping[int(end)])+(end_suffix or '')
        return prefix+' S'+str(mapping[int(start)])+suffix
    for pkg in pkgs.values():
        for p in pkg.doc.iter(W+'p'):
            if p.xpath('ancestor::w:sdt',namespaces=NS):continue
            if patch_text(p,REF,replace):pkg.changed.append(p)
    (OUT/'numbering_map.json').write_text(json.dumps(maps,indent=2))
    return maps


def render(paths):
    subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/flowpilot-section32-removal',
                    '--headless','--convert-to','pdf','--outdir',str(OUT/'rendered'),*map(str,paths)],
                   check=True,timeout=240)


def verify(pkgs,originals,removed,hashes,toc):
    captions={}
    for p in pkgs['ESI'].body:
        m=re.match(r'^(Figure|Table) S(\d+)\.',text(p))
        if m:
            key=(m[1],int(m[2]))
            check('Unique caption '+m[0],key not in captions)
            captions[key]=text(p)
    for kind,count in [('Figure',24),('Table',18)]:
        check(kind+' captions consecutive', [n for k,n in captions if k==kind]==list(range(1,count+1)))
    audit=[]
    for name,pkg in pkgs.items():
        src=originals[name]
        for part in ['word/styles.xml','word/fontTable.xml','word/numbering.xml']:
            check(name+': preserved '+part,pkg.parts[part]==src.parts[part])
        check(name+': page setup unchanged',E.tostring(pkg.body[-1])==E.tostring(src.body[-1]))
        check(name+': no Section 3.2 pointer/heading remains','Section 3.2' not in text(pkg.doc) and 'Retrieval settings, benchmark and source-exclusion controls' not in text(pkg.doc))
        expected=src
        if name=='ESI':
            expected=Package(src.source)
            nn=list(expected.body)
            start=next(i for i,n in enumerate(nn) if text(n)=='Retrieval settings, benchmark and source-exclusion controls')
            end=next(i for i in range(start+1,len(nn)) if nn[i].xpath('./w:pPr/w:pStyle[@w:val="Heading1"]',namespaces=NS))
            for n in nn[start:end]:expected.body.remove(n)
        check(name+': all retained images and their order unchanged',media(pkg)==media(expected))
        check(name+': retained image sizes/crops unchanged',[E.tostring(n) for n in pkg.doc.iter(W+'drawing')]==[E.tostring(n) for n in expected.doc.iter(W+'drawing')])
        math='{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'
        check(name+': retained equations untouched',[E.tostring(n) for n in pkg.doc.iter(math)]==[E.tostring(n) for n in expected.doc.iter(math)])
        check(name+': author information unchanged',[text(n) for n in list(pkg.body)[:5]]==[text(n) for n in list(src.body)[:5]])
        seen={'Figure':[],'Table':[]};bad=[]
        for i,n in enumerate(pkg.body):
            if n.tag==W+'sdt':continue
            for key in labeled(text(n)):
                if key not in captions:bad.append([i,key])
                if key[1] not in seen[key[0]]:
                    seen[key[0]].append(key[1]);audit.append({'document':name,'reference':f'{key[0]} S{key[1]}','text':text(n),'caption':captions.get(key)})
        check(name+': references resolve',not bad,bad)
        if name=='manuscript':
            for kind,count in [('Figure',24),('Table',18)]:
                check(f'Main first-citation order {kind} S1-S{count}',seen[kind]==list(range(1,count+1)),seen[kind])
        path=OUT/'rendered'/f'FlowPilot_{name}_october2_corrected.pdf'
        pdf=fitz.open(path)
        overflow=[];blank=[]
        for i,p in enumerate(pdf):
            if len(p.get_text().strip())<8 and not p.get_image_info():blank.append(i+1)
            for block in p.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if span['text'].strip() and (x0 < -1 or y0 < -1 or x1 > p.rect.width+1 or y1 > p.rect.height+1):overflow.append([i+1,span['text']])
        check(name+': no blank pages',not blank,blank)
        check(name+': no text overflow',not overflow,overflow)
        if name=='ESI':
            check('Contents omit 3.2',not any(r['section']=='3.2' for r in toc))
            check('Contents match rendered pages',all(normalized(r['heading']) in normalized(pdf[r['page']-1].get_text()) for r in toc))
            for i,p in enumerate(pdf):
                if i in [1,2] or any(label in p.get_text() for label in ['Figure S4.','Figure S5.','Ablation test and architecture comparison']):
                    p.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(OUT/'rendered'/f'ESI_page_{i+1:02}.png')
        (OUT/f'{name}_page_index.json').write_text(json.dumps([{'page':i+1,'start':p.get_text()[:180]} for i,p in enumerate(pdf)],indent=2))
    for node in removed:
        value=text(node)
        if len(value)>120:check('Deleted content not relocated: '+value[:65],value not in text(pkgs['ESI'].doc))
    for path,digest in hashes.items():check('Source preserved: '+path,sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    with (OUT/'citation_audit.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(audit[0]));wr.writeheader();wr.writerows(audit)
    (OUT/'verification.json').write_text(json.dumps(CHECKS,indent=2,ensure_ascii=False))
    failed=[c for c in CHECKS if not c['passed']]
    print(json.dumps({'passed':len(CHECKS)-len(failed),'total':len(CHECKS),'failed':failed},indent=2))
    assert not failed


def main():
    OUT.mkdir(exist_ok=True)
    originals={name:Package(BASE/f'FlowPilot_{name}_october1_postmeeting_round2.docx') for name in ['manuscript','ESI']}
    hashes={str(p.source.relative_to(ROOT)):sha256(p.source.read_bytes()).hexdigest() for p in originals.values()}
    pkgs={name:Package(p.source) for name,p in originals.items()}
    removed,rids=remove_section(pkgs['ESI'])
    main_references(pkgs['manuscript'])
    renumber(pkgs)
    paths={name:BASE/f'FlowPilot_{name}_october2_corrected.docx' for name in pkgs}
    for name,pkg in pkgs.items():
        order_properties(pkg.doc)
        pkg.save(paths[name])
    (OUT/'source_manifest.json').write_text(json.dumps(hashes,indent=2))
    (OUT/'change_log.json').write_text(json.dumps(CHANGES,indent=2,ensure_ascii=False))
    render(list(paths.values()))
    last=None;toc_runs=[]
    for _ in range(4):
        with fitz.open(OUT/'rendered'/paths['ESI'].with_suffix('.pdf').name) as pdf:toc=refresh(paths['ESI'],pdf)
        toc_runs.append(toc);render([paths['ESI']])
        if toc==last:break
        last=toc
    else:raise RuntimeError('Contents did not stabilize')
    (OUT/'toc_audit.json').write_text(json.dumps(toc_runs,indent=2))
    verify({name:Package(path) for name,path in paths.items()},originals,removed,hashes,toc)


if __name__=='__main__':main()
