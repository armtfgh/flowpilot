"""Check the narrative revision against source documents and recorded evidence."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import re
import shutil
import statistics

import fitz
from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

import revise_narrative_20261001 as rev
from overhaul_submission_20260930 import with_cites
from verify_overhaul_20260930 import labeled
from verify_inventory_gui_esi_20261001 import media_hashes
from refresh_revision_toc_20260922 import normalized
from revise_manuscript_cases_20260922 import expanded_cites
from revise_prior_work_20260928 import LABEL

ROOT, BASE, OUT, W, NS, text = rev.ROOT, rev.BASE, rev.OUT, rev.W, rev.NS, rev.text
CHECKS = []


def check(name, passed, detail=None):
    CHECKS.append({'check': name, 'passed': bool(passed), 'detail': detail})


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def package_checks():
    packages = {}
    for name in ['manuscript', 'esi']:
        old = rev.Package(rev.INPUTS[name])
        pkg = rev.Package(BASE / f'{name}_submission_narrative_20261001.docx')
        marked = rev.Package(BASE / f'{name}_submission_narrative_20261001_marked.docx')
        packages[name] = pkg
        for part in ['word/styles.xml', 'word/fontTable.xml', 'word/numbering.xml']:
            check(f'{name}: preserved {part}', old.parts.get(part) == pkg.parts.get(part))
        check(f'{name}: page setup preserved', E.tostring(old.body[-1]) == E.tostring(pkg.body[-1]))
        check(f'{name}: all embedded images unchanged', all(pkg.parts.get(k) == v for k, v in old.parts.items() if k.startswith('word/media/')))
        check(f'{name}: displayed image order unchanged', media_hashes(old) == media_hashes(pkg))
        drawings = lambda p: [E.tostring(n) for n in p.doc.iter(W + 'drawing')]
        check(f'{name}: image sizing, cropping and positioning unchanged', drawings(old) == drawings(pkg))
        check(f'{name}: complete tables unchanged', [E.tostring(t) for t in old.doc.iter(W + 'tbl')] == [E.tostring(t) for t in pkg.doc.iter(W + 'tbl')])
        ref_title = 'REFERENCES' if name == 'manuscript' else 'References'
        def bibliography(p):
            children = list(p.body)
            start = next(i for i, x in enumerate(children) if text(x) == ref_title)
            # The TOC refresh updates the heading bookmark, not the references.
            return [E.tostring(x) for x in children[start+1:]]
        check(f'{name}: bibliography and reference numbering unchanged', bibliography(old) == bibliography(pkg))
        check(f'{name}: clean and marked text identical', text(pkg.doc) == text(marked.doc))
        check(f'{name}: clean copy has no revision highlighting', not pkg.doc.xpath('.//w:highlight[@w:val="yellow"]', namespaces=NS))
        highlighted = marked.doc.xpath('.//w:r[w:rPr/w:highlight[@w:val="yellow"]]', namespaces=NS)
        check(f'{name}: marked copy highlights revisions', bool(highlighted))
        bad_font = [text(r) for r in highlighted if text(r).strip() and (
            not r.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]', namespaces=NS)
            or not r.xpath('./w:rPr/w:sz[@w:val="22"]', namespaces=NS))]
        check(f'{name}: revised text uses Times New Roman 11', not bad_font, bad_font)
        check(f'{name}: author and affiliation paragraphs preserved', [E.tostring(p) for p in old.ps[1:5]] == [E.tostring(p) for p in pkg.ps[1:5]])
        for part, content in pkg.parts.items():
            if part.endswith('.xml'):
                E.fromstring(content)
        check(f'{name}: all XML parses', True)
    check('Manuscript and ESI titles agree', text(packages['manuscript'].ps[0]) == text(packages['esi'].ps[0]) == rev.TITLE)
    for path, digest in json.loads((OUT / 'source_manifest.json').read_text()).items():
        check('Original remains unchanged: ' + path, sha256((ROOT / path).read_bytes()).hexdigest() == digest)
    # Frozen inputs, archived statements and laboratory records remain verbatim.
    old_esi = rev.Package(rev.INPUTS['esi'])
    new_esi_paras = [text(p) for p in packages['esi'].ps]
    for a, b, label in [(104, 110, 'benchmark wrapper'), (137, 192, 'benchmark case inputs'),
                        (283, 298, 'Giese responses'), (307, 319, 'amidation responses'),
                        (321, 375, 'experimental section')]:
        expected = [text(p) for p in old_esi.ps[a:b] if text(p)]
        check('Exact source text retained: ' + label, all(t in new_esi_paras for t in expected))
    return packages


def reference_checks(packages):
    main, esi = packages['manuscript'], packages['esi']
    defs = {}
    for p in esi.body:
        match = re.match(r'^(Figure|Table) S(\d+)\.', text(p))
        if match:
            key = (match[1], int(match[2]))
            check('Unique ESI caption ' + match[0], key not in defs)
            defs[key] = text(p)
            offset = 0
            bad = []
            for r in p.iter(W + 'r'):
                is_bold = r.xpath('./w:rPr/w:b[not(@w:val="0" or @w:val="false")]', namespaces=NS)
                if offset >= match.end() and text(r).strip() and is_bold:
                    bad.append(text(r))
                offset += len(text(r))
            check(match[0] + ' caption body remains regular', not bad, bad)
    audit = []
    for tag, pkg in packages.items():
        seen = defaultdict(list)
        unresolved = []
        for i, p in enumerate(pkg.body):
            if p.tag == W + 'sdt':
                continue
            for kind, n in labeled(text(p)):
                if (kind, n) not in defs:
                    unresolved.append((kind, n, text(p)))
                if n not in seen[kind]:
                    seen[kind].append(n)
                    audit.append({'document': tag, 'label': f'{kind} S{n}', 'body_index': i,
                                  'first_citation': text(p), 'caption': defs.get((kind, n))})
        check(tag + ': ESI references resolve', not unresolved, unresolved)
        for kind, count in [('Figure', 21), ('Table', 25)]:
            check(f'{tag}: all {kind} S1-S{count} cited in first-appearance order', seen[kind] == list(range(1, count+1)), seen[kind])
    with (OUT / 'citation_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    main_text = []
    for p in main.body:
        if text(p) == 'REFERENCES':
            break
        main_text.append(p)
    first = []
    for p in main_text[6:]:
        for s in re.findall(r'\[\[(.*?)\]\]', with_cites(p)):
            for n in expanded_cites(s):
                if n not in first:
                    first.append(n)
    check('All 61 literature references cited in first-appearance order', first == list(range(1, 62)), first)
    figure_first = []
    for p in main_text:
        for m in re.finditer(r'\bFigure[s]? (\d+)', text(p)):
            n = int(m[1])
            if n not in figure_first:
                figure_first.append(n)
    check('Main figures retain first-citation order 1-6', figure_first == list(range(1, 7)), figure_first)
    check('No separate institutional actor in main narrative', not any(re.search(r'\bKHU\b|Kyung Hee University \(KHU\)', text(p)) for p in main_text[6:]))
    check('Main narrative no longer describes FlowPilot as a translator', not any(re.search(r'batch.to.flow (?:translation|translator|design workflow)', text(p), re.I) for p in main_text))
    check('Scope explicitly defines end-to-end as design', 'End-to-end denotes this integrated design workflow' in text(main.doc))
    check('Response sets explained before cases', text(main.doc).index('For each chemistry, three standardized response sets') < text(main.doc).index('Staged oxygen delivery in photoredox'))
    nonbold=[]
    for p in main_text[6:]:
        if re.match(r'^Figure \d+\.',text(p)):
            continue
        value='';bold_mask=[]
        for r in p.iter(W+'r'):
            t=text(r);value+=t
            flag=bool(r.xpath('./w:rPr/w:b[not(@w:val="0" or @w:val="false")]',namespaces=NS))
            bold_mask.extend([flag]*len(t))
        for match in LABEL.finditer(value):
            if not all(bold_mask[match.start():match.end()]):
                nonbold.append(match[0])
    check('Main-text figure and table references are bold',not nonbold,nonbold)


def evidence_checks():
    data = ROOT / 'deliverables/manuscript_benchmark_visualizations_20260825/source_data_revised'
    campaigns = rows(data / 'fig08_campaign_error_summary.csv')
    scores = rows(data / 'fig01_model_architecture_mean_sd_revised.csv')
    cases = rows(data / 'fig04_case_paired_effects_revised.csv')
    criteria = rows(data / 'fig07_criterion_gain_heatmap_revised.csv')
    module = rows(data / 'fig11_module_condition_scores_revised.csv')
    out = {'archives': {}, 'architecture': {}, 'model_deltas': {}, 'interpretation_checks': {}}
    for file in ['fig08_campaign_error_summary.csv', 'fig01_model_architecture_mean_sd_revised.csv', 'fig04_case_paired_effects_revised.csv',
                 'fig07_criterion_gain_heatmap_revised.csv', 'fig11_module_condition_scores_revised.csv']:
        out['archives'][str((data/file).relative_to(ROOT))] = sha256((data/file).read_bytes()).hexdigest()
    check('90 benchmark outcomes in archived evidence', len(campaigns) == 90, len(campaigns))
    for architecture, expected in [('One-shot', (0.794, 98, 18)), ('FlowPilot', (0.917, 4, 43))]:
        rr = [r for r in campaigns if r['architecture'] == architecture]
        mean = statistics.mean(float(r['benchmark_score']) for r in rr)
        flags = sum(int(r['judge_flag_count']) for r in rr)
        zero = sum(int(r['judge_flag_count']) == 0 for r in rr)
        out['architecture'][architecture] = {'n': len(rr), 'mean': mean, 'flags': flags, 'zero_flags': zero}
        check(architecture + ': aggregate mean/flags/zero-flag count verified', (round(mean,3),flags,zero) == expected, out['architecture'][architecture])
    by_model = defaultdict(dict)
    for r in scores:
        by_model[r['model']][r['architecture']] = float(r['mean'])
        rr = [r0 for r0 in campaigns if r0['model'] == r['model'] and r0['architecture'] == r['architecture']]
        by_repeat = defaultdict(list)
        for r0 in rr:
            by_repeat[r0['repeat_id']].append(float(r0['benchmark_score']))
        values = [statistics.mean(v) for v in by_repeat.values()]
        check(f"{r['model']} {r['architecture']}: mean/SD reconstructed from 3 repeat-level means",
              len(values) == 3 and all(len(v) == 3 for v in by_repeat.values())
              and abs(statistics.mean(values)-float(r['mean'])) < 2e-6
              and abs(statistics.stdev(values)-float(r['sample_sd'])) < 2e-6)
    out['model_deltas'] = {model: value['FlowPilot']-value['One-shot'] for model,value in by_model.items()}
    check('Positive mean gains for all five generators', len(out['model_deltas']) == 5 and all(v > 0 for v in out['model_deltas'].values()))
    check('Both 27B pipeline means exceed tested commercial one-shot means',
          min(v['FlowPilot'] for k,v in by_model.items() if 'Qwen' in k) > max(v['One-shot'] for k,v in by_model.items() if 'Qwen' not in k))
    check('Hydrogenolysis largest gain for Qwen and Opus', all(max([r for r in cases if r['model'] == model], key=lambda r: float(r['mean']))['case'] == 'Hydrogenolysis'
                                                            for model in ['Qwen3.6-27B','Qwen3.8-27B','Claude Opus 4.6']))
    check('Topology criterion gain positive in all 15 model-case cells', len([r for r in criteria if r['criterion_id']=='UO-05' and float(r['mean_paired_delta'])>0]) == 15)
    means = {r['condition']:float(r['mean_score_0_1']) for r in module}
    check('Internal screen full/no council/one-shot numbers verified', [round(means[k],3) for k in ['Full FlowPilot','No council','One-shot']] == [.928,.670,.743])
    wet = {
        'oxygen_mmol_min': .09 / 22.414,
        'oxygen_equiv': (.09/22.414)/(.1*.02),
        'stage2_inlet_reference_index_min': 20/(.02+.09),
        'amide_set1_stage_min': [5/.16,5/(.16+.04)],
        'amide_set23_stage_min': [10/.28,5/(.28+.07)],
        'amide_set1_feed_mmol_min': [.5*.16,2.1*.04],
        'amide_set23_feed_mmol_min': [.5*.28,2.1*.07],
    }
    out['calculation_rechecks'] = wet
    check('Oxygen equivalent and inlet index rechecked', round(wet['oxygen_equiv'],2) == 2.01 and round(wet['stage2_inlet_reference_index_min'],2) == 181.82)
    check('Amidation stage calculations rechecked', [round(x,2) for x in wet['amide_set1_stage_min']] == [31.25,25.0] and [round(x,2) for x in wet['amide_set23_stage_min']] == [35.71,14.29])
    (OUT/'evidence_audit.json').write_text(json.dumps(out,indent=2)+'\n')


def render_checks():
    visual = OUT / 'visual_review'
    visual.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf', 17)
    toc = json.loads((OUT/'contents_page_audit.json').read_text())
    for name in ['manuscript','esi']:
        path = OUT/f'rendered/{name}_submission_narrative_20261001.pdf'
        pdf = fitz.open(path)
        marked = fitz.open(path.with_stem(path.stem+'_marked'))
        check(f'{name}: identical clean/marked pagination', len(pdf) == len(marked), len(pdf))
        check(f'{name}: identical clean/marked rendered text', [' '.join(p.get_text().split()) for p in pdf] == [' '.join(p.get_text().split()) for p in marked])
        blank, overflow, detached = [], [], []
        thumbs = []
        for i,p in enumerate(pdf):
            t = p.get_text()
            if len(t.strip()) < 8 and not p.get_image_info():
                blank.append(i+1)
            if re.search(r'^Figure S?\d+\.', t, re.M) and not p.get_image_info():
                detached.append(i+1)
            for block in p.get_text('dict')['blocks']:
                if block.get('type') != 0:
                    continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0,y0,x1,y1 = span['bbox']
                        if x0 < -1 or y0 < -1 or x1 > p.rect.width+1 or y1 > p.rect.height+1:
                            overflow.append((i+1,span['text']))
            if name == 'manuscript' or i in [0,2,3,4,5,21,52,53,54]:
                pix = p.get_pixmap(matrix=fitz.Matrix(.58,.58),alpha=False)
                im = Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
                canvas=Image.new('RGB',(im.width+8,im.height+30),'#eeeeee')
                canvas.paste(im,(4,26));ImageDraw.Draw(canvas).text((6,3),f'{name} page {i+1}',fill='black',font=font)
                thumbs.append(canvas)
            if name == 'manuscript' and (p.get_image_info() or any(h in t for h in ['Architecture effects','Staged oxygen','Coupled feed','DISCUSSION']) or i == 0):
                p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(visual/f'{name}_{i+1:02d}.png')
        check(f'{name}: no blank pages', not blank, blank)
        check(f'{name}: no text outside page', not overflow, overflow)
        check(f'{name}: captions share page with artwork', not detached, detached)
        if name == 'esi':
            check('ESI table of contents page numbers correct', all(normalized(r['heading']) in normalized(pdf[r['page']-1].get_text()) for r in toc))
        else:
            heading_pages=[p for p in pdf if 'Availability of data and code' in p.get_text()]
            check('Data-availability heading stays with its paragraph',len(heading_pages)==1 and 'Source code is publicly accessible' in heading_pages[0].get_text())
        for start in range(0,len(thumbs),12):
            batch=thumbs[start:start+12];cw=max(im.width for im in batch);ch=max(im.height for im in batch)
            canvas=Image.new('RGB',(cw*4,ch*math.ceil(len(batch)/4)),'white')
            for j,im in enumerate(batch):canvas.paste(im,((j%4)*cw,(j//4)*ch))
            canvas.save(visual/f'{name}_contact_{start//12+1}.png')
        shutil.copy2(path,BASE/path.name)
    outline_path=OUT/'rendered/FlowPilot_paragraph_map_20261001.pdf'
    outline=fitz.open(outline_path)
    outline_text='\n'.join(p.get_text() for p in outline)
    check('Paragraph outline is exactly two pages',len(outline)==2,len(outline))
    check('Paragraph outline includes all 56 paragraph IDs exactly once',
          re.findall(r'(?m)^[^\w\n]*(P\d{2})\b',outline_text)==[f'P{i:02d}' for i in range(1,57)])
    for i,page in enumerate(outline):
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(visual/f'outline_page_{i+1}.png')
    shutil.copy2(outline_path,BASE/outline_path.name)


def main():
    pkgs=package_checks()
    reference_checks(pkgs)
    evidence_checks()
    render_checks()
    report={'passed':sum(c['passed'] for c in CHECKS),'total':len(CHECKS),'checks':CHECKS}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'total':report['total'],'failed':[c for c in CHECKS if not c['passed']]},indent=2))
    assert all(c['passed'] for c in CHECKS)


if __name__=='__main__':
    main()
