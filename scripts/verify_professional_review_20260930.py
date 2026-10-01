"""Audit the reviewed Word packages, source calculations and rendered layout."""
from collections import Counter
from copy import deepcopy
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

import professional_review_20260930 as rev
import verify_submission_20260929 as prior
from verify_manuscript_20260930 import xml, images
from refresh_revision_toc_20260922 import normalized, refresh

OUT, BASE, ROOT = rev.OUT, rev.BASE, rev.ROOT
W, NS, text = rev.W, rev.NS, rev.text
checks = []


def check(name, condition, detail=None):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-7)


def without_keep_next(node):
    copy = deepcopy(node)
    for flag in copy.xpath('.//w:keepNext', namespaces=NS):
        flag.getparent().remove(flag)
    return xml(copy)


def packages():
    result = {}
    for stem, source in [('manuscript', rev.MAIN), ('esi', rev.ESI)]:
        original = rev.Package(source)
        clean = rev.Package(BASE / f'{stem}_submission_reviewed_20260930.docx')
        marked = rev.Package(BASE / f'{stem}_submission_reviewed_20260930_marked.docx')
        result[stem] = clean
        check(stem + ': styles, numbering and font table preserved',
              all(clean.parts.get(k) == original.parts.get(k) for k in
                  ['word/styles.xml', 'word/numbering.xml', 'word/fontTable.xml']))
        check(stem + ': page geometry preserved', xml(clean.body[-1]) == xml(original.body[-1]))
        check(stem + ': all embedded media byte-identical',
              all(clean.parts.get(k) == v for k, v in original.parts.items() if k.startswith('word/media/')))
        check(stem + ': clean and marked text identical', text(clean.doc) == text(marked.doc))
        check(stem + ': yellow revision markings present',
              bool(marked.doc.xpath('.//w:highlight[@w:val="yellow"]', namespaces=NS)))
        bad_fonts = []
        for run in marked.doc.xpath('.//w:r[w:rPr/w:highlight[@w:val="yellow"]]', namespaces=NS):
            if not text(run).strip():
                continue
            if not run.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]', namespaces=NS) or not run.xpath('./w:rPr/w:sz[@w:val="22"]', namespaces=NS):
                bad_fonts.append(text(run))
        check(stem + ': revised text is Times New Roman 11', not bad_fonts, bad_fonts)
        rels = {r.get('Id'): r.get('Target') for r in clean.rels}
        missing = [rid for rid in clean.doc.xpath('.//a:blip/@r:embed', namespaces=NS)
                   if 'word/' + rels.get(rid, 'MISSING') not in clean.parts]
        check(stem + ': all image relationships resolve', not missing, missing)
        check(stem + ': no unresolved editorial markers', not any(s in text(clean.doc) for s in ['[[', 'XXX', 'YYY', '[Name to be added]']))
        check(stem + ': tables unchanged except declared pagination flags',
              [without_keep_next(t) for t in clean.doc.iter(W + 'tbl')] == [without_keep_next(t) for t in original.doc.iter(W + 'tbl')])
        prior.fonts_tables(clean.doc, stem)
    m, s = result['manuscript'], result['esi']
    check('Author lines match', text(m.body[1]) == text(s.body[1]))
    for email in ['boyoungy.park@khu.ac.kr', 'gnahn@krict.re.kr']:
        check('Corresponding-author email in both documents: ' + email,
              email in text(m.doc) and email in text(s.doc))
    source_main = rev.Package(rev.MAIN)
    original_figure4 = rev.Package(BASE / 'manuscript_revised_20260928.docx')
    for n in range(1, 7):
        actual, caption = rev.base.image_before_caption(m, n)
        expected_pkg = original_figure4 if n == 4 else source_main
        expected, old_caption = rev.base.image_before_caption(expected_pkg, n)
        check(f'Figure {n}: accepted artwork and dimensions preserved',
              xml(actual) == xml(expected) and images(m, actual) == images(expected_pkg, expected))
        if n in (1, 2, 5, 6):
            check(f'Figure {n}: caption unchanged', xml(caption) == xml(old_caption))
    check('No main-text table / Table 1', not list(m.doc.iter(W + 'tbl')) and not re.search(r'\bTable 1\b', text(m.doc)))
    prior.crossrefs(m.doc, s.doc)
    methods = rev.base.section(m, 'METHODS', 'AUTHOR INFORMATION')
    headings = [text(p) for p in methods if p.xpath('./w:pPr/w:pStyle[@w:val="Heading3"]', namespaces=NS)]
    check('Eight Methods subsections', len(headings) == 8, headings)
    discussion = rev.base.section(m, 'DISCUSSION', 'CONCLUSION')
    check('Discussion is four connected paragraphs without subheadings', len(discussion) == 4 and not any(p.xpath('./w:pPr/w:pStyle[starts-with(@w:val,"Heading")]', namespaces=NS) for p in discussion))
    # Reference numbers are the numeric superscript runs, excluding footnotes and units.
    before_refs = list(m.body)[:list(m.body).index(rev.base.find(m, 'REFERENCES'))]
    citations, ordered = [], []
    for p in before_refs:
        for run in p.xpath('.//w:r[w:rPr/w:vertAlign[@w:val="superscript"]]', namespaces=NS):
            value = text(run)
            if not re.fullmatch(r'\d+(?:[,\-–]\d+)*', value):
                continue
            for token in value.split(','):
                bounds = re.split('[-–]', token)
                numbers = range(int(bounds[0]), int(bounds[-1]) + 1)
                for n in numbers:
                    if n > 3 or (n in (1, 2, 3) and p in rev.base.section(m, 'INTRODUCTION', 'RESULTS')):
                        citations.append(n)
                        if n not in ordered:
                            ordered.append(n)
    check('All 62 bibliography entries cited', set(citations) == set(range(1, 63)), sorted(set(citations)))
    check('Bibliographic first appearances ordered 1-62', ordered == list(range(1, 63)), ordered)
    refs = list(m.body)[list(m.body).index(rev.base.find(m, 'REFERENCES')) + 1:]
    ref_text = [text(p) for p in refs if text(p).strip()]
    check('Bibliography retains 62 entries', len(ref_text) == 62, len(ref_text))
    check('Corrected Yasukawa reference includes volume and pages', any('Yasukawa' in t and '7 (7), 1099–1101' in t for t in ref_text))
    check('Materealize explicitly identified as preprint', any('2601.15743v1. Preprint.' in t for t in ref_text))
    return result


def evidence():
    src = BASE / 'revision_20260929/source_data'
    data = rows(ROOT / 'deliverables/manuscript_benchmark_visualizations_20260825/source_data_revised/fig08_campaign_error_summary.csv')
    check('Benchmark has 90 unique outcomes', len(data) == 90 and len({r['candidate_id'] for r in data}) == 90)
    for arch, score, count, free in [('One-shot', .794, 98, 18), ('FlowPilot', .917, 4, 43)]:
        subset = [r for r in data if r['architecture'] == arch]
        check(arch + ': mean score verified', round(statistics.mean(float(r['benchmark_score']) for r in subset), 3) == score)
        check(arch + ': critical flags and flag-free count verified', sum(int(r['judge_flag_count']) for r in subset) == count and sum(int(r['judge_flag_count']) == 0 for r in subset) == free)
    for row in rows(src / 'figure4_scores.csv'):
        subset = [r for r in data if r['model'] == row['model'] and r['architecture'] == row['architecture']]
        means = [statistics.mean(float(r['benchmark_score']) for r in subset if r['repeat_id'] == repeat) for repeat in sorted({r['repeat_id'] for r in subset})]
        check(row['model'] + ' ' + row['architecture'] + ': mean/SD recomputed', len(subset) == 9 and close(statistics.mean(means), row['mean']) and close(statistics.stdev(means), row['sample_sd']))
    for row in rows(src / 'khu_reported_experimental_results.csv'):
        q = float(row['Q1_mL_min']) + float(row['Q2_mL_min']) + float(row['Qgas_reference_mL_min'])
        check(f"Figure {row['figure']} sets {row['sets']}: stage-time arithmetic", close(float(row['V1_mL']) / float(row['Q1_mL_min']), row['t1_min']) and close(float(row['V2_mL']) / q, row['t2_index_min']))
    check('Oxygen equivalents at explicitly assumed STP', math.isclose(.09 / 22.414 / (.1 * .02), 2.0078, rel_tol=1e-4))
    for qa, qb in [(.16, .04), (.28, .07)]:
        check(f'Amine equivalents at QA={qa}', close(2.1 * qb / (.5 * qa), 1.05))
    check('Isolated masses agree with rounded yields', round(42 / (.2 * 223.29) * 100) == 94 and round(227 / 270.29 * 100) == 84)
    modules = rows(src / 'tableS10_all15_module_conditions.csv')
    check('Module screen has 15 conditions and 45 outcomes', len(modules) == 15 and sum(int(r['n_outcomes']) for r in modules) == 45)
    pairs = rows(ROOT / 'visualization/panel_data_exports/fig3c_retrieval_pairs_raw.csv')
    check('Retrieval rank analysis: 80 queries / 1600 pairs / 20 per query', len(pairs) == 1600 and len(Counter(r['query_id'] for r in pairs)) == 80 and set(Counter(r['query_id'] for r in pairs).values()) == {20})
    check('Retrieval rank analysis: 401 rank changes', sum(int(r['rank_delta']) != 0 for r in pairs) == 401)
    check('Retrieval rank analysis: no query-ID self-hit', all(r['query_id'] != r['result_id'] for r in pairs))
    check('Retrieval rank analysis: reduced three-field score closes', all(close(sum(float(r[k]) for k in ('fs_pc', 'fs_sol', 'fs_wl')), r['field_score']) for r in pairs))
    check('Retrieval rank analysis: final weighted score closes', all(close(.6 * float(r['sem_score']) + .4 * float(r['field_score']), r['final_score']) for r in pairs))


def renders():
    result = {}
    for stem in ['manuscript', 'esi']:
        path = OUT / f'rendered/{stem}_submission_reviewed_20260930.pdf'
        marked_path = path.with_stem(path.stem + '_marked')
        result[stem] = prior.rendered_pdf(path, stem)
        clean, marked = fitz.open(path), fitz.open(marked_path)
        check(stem + ': clean/marked pagination matches', len(clean) == len(marked))
        check(stem + ': clean/marked rendered text matches',
              [''.join(p.get_text().split()) for p in clean] == [''.join(p.get_text().split()) for p in marked])
        needles = ['INTRODUCTION', 'DISCUSSION', 'METHODS', 'Retrospective', 'retrieval analysis', 'KHU implementation', 'backflow', 'Figure 4.', 'Figure 5.', 'Figure 6.', 'Figure S24.']
        for i, page in enumerate(clean):
            if any(word.lower() in page.get_text().lower() for word in needles):
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT / f'visual_review/{stem}_detail_{i+1:03d}.png')
        shutil.copy2(path, BASE / path.name)
    toc = json.loads((OUT / 'contents_page_audit.json').read_text())
    esi_pdf = fitz.open(OUT / 'rendered/esi_submission_reviewed_20260930.pdf')
    bad = [r for r in toc if normalized(r['heading']) not in normalized(esi_pdf[r['page'] - 1].get_text())]
    check('All 45 ESI contents entries point to current heading pages', len(toc) == 45 and not bad, bad)
    (OUT / 'rendered_page_map.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    prior.OUT, prior.checks = OUT, checks
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    for path, digest in manifest.items():
        check('Source unchanged: ' + Path(path).name, sha256((ROOT / path).read_bytes()).hexdigest() == digest)
    packages()
    evidence()
    renders()
    result = {'passed': sum(c['passed'] for c in checks), 'total': len(checks), 'checks': checks}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'total': result['total'], 'failed': [c for c in checks if not c['passed']]}, indent=2))
    assert result['passed'] == result['total'], 'Review verification failed'


if __name__ == '__main__':
    import sys
    if '--refresh-toc' in sys.argv:
        pdf = fitz.open(OUT / 'rendered/esi_submission_reviewed_20260930.pdf')
        for suffix in ('', '_marked'):
            path = BASE / f'esi_submission_reviewed_20260930{suffix}.docx'
            entries = refresh(path, pdf)
            parts, doc = prior.read(path)
            rev.base.prior.bold_references(doc)
            rev.order_properties(doc)
            rev.base.prior.intro_tools.write_package(parts, doc, path)
        (OUT / 'contents_page_audit.json').write_text(json.dumps(entries, indent=2) + '\n')
        print(f'Refreshed {len(entries)} contents entries; rerender to verify pagination.')
    else:
        main()
