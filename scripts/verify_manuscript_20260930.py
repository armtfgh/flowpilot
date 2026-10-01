"""Check the narrowly scoped manuscript revision and render a review record."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import re
import shutil

import fitz
from lxml import etree as E

import revise_manuscript_20260930 as rev
import verify_submission_20260929 as previous

BASE, OUT, W, NS = rev.BASE, rev.OUT, rev.W, rev.NS
checks = []


def check(name, condition, detail=None):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})


def xml(node):
    # Namespace declarations inherited from parents do not change the content.
    result = deepcopy(node)
    E.cleanup_namespaces(result)
    return E.tostring(result, method='c14n')


def images(pkg, paragraph):
    rels = {r.get('Id'): r.get('Target') for r in pkg.rels}
    return [sha256(pkg.parts['word/' + rels[rid]]).hexdigest()
            for rid in paragraph.xpath('.//a:blip/@r:embed', namespaces=NS)]


def main():
    manifest = json.loads((OUT / 'revision_manifest.json').read_text())
    for filename, digest in manifest['inputs'].items():
        check('Input unchanged: ' + Path(filename).name,
              sha256((rev.ROOT / filename).read_bytes()).hexdigest() == digest)
    current = rev.Package(BASE / 'manuscript_submission_revised_20260930.docx')
    marked = rev.Package(BASE / 'manuscript_submission_revised_20260930_marked.docx')
    source = rev.Package(rev.SOURCE)
    original = rev.Package(rev.ORIGINAL)
    esi = rev.Package(rev.ESI)
    check('No main-text table or Table 1 reference',
          not current.doc.xpath('.//w:tbl', namespaces=NS)
          and not re.search(r'\bTable\s+1\b', rev.text(current.doc)))
    check('ESI literature table remains cited', 'Table S36' in rev.text(current.doc))
    for n in range(1, 7):
        actual, caption = rev.image_before_caption(current, n)
        expected, expected_caption = rev.image_before_caption(original if n == 4 else source, n)
        check(f'Figure {n} drawing, geometry and content preserved from correct source',
              xml(actual) == xml(expected) and images(current, actual) == images(original if n == 4 else source, expected))
        if n != 4:
            check(f'Figure {n} caption unchanged', xml(caption) == xml(expected_caption))
    check('Six-model figure and five-model aggregate explicitly distinguished',
          'six-model display' in rev.text(rev.image_before_caption(current, 4)[1])
          and 'not included in the five-model aggregates' in rev.text(current.doc))
    for part in ('word/styles.xml', 'word/numbering.xml', 'word/fontTable.xml'):
        check(part + ' unchanged', current.parts.get(part) == source.parts.get(part))
    check('Section geometry unchanged', xml(current.body[-1]) == xml(source.body[-1]))
    check('All existing media retained unchanged',
          all(current.parts.get(k) == v for k, v in source.parts.items() if k.startswith('word/media/')))
    changed = set(manifest['changed_source_body_indices'])
    current_nodes = {xml(p) for p in current.body}
    unexpected = [(i, rev.text(p)[:100]) for i, p in enumerate(source.body) if i not in changed and xml(p) not in current_nodes]
    check('All paragraphs outside requested scope unchanged', not unexpected, unexpected)
    for i in manifest['layout_only_source_indices']:
        expected = list(source.body)[i]
        actual = next(p for p in current.body if rev.text(p) == rev.text(expected))
        changed_copy = deepcopy(actual)
        for flag in changed_copy.xpath('./w:pPr/w:keepLines', namespaces=NS):
            flag.getparent().remove(flag)
        check('Pagination-only fix preserves complete paragraph text and formatting', xml(changed_copy) == xml(expected))
    source_refs = list(source.body)[list(source.body).index(rev.find(source, 'REFERENCES')):]
    current_refs = list(current.body)[list(current.body).index(rev.find(current, 'REFERENCES')):]
    check('Complete reference list unchanged', [xml(p) for p in source_refs] == [xml(p) for p in current_refs])
    intro = rev.section(current, 'INTRODUCTION', 'RESULTS')
    check('Introduction contains eight prose paragraphs', len(intro) == 8 and all(p.tag == W + 'p' for p in intro))
    methods = rev.section(current, 'METHODS', 'AUTHOR INFORMATION')
    expected_headings = [t for t, _ in rev.METHODS]
    actual_headings = [rev.text(p) for p in methods if p.xpath('./w:pPr/w:pStyle[@w:val="Heading3"]', namespaces=NS)]
    check('Eight Methods subsections in order', actual_headings == expected_headings, actual_headings)
    bad_fonts = []
    for p in intro + methods:
        for r in p.findall(W + 'r'):
            if not rev.text(r).strip():
                continue
            if not r.xpath('./w:rPr/w:rFonts[@w:ascii="Times New Roman"]', namespaces=NS) or not r.xpath('./w:rPr/w:sz[@w:val="22"]', namespaces=NS):
                bad_fonts.append(rev.text(r))
    check('New prose and Methods headings retain TNR 11', not bad_fonts, bad_fonts)
    check('Clean and highlighted text identical', rev.text(current.doc) == rev.text(marked.doc))
    check('Yellow revisions included', bool(marked.doc.xpath('.//w:highlight[@w:val="yellow"]', namespaces=NS)))
    previous.OUT = OUT
    previous.checks = checks
    previous.crossrefs(current.doc, esi.doc)
    previous.fonts_tables(current.doc, 'Main')
    clean_pdf = OUT / 'manuscript_submission_revised_20260930.pdf'
    marked_pdf = OUT / 'manuscript_submission_revised_20260930_marked.pdf'
    pages = previous.rendered_pdf(clean_pdf, 'main')
    clean_render, marked_render = fitz.open(clean_pdf), fitz.open(marked_pdf)
    check('Clean and highlighted PDF pagination matches', len(clean_render) == len(marked_render))
    check('Clean and highlighted PDF text matches',
          ''.join(''.join(p.get_text().split()) for p in clean_render)
          == ''.join(''.join(p.get_text().split()) for p in marked_render))
    # Export edited sections and each figure page for high-resolution inspection.
    selected = set()
    for i, p in enumerate(clean_render):
        normalized = ' '.join(p.get_text().split())
        if 'INTRODUCTION' in normalized:
            selected.update(range(i, min(i + 3, len(clean_render))))
        if 'METHODS' in normalized:
            selected.update(range(i, min(i + 6, len(clean_render))))
        if any(f'Figure {n}.' in normalized for n in (4, 5, 6)):
            selected.add(i)
    for i in sorted(selected):
        clean_render[i].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(OUT / 'visual_review' / f'page_{i + 1:02d}.png')
    (OUT / 'rendered_page_map.json').write_text(json.dumps(pages, indent=2) + '\n')
    result = {'passed': sum(x['passed'] for x in checks), 'total': len(checks), 'checks': checks}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'total': result['total'], 'failed': [x for x in checks if not x['passed']]}, indent=2))
    assert result['passed'] == result['total'], 'Revision verification failed'
    shutil.copy2(clean_pdf, BASE / clean_pdf.name)
    (OUT / 'REVISION_NOTES.md').write_text(
        '# Focused manuscript revision, 30 September 2026\n\n'
        '- Removed the main-text literature table and every Table 1 reference. The comparison is integrated into the Introduction; the full ESI Table S36 is unchanged.\n'
        '- Rewrote the Introduction as eight connected paragraphs, retaining the cited literature and avoiding response-to-reviewer language.\n'
        '- Restored Figure 4 exactly from manuscript_revised_20260928.docx, including the original drawing and image bytes. Its caption explicitly distinguishes the six-model display from the five-model ESI aggregates.\n'
        '- Preserved Figures 1-3 and 5-6, their captions, the case-study results, all references, and the existing document styles and page settings.\n'
        '- Kept one existing paragraph together to avoid a two-line continuation stranded before Figure 5; its wording and font are unchanged.\n'
        '- Expanded Methods to eight subsections. The main text describes the methodological decisions, calculations, study designs, and statistical units; full prompts, inventories, rubric, recipes, and analytical records remain in the ESI.\n'
        '- Cross-checked against esi_submission_revised_20260929.docx without editing that file; methods_esi_crosscheck.csv records the mapping.\n'
        '- Clean and yellow-highlighted Word copies are supplied; their text and PDF pagination match.\n\n'
        f'Automated checks: {result["passed"]}/{result["total"]}. See verification.json, cross_reference_audit.csv, and visual_review/. '
        'The outstanding laboratory metadata and data-release items from the September 29 revision are unchanged. No new experiments, benchmark runs, or re-scoring were performed.\n'
    )


if __name__ == '__main__':
    main()
