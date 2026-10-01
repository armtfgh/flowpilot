"""Author-facing revision notes and an evidence index, separate from the ESI."""
from hashlib import sha256
import csv
import json
import shutil

from docx import Document
from docx.shared import Cm, Pt
import fitz

import overhaul_submission_20260930 as rev
from report_professional_review_20260930 import paragraph, heading


def main():
    audit=json.loads((rev.OUT/'verification.json').read_text())
    assert audit['passed']==audit['total']
    pages=json.loads((rev.OUT/'rendered_page_map.json').read_text())
    shutil.copytree(rev.BASE/'revision_20260929/source_data',rev.OUT/'source_data',dirs_exist_ok=True)
    sources=[rev.MAIN,rev.ESI,rev.BASE/'Supplementary Information (KRICT)_final (1).docx',
             rev.ROOT/'deliverables/manuscript_benchmark_visualizations_20260825/source_data_revised/fig08_campaign_error_summary.csv',
             rev.ROOT/'visualization/panel_data_exports/fig3c_retrieval_pairs_raw.csv',
             rev.ROOT/'visualization/panel_data_exports/fig3d_family_match_rates.csv']
    manifest=[{'path':str(p.relative_to(rev.ROOT)),'sha256':sha256(p.read_bytes()).hexdigest()} for p in sources]
    (rev.OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Preserve intermediate contact sheets separately from the final visual audit.
    expected=set()
    for stem in ['manuscript','esi']:
        pdf=fitz.open(rev.OUT/f'rendered/{stem}_submission_overhauled_20260930.pdf')
        for start in range(0,len(pdf),12):
            expected.add(f'{stem}_pages_{start+1:03d}_{min(start+12,len(pdf)):03d}.png')
        for i,page in enumerate(pdf):
            if any(n in page.get_text() for n in ['Figure 5.', 'Figure 6.', 'Table S1.', 'Figure S17.', 'CONCLUSION']):
                expected.add(f'{stem}_detail_{i+1:03d}.png')
    intermediate=rev.OUT/'companion_archive/intermediate_visual_checks'
    for path in (rev.OUT/'visual_review').glob('*.png'):
        if path.name not in expected:
            intermediate.mkdir(exist_ok=True)
            shutil.move(path,intermediate/path.name)
    text=(
        '# FlowPilot submission overhaul: 30 September 2026\n\n'
        'Use the clean manuscript and ESI ending in `_submission_overhauled_20260930.docx` in the parent Submission folder. '
        'Matching `_marked.docx` copies show yellow revisions. Original Word files are unchanged.\n\n'
        '## Package contents\n'
        '- figures/: redesigned Figures 5 and 6, consolidated irradiation panel, chemical structures; PNG, PDF and SVG exports.\n'
        '- yield_comparison.csv: plotted NMR yields and source identification.\n'
        '- excluded_yield_bases.json: isolated literature values excluded from the NMR comparison.\n'
        '- numbering_map.csv and citation_audit.csv: old/new labels and every first manuscript citation.\n'
        '- contents_page_audit.json: refreshed ESI section/page mapping.\n'
        '- verification.json: deterministic document, arithmetic, evidence and layout checks.\n'
        '- editorial_changes.json: before/after text for recorded substantive changes.\n'
        '- source_data/: unchanged experimental and benchmark exports used in the audit.\n'
        '- companion_archive/: complete pre-streamlining ESI, long worked-run transcript, development history, '
        'and machine-readable case inventories with full design inputs.\n'
        '- rendered/ and visual_review/: PDFs, page contact sheets and enlarged inspected pages.\n\n'
        '## Yield convention\n'
        'Figures 5 and 6 use the label Yield (%); captions identify 1H NMR as the measurement method. '
        'Figure 5 compares KHU 97% (sets 1/3) and 98% (set 2) with Park et al., Table 4 entry 5: 90% NMR in flow '
        '(https://doi.org/10.1039/D4GC05769D). Figure 6 compares KHU 86% (set 1) and 68% (sets 2/3) '
        'with Saunders et al., Table 1 entry 5: 98% NMR in batch '
        '(https://doi.org/10.1021/acssuschemeng.5c00914). The 95% photochemical batch and 97% amidation flow literature values '
        'are isolated yields and are not relabeled as NMR. These references are contextual, not matched controls.\n\n'
        'Grouped response sets are not independent experiments. No experimental SD or missing yield has been invented. '
        'Isolation masses and characterization records remain in the ESI as analytical evidence, not extra plotted endpoints.\n\n'
        '## Verification scope\n'
        f'{audit["passed"]}/{audit["total"]} automated checks passed. '
        f'The clean manuscript has {pages["manuscript"]["pages"]} pages; the ESI has {pages["esi"]["pages"]} pages, '
        '23 supplementary figures and 25 tables. Main Figures 1-4 are preserved exactly. '
        'All main and supplementary figures/tables are cited in increasing first-appearance order. '
        'No archived score, critical-flag count or experimental observation was rescored or discarded. '
        'Automated checks and visual review do not establish experimental reproducibility or submission completeness.\n'
    )
    (rev.OUT/'README.md').write_text(text)
    (rev.OUT/'companion_archive/README.md').write_text(
        '# Preserved material\n\nThe full original reviewed ESI is retained as `esi_before_streamlining.docx`. '
        'It includes the retired supplementary figures/tables, complete worked-run output, and preprint bromination case. '
        'Its historical numbering is intentionally unchanged; consult `../numbering_map.csv` for retained material. '
        'No result has been erased or selectively rescored. The text and JSON exports preserve the exact recorded benchmark inputs. '
        'The source project retains the original full run folders identified within these records.\n')

    d=Document()
    section=d.sections[0]
    section.page_width,section.page_height=Cm(21),Cm(29.7)
    section.top_margin=section.bottom_margin=Cm(2)
    section.left_margin=section.right_margin=Cm(2.1)
    d.styles['Normal'].font.name='Times New Roman'
    d.styles['Normal'].font.size=Pt(11)
    heading(d,'FlowPilot: manuscript and ESI overhaul')
    paragraph(d,'30 September 2026 | Author-facing revision record, not manuscript text')
    heading(d,'Principal changes')
    points=[
        'Figures 5 and 6 have been rebuilt as chemistry, connected equipment and experimental-outcome panels. '
        'Each now shows Yield (%) without a separate isolated-yield column or empty yield cells. '
        'NMR is stated in the captions. Reaction structures are drawn from explicit molecular structures, and the existing equipment icons are reused.',
        'The reference endpoint is 90% NMR in flow for Figure 5 and 98% NMR in batch for Figure 6. '
        'The batch/flow distinction is visible in each figure. Literature isolated yields were not converted into NMR yields. '
        'The grouped KHU entries remain grouped, without invented replicate counts.',
        'The Conclusion now synthesizes the computational findings, connected-process chemistry, practical contribution and prospective next step. '
        'The Introduction remains prose-based; no main-text comparison table was added. The Methods retains eight subsections.',
        'The ESI is reduced from 31 figures/39 tables to 23 figures/25 tables. '
        'Long worked-output dumps, duplicate topology pages, the legacy bromination case, and administrative/redundant tables have been moved out of the active ESI, not deleted. '
        'The full pre-streamlining ESI is archived. The rubric, complete 15-condition module screen, failure examples, actual GUI captures, council excerpts, '
        'laboratory methods, photographs and all four supplied characterization spectra remain.',
        'Caption labels are bold; explanatory ESI caption text is regular weight. Tables use one gray-header style. '
        'Revised text remains Times New Roman 11 pt. The three irradiation-source pages are consolidated into one panel without changing the supplied spectral traces.',
        'All retained supplementary labels and section references were remapped. The first main-text citations now run Figure S1-S23 and Table S1-S25 in order; '
        'main figures run 1-6. Both bibliographies are checked for sequential first use. The ESI contents has been refreshed against the rendered pages.',
    ]
    for i,value in enumerate(points,1):paragraph(d,f'{i}. {value}')
    heading(d,'Checks and retained scope')
    paragraph(d,f'{audit["passed"]}/{audit["total"]} automated checks passed. '
              f'Clean PDFs: manuscript {pages["manuscript"]["pages"]} pages; ESI {pages["esi"]["pages"]} pages '
              '(previous ESI: 118 pages). Clean and highlighted Word copies have identical text and pagination. '
              'Page contact sheets and enlarged figure/caption pages were inspected; no out-of-page text was detected. '
              'Figures 1-4 retain their original artwork and dimensions, including the requested Figure 4 version.')
    paragraph(d,'The benchmark means (0.794 one-shot; 0.917 FlowPilot), flag totals (98; 4), model-level SDs, '
              'stage-time calculations and reagent equivalents were checked against the stored data. '
              'No new model campaign, experimental run, score adjustment or selection of favorable outcomes was performed.')
    heading(d,'Remaining author and laboratory checks')
    pending=[
        'KHU metadata: confirm MFC reference temperature/pressure and calibration; pressure reference and sensor position; oxygen-free feed preparation; '
        'check-valve branch/orientation and startup procedure; sample-loop collection and second-feed timing. '
        'Supply quantitative-NMR calculation sheets, internal-standard details, original analytical files and independent repeat IDs, if any. '
        'Confirm the reported tubing dimensions and 95 °C bath arrangement.',
        'Preserved Figure 3: its old Tier 2 label is repeated and some gain labels use percent signs for percentage-point changes. '
        'The caption/Methods explains the offline versus live retrieval distinction, but this unchanged artwork still needs final author-approved correction. '
        'The family-analysis helper also requires a strict self-exclusion audit before a leakage-free retrieval claim can be made.',
        'Preserved Figure 4: the display includes six generators, including archived GPT-5.4, while the headline cohort and ESI aggregate tables use five. '
        'This is explicitly disclosed in the caption and Methods; the requested artwork was not silently changed.',
        'Before submission: finalize contributions and competing-interests declarations, obtain author approvals, and complete the tagged code release, '
        'full benchmark/analytical data archive and persistent deposition details. The documents do not pretend that these materials are already deposited.',
    ]
    for i,value in enumerate(pending,1):paragraph(d,f'{i}. {value}')
    heading(d,'Files')
    for stem in ['manuscript','esi']:
        paragraph(d,f'{stem}_submission_overhauled_20260930.docx; matching _marked.docx and clean PDF.')
    paragraph(d,'overhaul_20260930/ contains figure exports, yield provenance, the citation and numbering audits, '
              'source-data snapshots, archived removed material, verification results and rendered-page review. '
              'Original input documents are unchanged. The CSV citation audit maps every retained ESI figure and table to its first manuscript citation.')
    d.save(rev.BASE/'Overhaul_notes_20260930.docx')
    print('Wrote author notes, package README, source-data snapshots and evidence manifest.')


if __name__=='__main__':main()
