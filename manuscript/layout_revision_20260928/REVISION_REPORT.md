# ESI formatting and Discussion revision

Date: 28 September 2026. This pass starts from the clean documents delivered in the preceding revision. Those files remain unchanged.

## Deliverables

- `../manuscript_revised_20260928_v2.docx`
- `../manuscript_revised_20260928_v2_marked.docx`
- `../esi_revised_20260928_v2.docx`
- `../esi_revised_20260928_v2_marked.docx`

The marked copies highlight this revision's rewritten text and affected captions in yellow. Formatting-only table changes are indicated by highlighted table captions, keeping the table cells in the requested uniform style.

## 1. Uniform ESI tables

All 39 editable table blocks, including continued panels, now use:

- Light-gray header fill (`E6E6E6`) with bold black text.
- White body cells, no alternating blue bands.
- Consistent 0.5-point gray borders (`B8B8B8`).
- Times New Roman, 11 pt throughout the cells.
- Consistent padding, single line spacing, left-aligned text and top-aligned cells.
- Repeated column headers when a table continues onto another page.
- Page-width alignment; column proportions remain content-dependent. Two narrow headers were given enough room to avoid breaking words mid-word.

All cell text, numerical data and table numbering are unchanged. Tables embedded within genuine GUI screenshots or the archived preprint artwork are part of those images and have not been repainted.

## 2. Shorter, continuous Discussion

The Discussion was reduced from 1,098 to 452 whitespace-delimited words, approximately 59%. It now consists of five continuous paragraphs under the existing DISCUSSION heading. All internal subheadings and the long standalone limitations subsection were removed.

The revised sequence covers the connected-process contribution, model/council results, positioning relative to prior work, record-level auditability, and the two-stage experimental evaluation and feedback direction. A few concise qualifications necessary to interpret the evidence remain within the narrative; detailed scope and evidence boundaries remain available in the ESI.

Every other main-manuscript section, the bibliography and all main-text figures are unchanged. All 62 references remain cited in first-appearance order. Main-text citations still cover all ESI Figures S1-S22 and Tables S1-S36.

## 3. Straightened process diagrams

Eight panels were regenerated and inserted:

- Figure S13.
- Figure S17(a).
- Figure S20(a-c).
- Figure S21(a-c).

The main process icons now share one horizontal centerline. Secondary feed trains also use level rows, with right-angle connections into their original mixing points. No material connection was added, deleted or redirected.

Figure S13 was regenerated from its archived topology using the existing equipment renderer, then re-laid out. Its recorded 1.4 oxygen-equivalent annotation is retained from the archived rationale. S17(a) retains the archived graph's numerical labels and equipment names, with generic Pump symbols replacing the misleading legacy syringe glyphs. S20/S21 retain their original node labels and embedded icons; only positions and arrow geometry changed.

Captions explicitly identify the publication redraws. S17(b) remains the original GUI screenshot. All other screenshots and the unmodified preprint artwork in S22 are preserved. No new LLM generation, benchmark campaign, experiment or design recalculation was performed.

Individual SVG, PNG and PDF exports are in `figures/`. The PNGs are 3,900 pixels wide, exceeding 300 dpi at the document's figure widths. SVG labels and arrows remain vector elements; the original equipment icons are embedded bitmaps. `figure_layout_audit.json` records source paths, hashes, node coordinates and connections.

## Verification

The documents were rendered and visually checked, including the Discussion, representative table styles, long headers, the new comparison table, and all eight topology panels. Final rendering produced 29 main-manuscript pages and 98 ESI pages in the available LibreOffice environment; pagination can vary by Word installation.

`independent_verification.json` records 56 checks covering:

- Source-file hashes and the scope of changed Word-package components.
- Unchanged main-text content outside the Discussion.
- Unchanged ESI table contents, data and image assignments.
- Uniform table styling, 11 pt typography and repeated headers.
- Complete reference/cross-reference coverage and retained bold labels.
- Horizontal process paths, orthogonal arrows and separated label bounds.
- Unchanged archived topology JSON files.
- Rendered page bounds, absence of blank pages, and updated contents-page targets.
- Matching clean/marked content and formatting apart from yellow highlights.

Original figures extracted for inspection are in `original_figures/`; inspection contact sheets and page previews are retained alongside the audit. Source documents and run archives were not overwritten.

## Reproduction

Run `scripts/redraw_esi_linear_topologies_20260928.py`, followed by `scripts/revise_layout_discussion_20260928.py`, using `.venv-flowpilot/bin/python`. Render the four v2 DOCX files into this folder's `rendered/` directory with LibreOffice. Run `scripts/verify_layout_revision_20260928.py --refresh-toc`, rerender the two ESI files, then run `scripts/verify_layout_revision_20260928.py` without arguments.
