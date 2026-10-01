# FlowPilot literature-positioning revision

Date: 28 September 2026

## Revised Word files

- `../manuscript_revised_20260928.docx`: clean manuscript.
- `../manuscript_revised_20260928_marked.docx`: matching manuscript with substantive revisions highlighted yellow.
- `../esi_revised_20260928.docx`: clean ESI.
- `../esi_revised_20260928_marked.docx`: matching ESI with substantive additions highlighted yellow.

The source files remain unchanged:

- `../manuscript_introduction_revised_20260923.docx`
- `../esi_revised_20260922.docx`

## Manuscript changes

1. Revised the Introduction's related-work and contribution paragraphs. The first two opening paragraphs are retained. The argument now distinguishes reaction-to-configuration translation from local flow-pattern prediction, optimization on an established platform, and general process simulation.
2. Added all six studies suggested by Dr. Ahn: Chat-microreactor, LLM-RDF, SapoMind, Text-to-Simulation, CeProAgents and CAAF. CeProAgents and CAAF are explicitly identified as preprints, with pinned versions in the bibliography.
3. Revised the Discussion to acknowledge overlap in reactor engineering, experimental development and deterministic validation. The one-shot architecture comparison is explicitly distinguished from a head-to-head benchmark against these published systems.
4. Added one implementation-focused sentence to each of the Figure 5 and Figure 6 case introductions. No proposed operating settings, chemical structures, figures, or yield fields were changed.
5. Added a Methods cross-reference to the new comparison table, after the existing Table S35 citation. Existing ESI figure/table numbers are unchanged.
6. Renumbered literature citations in first-appearance order. There are 62 references: six added, with three no-longer-cited, less-direct entries removed from the bibliography (El Agente Cuantico, Virtual Lab and Robin). All original bibliography entries and the mapping are retained in `reference_mapping.csv` and the source document.

## ESI changes

- Added Section 11, "Related flow-chemistry and process-design agents."
- Added Table S36 in two panels. Panel I covers inputs, outputs and reported validation. Panel II covers quantitative engineering, equipment context and the scope of comparison with FlowPilot.
- Included FlowPilot in the same descriptive framework. An unreported capability is not scored as absent; no numerical rankings of the external systems are introduced.
- Added an interpretation subsection separating numerical consistency, laboratory implementation and measured chemical performance.
- Added ESI references 7-12. The References heading is now Section 12; existing Sections 1-10 and their numbering remain unchanged.
- Refreshed all 37 table-of-contents entries and checked their page targets against the rendered ESI.

## Formatting and preservation

- Bolded 92 figure/table labels in the main manuscript and 138 in the ESI. The formatter targets reference labels, not entire sentences. Existing bold formatting elsewhere is retained.
- Revised prose and new table content use Times New Roman, 11 pt. Existing style definitions, margins, numbering definitions and document relationships are unchanged.
- Existing embedded figure files and drawing elements are preserved exactly. No figures were recreated, replaced, resized or removed.
- Existing ESI body paragraphs and table contents remain in their original order. Table S36 is appended before References rather than renumbering existing tables.
- Original files are protected by SHA-256 checks. Every package component other than `word/document.xml` is byte-identical to its source.
- Clean and marked copies contain the same text and layout; yellow highlighting identifies substantive revisions. Bold-only formatting is not separately highlighted.

## Validation

`independent_verification.json` records the automated checks, including source preservation, figure placements, bibliography completeness, superscript citations, bold label coverage, formatter idempotence, typography, rendered page bounds and contents pagination.

The four Word files were rendered to PDF for inspection. The revised main manuscript renders to 31 pages and the ESI to 99 pages in the available LibreOffice environment. Selected introduction, comparison-table and contents pages were visually inspected; their preview images are in `rendered/`. Word on another platform may paginate differently.

Reference metadata and source/version identifiers are archived in `verified_sources.json`, `new_references.json` and `source_metadata/`. Figure/table formatting locations are logged in `manuscript_bold_references.csv` and `esi_bold_references.csv`.

## Intentionally pending

XXX/YYY experimental placeholders and the existing laboratory-review caveats are retained. This revision does not add wet-lab outcomes, certify an executable setup as safe, change benchmark scores, or claim superiority over the six external systems without a controlled comparison.

## Reproduction

From the project root, run the generator with `.venv-flowpilot/bin/python scripts/revise_prior_work_20260928.py`. Render the four outputs to the `rendered/` folder with LibreOffice. Run `.venv-flowpilot/bin/python scripts/refresh_prior_work_toc_20260928.py`, rerender the ESI, then run `.venv-flowpilot/bin/python scripts/verify_prior_work_20260928.py`. The contents refresh uses the rendered page positions, so the final verification must follow the final PDF export.
