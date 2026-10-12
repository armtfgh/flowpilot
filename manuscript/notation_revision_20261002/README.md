# Scientific notation revision, October 2, 2026

## Deliverables

- `../FlowPilot_manuscript_october2_notation_corrected.docx`
- `../FlowPilot_ESI_october2_notation_corrected.docx`

These are new copies of the October 2 corrected manuscript and the latest
council-extended ESI. Original documents remain unchanged.

## Scope

Editable scientific text now uses Word subscripts and superscripts for
quantity indices, chemical formulas, isotopes and unit exponents. Quantity
symbols are italic and descriptive subscripts are upright. Existing equation
structure is retained. Reactor volume, liquid holdup and inlet/STP time indices
remain distinct; no numerical results or operating conditions were changed.

The ESI includes a notation convention and a disclosure that scientific
typography in reproduced council reports has been normalized. Literal software
fields, paths, JSON and the separate raw council archives remain unchanged.
Embedded artwork is retained byte-for-byte; text embedded in images was not
redrawn. Existing fonts, styles, tables, captions and numbering are preserved.

## Verification

- `manifest.json` records sources, hashes, outputs and edit counts.
- `manuscript_changes.json` and `ESI_changes.json` record paragraph edits.
- `verification.json` contains document-integrity and rendered-page checks.
- `contents_page_audit.json` records the updated ESI contents-page references.
- `rendered/` contains PDFs for inspection; `visual_check/` contains page images.
- Twelve focused notation tests cover split Word runs, chemical formulas,
  unit exponents, citations, literal identifiers, idempotence and equations.
  NMR isotope superscripts are distinguished from baseline peak integrations.

The convention for subsequent revisions is in `../SCIENTIFIC_NOTATION.md`.
