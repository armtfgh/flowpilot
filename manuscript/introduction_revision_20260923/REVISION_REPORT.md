# Introduction revision, 23 September 2026

## Deliverables

- `../manuscript_introduction_revised_20260923.docx`: revised manuscript.
- `../manuscript_introduction_revised_20260923_marked.docx`: identical manuscript with the new introduction, new bibliography entries, and changed citation numbers highlighted yellow.
- `rendered/`: PDF proofs and visual inspection images.
- `new_references_and_claims.csv`: source-by-source claim and scope audit.
- `reference_renumbering.csv`: complete old-to-new reference mapping, including removed entries.
- `verified_references.json` and `source_metadata/`: cached publisher-deposited Crossref records and version-specific arXiv metadata.
- `verification.json` and `independent_verification.json`: preservation and document checks.

Source: `../manuscript_revised_20260922.docx`. This file and both matching ESI copies are unchanged. Only the introduction and literature bibliography were substantively revised; superscript reference numbers were updated throughout the manuscript. All figures, captions, tables, results, author information, experimental placeholders, and main-text/ESI cross-references were preserved.

## New introduction

The introduction has been rewritten as nine paragraphs, approximately 1,105 words. Its argument progresses through:

1. Why translating a batch procedure requires a connected process specification.
2. Coupled stream, stage, transport, and equipment decisions.
3. Existing flow automation, route planning, simulation, and optimization.
4. Tool-using chemistry agents and computational scientific workflows.
5. Recent multi-agent discovery and hardware-aware procedural systems.
6. Tested scientific tools, simulation feedback, and explicit execution state.
7. Why design-level evaluation must go beyond fluent answers or aggregate chemical knowledge.
8. FlowPilot's specific contribution: intake, engineering calculations, specialist review, inventory reconciliation, and a shared process record.
9. The distinction between retrieval evidence, computational benchmarking, and prospective wet-laboratory evaluation.

The contribution is positioned around equipment-constrained batch-to-flow proposal construction, not a claim to have invented multi-agent science, hardware-aware protocol translation, or autonomous flow optimization. The introduction does not claim that more agents necessarily outperform other architectures, that the comparison is equal-compute, or that benchmark scores establish yield or safety.

## Added literature

The search was bounded by the revision date, 23 September 2026. Thirteen references were added: eleven from 2026 and two relevant 2025 foundations. Eleven are journal research articles and two are explicitly labeled preprints. Four previously introduction-only references became uncited and were removed from this revised bibliography, which now contains 59 entries. Their complete records remain in the source manuscript and mapping file.

| New reference | Study | Venue/status | First online or cited version date |
|---|---|---|---|
| 36 | RoboChem-Flex | Nature Synthesis, research article | 13 April 2026 |
| 41 | El Agente Cuantico | Reports on Progress in Physics, research article | 31 July 2026 |
| 42 | Virtual Lab | Nature, research article | 29 July 2025 |
| 43 | Co-Scientist | Nature, research article | 19 May 2026 |
| 44 | Robin | Nature, research article | 19 May 2026 |
| 45 | ACRA | Communications Chemistry, research article | 3 April 2026 |
| 46 | AutoLabs | Scientific Reports, research article | 25 June 2026 |
| 47 | Paper2Agent | Nature, research article | 16 September 2026 |
| 48 | PRISM | Digital Discovery, research article | 20 May 2026 |
| 49 | El Agente Grafico | arXiv preprint, v2 | 7 August 2026; initially 19 February |
| 50 | La Agente Optima | arXiv preprint, v1 | 3 September 2026 |
| 51 | ChemBench | Nature Chemistry, research article | 20 May 2025 |
| 52 | Uncertainty-calibrated experimental optimizers | Nature Machine Intelligence, research article | 28 August 2026 |

Exact titles, author lists, identifiers, source links, and claim limitations are in `verified_references.json` and `new_references_and_claims.csv`. Dates above distinguish first online publication from later issue assignment. The bibliography uses available volume/pages or article numbers; RoboChem-Flex and Paper2Agent remain advance online publications where the checked records do not yet provide final pagination. The accented Spanish names are retained in the Word documents.

The two Aspuru-Guzik preprints are acknowledged as relevant prior art, not presented as peer-reviewed journal articles. The published El Agente Cuantico reference provides an additional verified 2026 example from that group. The previously cited El Agente and Materealize remain in the bibliography; Materealize retains its arXiv identification. All four CV-derived references added in the preceding revision were retained.

Removed old reference numbers: 1 (general flow-chemistry editorial), 35 (general LLM-agent review), 42 (general engineering-AGI preprint), and 43 (requirements-engineering study). Their former supporting role is now served by more direct primary research. No other original bibliography entry was deleted.

## Verification

- **37/37 independent document checks passed.**
- All eight embedded media assets, image relationships, styles, numbering definitions, margins, headers, footers, and other non-document package parts were preserved byte-for-byte.
- Outside-introduction prose and paragraph layout properties are unchanged, apart from numerical literature citation updates.
- All 59 references are cited and numbered in first-appearance order; no dangling citations or draft delimiters remain.
- New introduction text uses Times New Roman, 11 pt. The marked copy highlights every revised introduction run and the added references.
- The revised and marked copies contain identical text and layout properties, differing only in highlighting.
- Both Word documents were rendered with LibreOffice: 29 pages each. Automated checks found no text outside the page bounds and no blank pages.
- Introduction pages 1-4, the final bibliography pages, and a marked introduction page were visually inspected. No clipping, overlap, broken accent characters, or lost superscript citations were observed in those views.

The longer introduction and bibliography necessarily change pagination; no figure was replaced, redrawn, or removed. Existing highlights outside the revised material, where present in the source, were retained rather than silently cleared. The ESI and the wet-laboratory yield placeholders were not modified.

## Reproduction

From the project root:

```bash
.venv-flowpilot/bin/python scripts/revise_introduction_20260923.py
libreoffice -env:UserInstallation=file:///tmp/flowpilot_intro_20260923 --headless --convert-to pdf --outdir manuscript/introduction_revision_20260923/rendered manuscript/manuscript_introduction_revised_20260923.docx manuscript/manuscript_introduction_revised_20260923_marked.docx
.venv-flowpilot/bin/python scripts/verify_introduction_20260923.py
```

Metadata is reused from the saved source records for reproducibility. No changes to FlowPilot software, benchmarks, measured results, or ESI were made in this revision.
