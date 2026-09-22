# Manuscript and ESI revision - 22 September 2026

## Deliverables

- `../manuscript_revised_20260922.docx`: revised manuscript.
- `../esi_revised_20260922.docx`: revised ESI.
- Corresponding `_marked.docx` copies: newly revised/added text highlighted yellow. Existing highlighting in the source documents is retained.
- `figures/figure5_prototype.*` and `figure6_prototype.*`: PNG (600 dpi), PDF and SVG prototypes. Their measured-yield strips intentionally contain XXX or YYY, not invented observations.
- `figures/figureS20*` and `figureS21*`: separate full-resolution topology panels for the three response sets in each case.
- `figures/figureS22_preprint_figure6_exact.png`: exact original preprint Figure 6, including transparency.
- `rendered/`: PDF review copies and visual-check images.

The source manuscript, ESI, CV, preprint and archived design runs were not overwritten. No new design benchmark or laboratory experiment was performed for this revision.

## Completed Changes

1. Added the requested author order to both documents. Amirreza Mottafegh and Mincheol Park are identified as co-first authors. The additional name and final affiliations remain for author confirmation; no contributor roles were invented.
2. Expanded the Figure 5 discussion around radical addition, stage-specific oxygen exclusion/delivery, compatible irradiation modules, gas molar feed, and the distinction between an inlet/STP time index and actual contact time.
3. Expanded Figure 6 around activation to a thioester, downstream aminolysis, cumulative flow, concentration-dependent stoichiometry, premix stability and connected-process evaluation.
4. Replaced only the Figure 5/6 placeholders with new prototypes using actual archived GUI topologies. Figures 1-4 remain unchanged.
5. Added ESI Section 8 with exact supplied batch protocols and fixed-ID responses, source/run identities, full stage and component-feed tables, and experimental completion templates: Figures S20-S21 and Tables S29-S34.
6. Added the exact preprint bromination artwork as Figure S22 in Section 9. Its historical predictions and unresolved qualifications are distinguished from the current amidation case and from measured results.
7. Added Section 10 and Table S35: major dated pipeline milestones, reasons, evidence bases and version boundaries. Later scientific-policy and backflow tools are not retroactively attributed to the earlier benchmark.
8. Added four verified CV publications: multistep apixaban optimization, integrated photochemical reaction/separation, LLM-assisted auto-reactometry, and language-guided optimization priors. Publisher metadata supersedes two outdated CV entries. The separate bromination source was also added.
9. Renumbered the main bibliography by first citation and updated all numerical citations. The main text cites all 22 ESI figures and 35 ESI tables in sequence. The ESI contents were refreshed from the rendered document.

## Preservation and Validation

- Final automated audit: **104/104 checks passed**. Rendered review copies contain 27 manuscript pages and 94 ESI pages.
- Source Word style definitions, numbering definitions and page geometry are preserved. New body text and tables use Times New Roman, 11 pt.
- All six original main-document image assets and all 22 original ESI image assets are retained byte-for-byte. The main document displays the same Figures 1-4; only the two designated placeholders are replaced.
- Each newly embedded topology is checked against its own source PNG. An identically named `topology.png` packaging collision found during independent review was corrected before delivery.
- Figures 5 and 6 and their captions were checked together on rendered pages. New tables have left-aligned cells, repeated column headings where required, and protected rows.
- Volume/flow/time and concentration/flow/molar-feed calculations were independently recalculated from the archived CSVs. This confirms arithmetic, not experimental conversion or hydraulic safety.
- The original preprint Figure 6 was checked pixel-for-pixel, including its alpha channel.
- See `verification_results.json`, `document_audit.json`, `main_to_esi_citation_audit.csv`, `contents_page_audit.json`, and `reference_verification.json` for the detailed checks and source links.

## Experimental and Submission Items Still Open

- Replace XXX/YYY only with verified experimental results, including the yield basis, analytical method, actual operating conditions, sample timing and true replicate count. No standard deviations should be inferred from the three response sets.
- Figure 5 Sets 1/3 share the same reported settings; Figure 6 Sets 2/3 also coincide. They are response-set variants, not automatically independent experiments.
- The Figure 5 proposal still needs pressure/backflow review. Pure oxygen at lower volume and a gas-line check valve do not establish protection of the upstream liquid branch. Illustrative transient-tool inputs are not approved laboratory settings or calibrated failure probabilities.
- Confirm feed solubility, activation-premix stability, component ratings, interstage handling and the eventual workup/analytical procedure with the laboratory.
- Finalize the blank author, exact author names/titles, affiliations, contributor roles and author approval.
- The prototype figures include abbreviated chemical formulas and archived detailed topology labels. Final wet-lab panels and journal-specific artwork sizing remain to be completed after the measured results are available.

## Rebuilding

Use `.venv-flowpilot/bin/python` for the three revision scripts and the figure builder. Build figures with `scripts/build_case_figures_20260922.py`, then build documents with `scripts/revise_manuscript_cases_20260922.py`. Render both Word files to PDF with LibreOffice, refresh the ESI contents using `scripts/refresh_revision_toc_20260922.py`, and render again. Repeat the contents refresh if pagination changes. Finally run `scripts/verify_manuscript_revision_20260922.py`.

The scripts are restricted to the dated deliverable files and revision folder; they do not change the original manuscript or the FlowPilot design pipeline.
