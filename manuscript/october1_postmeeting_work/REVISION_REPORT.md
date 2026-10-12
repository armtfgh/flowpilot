# October 1 Postmeeting Revision

## Deliverables

- `../FlowPilot_manuscript_october1_postmeeting.docx`: clean manuscript.
- `../FlowPilot_ESI_october1_postmeeting.docx`: clean ESI.
- `marked/`: matching Word review copies with revised passages highlighted yellow.
- `rendered/`: PDF reading copies and page contact sheets.
- `figures/`, `gui/`, `raw/`: restored/new figure assets, actual browser capture and underlying data.

The two supplied `__20261001.docx` files were not overwritten. Their hashes are recorded in `source_manifest.json`.

## Requested Changes

1. **Experience-based design.** Expanded the Introduction to explain the practical judgments involved in flow chemistry. Literature retrieval supplies precedents, handbook rules supply principles and cautions, and LLM reasoning interprets their relevance. This is explicitly documented experience, not a claim to replace experimental judgment or to infer validated kinetics.
2. **Formatting.** Removed inconsistent direct bold in body paragraphs. Caption labels remain bold, caption bodies regular, and legitimate headings/table headers are retained. Clean copies have no inherited yellow highlighting; separate marked copies support review. Existing document styles, font tables, numbering definitions and page setup remain unchanged.
3. **Figure 4c.** Explained how critical flags are counted, model-dependent differences and their chemical/operational significance. Added traceable pressure-boundary, residence-time and catalyst-placement examples, including the evaluator's gas-unit false positive. No scores or archived counts were changed.
4. **Figure 4d.** Expanded interpretation of council removal and individual specialist removals. Explained overlapping responsibilities, the limits of causal attribution and why the error bars represent between-case variation, not replicated module uncertainty.
5. **Old Section 3.2 and duplication.** The supplied file's Section 3.2 was the engineering-rule subsection; Section 3.3 contained the workflow overlapping main-text Figure 3. Removed the old 3.2 narrative and its redundant category-count table, and removed the duplicate retrieval-workflow subsection/figure. Retained Figure S4(a-c) as a compact data audit and the operational settings needed for reproducibility. The surviving retrieval-evaluation subsection is now numbered 3.2; it is not the former subsection.
6. **Figure S4d.** Removed the concept co-occurrence network. Replotted the three retained panels from the original CSVs, with a shared category order in (a-b), separated labels/legend and an explicit interpretation of coverage versus validation.
7. **Benchmark case selection.** Explained the distinct catalyst/thermal constraints of CuAAC, photochemical oxygen bookkeeping of methionine oxidation, and three-phase contact/pressure requirements of hydrogenolysis. Distinguished a controlled stress test from representative coverage of all flow chemistry.
8. **Response sets.** Replaced the repeated paragraphs with two tables: **Table S15** for Giese/oxidation and **Table S16** for DPDTC amidation. Shared answers are listed once; the substantive wording of all 21 set/question answers is retained. Only form prefixes and duplicated punctuation were removed. The later pure-oxygen override is identified separately.
9. **GUI screenshot.** **Figure S10** is one actual full-front-page browser capture with six numbered module annotations and arrows. It shows protocol entry, fixed-ID questions, model choices, inventory binding and the readiness gate. No design-generation call was made for this capture. The accepted inventory-management figure is preserved unchanged, now **Figure S11**.
10. **Computational dry run.** Restored the exact former preprint Figure 6 as **Figure S12**, with new **Section 6.4** and a main-text reference. Its mixing warnings, conflicting flow/BPR statements and legacy validation label are disclosed; it is not presented as a new experiment or a current executable design.
11. **Six proposed topologies.** Restored Giese/oxidation Sets 1-3 as **Figures S13-S15** and amidation Sets 1-3 as **Figures S16-S18**. These use the existing horizontal, icon-based renderings. Readable text keys reproduce stage volumes, incoming flows, time basis, temperature, feed concentrations and molar feeds from the archived CSVs. Actual operating conditions and yields remain in the experimental section, not inferred from the proposals.
12. **Cross-references and editing.** Renumbered the ESI to Figures S1-S25 and Tables S1-S20, updated main-text callouts and refreshed the contents. Removed a stale Section 7.5 pointer. Restored an appropriate citation to existing reference 56, which was uncited in the supplied manuscript, without renumbering the main bibliography. Added the bromination source as ESI reference 10, before the experimental-case references, and updated the affected ESI citations in first-appearance order.

## Preservation and Verification

- All six main-text figure images are unchanged, including Figure 4; their drawing size, crop and positioning are unchanged.
- The accepted inventory screenshot, experimental apparatus images and product spectra remain unchanged.
- Experimental operating/yield tables and component-feed values remain unchanged.
- All main-text ESI figure/table callouts resolve, cover the full set and follow ascending first-citation order.
- All 61 main-text literature references remain cited in order.
- All twelve restored stage keys close numerically on their stated liquid or inlet/STP basis. The gas-based index is not relabeled as residence time under operating pressure.
- Browser capture recorded no page errors and no design-job submission. The temporary capture server was stopped; the user's existing server was not stopped or changed.
- Word packages were rendered with LibreOffice. The contents-page numbers were refreshed to convergence. Page bounds, blank pages, figure relationships and caption formatting were checked, with visual inspection of the revised figures, response tables and manuscript discussion pages.

Machine-readable checks: `verification.json`. Cross-reference list: `citation_audit.csv`. Exact edits: `text_change_log.json`. Source-to-final numbering: `numbering_map.json`.

## Remaining Scientific Boundaries

This revision does not add experimental measurements, repeat benchmarks or resolve the outstanding laboratory metadata already listed in the ESI. The historical bromination figure is deliberately archival and contains unresolved inconsistencies; it should not be described as proof of a validated or safe design. Small computational score differences and the wet-lab operating-point comparisons remain descriptive, not statistically resolved causal effects.

Reference-56 citation checked against the publisher's abstract: https://doi.org/10.1021/acssuschemeng.5c00580 (October 1, 2026).
