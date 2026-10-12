# October 1 Postmeeting: Second Review

## Revised Files

- `../FlowPilot_manuscript_october1_postmeeting_round2.docx`
- `../FlowPilot_ESI_october1_postmeeting_round2.docx`
- Yellow-highlighted review copies: `marked/`.
- Rendered reading and review PDFs: `rendered/`.

The previous postmeeting documents were not overwritten. This pass revises the narrative and methods against the code, archived benchmark records and supplied experimental material. It does not generate new benchmark outcomes, alter scores, infer missing measurements or replace the existing figures.

## Substantive Corrections

1. **Intake reproducibility.** The ESI previously described the LLM as selecting questions and omitted the conditional question bank. It now distinguishes deterministic question activation/readiness from LLM extraction. Table S3 includes all eight core and five conditional IDs, their activation conditions, permissible unavailable answers and design consequences. Readiness is not equated with equipment feasibility or authorization to operate.
2. **Missing benchmark inventories.** Each case now has a readable equipment/limit summary. The exact archived case JSON, inventory JSON and complete design-input string are exported in `frozen_benchmark_inputs/`, with hashes and original paths. Protocol, objective, hard constraints and inventory have matching canonical hashes across the three campaign sources supplying the five reported generators. These computational inventories are explicitly distinguished from the experimental laboratory profile.
3. **Benchmark provenance.** The campaign record says the three cases were selected after a five-case pilot. The text now calls this a retrospective development-stage comparison, not a prospective unseen-reaction test. The unsupported prospective stopping-rule statement was removed. Matching task inputs does not remove case-selection or development exposure effects.
4. **Ablation interpretation.** All four individual domain-reviewer removals had lower means than the full condition. The old ESI wording incorrectly suggested otherwise. The corrected explanation preserves the higher one-pass and no-preselection means and does not infer a causal ranking from one outcome per case-condition pair. Table S9 retains every condition and score, defines calls/flags, and removes the architecture-specific execution-status column from the comparison. That field remains in the raw archive.
5. **Criteria and critical flags.** UO-08 applies to the gas-using cases but not CuAAC; UO-09 is inapplicable to all three single-reactive-stage cases. The caption now explains these distinct reasons. The 11-row critical-flag map omits zero-flag UO-01/UO-14 and inapplicable UO-09; all 102 judge flags remain represented. Flags are judge-level findings, not a count of independently verified physical failures.
6. **Engineering definitions.** Nominal liquid space time, the inlet/STP index, pressure-corrected no-slip estimates and measured phase-contact time are distinguished. Table S12 defines gas equivalents through molar flows, includes active gas fraction, states the STP molar volume, and prevents mixed-basis stage sums from being described as physical residence time. The text explicitly identifies dissolution, consumption, slip and axial-pressure assumptions. Finite-loop concentrations remain nominal rather than measured throughout the sample plug.
7. **Scientific interpretation.** The main text keeps the end-to-end, equipment-aware process-design narrative. The module-screen and amidation interpretations are tied to the observed data, without presenting a small descriptive comparison as identification of a causal mechanism or rate-limiting step.
8. **Reference repair.** The amide characterization still cited ESI references [11-12] after earlier renumbering. This is corrected to [12-13]: Saunders and Wang, not the unrelated Park photochemistry paper. The bibliography itself is preserved.
9. **Formatting and navigation.** The missing engineering subsection styles are restored, including Sections 5.3-5.6 in the refreshed contents. Revised text uses Times New Roman 11. Caption bodies remain regular weight. Empty spacer paragraphs and an unnecessary figure-page break are removed. Existing figures, image sizes, spectra, author information and experimental/response tables are preserved.

## Evidence Rechecked

- 90 retained model/architecture outcomes: 45 per architecture, three cases and three repeats for each of five generators.
- Mean scores: one-shot **0.794**; FlowPilot **0.917**.
- Judge-level critical flags: one-shot **98**; FlowPilot **4**. Outcomes with no flag: **18/45** and **43/45**, respectively.
- Means and sample SD reconstructed from the three repeat-level means, each averaging the three chemistry cases. This is not the SD of nine pooled outcomes.
- Internal screen: 45 outcomes, 15 conditions and three cases; full **0.928**, no council **0.670**, one-shot **0.743**. Its SD represents between-case variation, not repeatability.
- Experimental arithmetic: oxygen **2.01 equivalents** under the proposal STP convention; second-stage inlet-reference index **181.82 min**; amidation nominal stage times **31.25/25.00 min** and **35.71/14.29 min**.
- Literature yield labels checked against the supplied experimental comparison graphics: photochemical flow reference **90% NMR yield** and amidation batch reference **98% NMR yield**. No new independent source-paper verification is claimed in this review.

Full check results are in `verification.json`; numerical recomputation in `evidence_audit.json` and `module_recalculation.json`; figure/table references in `citation_audit.csv`; exact edits in `text_change_log.json`. These checks establish the stated document/data invariants, not experimental validity or publication readiness.

## Still Requires Resolution

1. **CuAAC benchmark volume basis.** The frozen inventory lists a 0.30 mL reactor and a 4.0 mm-ID, 60 mm cartridge body, whose empty cylindrical volume is 0.754 mL. An accessible liquid volume or shorter active section might explain the difference, but neither is specified. The original input and scores are retained, and the ambiguity is disclosed. Before treating this as an unambiguous numerical-closure benchmark, clarify the source basis and assess whether affected UO-07 judgments need a separately versioned sensitivity analysis. Do not silently amend the old data.
2. **Photochemical benchmark source status.** The archived manifest labels this case `machine_extracted_unverified`. Matching frozen inputs is verified; agreement of every extracted fact with the source paper is not. A page-level source check is still needed before presenting its oracle as verified ground truth.
3. **Existing Figure 1 artwork.** The retained image lacks a visible panel-(b) label and uses an approximate handbook count that differs from the six-source audited export. The caption distinguishes illustration from audited configuration; the final native artwork should correct those two labels. This review deliberately does not replace the figure or invent a native source.
4. **Experimental metadata.** Outstanding items remain: raw quantitative-NMR calculations and internal-standard details; individual experimental identifiers/repeats; collection windows and second-feed timing; MFC reference/calibration; pressure basis/sensor location; oxygen-free feed preparation; check-valve branch/orientation; and confirmation of reported apparatus details. Current yields do not establish reproducibility, steady production or a backflow probability.
5. **Release and author approvals.** Confirm contributor roles and the final code/data release, complete outcome archive and persistent data identifier. The manuscript does not substitute the public code index for an accessible full dataset.

The revision is more defensible and reproducible, but those source/measurement and artwork items should remain visible on the submission checklist.
