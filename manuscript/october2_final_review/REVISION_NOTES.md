# October 2 Final Real1

## Files

- `../FlowPilot_manuscript_october2_final_real1.docx`
- `../FlowPilot_ESI_october2_final_real1.docx`
- Rendered review PDFs are in `rendered/`.

The two `october2_notation_corrected` source documents are unchanged. Deletion requests below refer to the numbering in those source documents, not the renumbered output.

## Latest Requested Changes

- Removed original Figure S4 completely, including its embedded image, caption and obsolete references.
- Removed original Tables S4, S8, S9, S17 and S18 and repaired their cross-references.
- Restricted original Table S7 to the seven conditions actually shown in main Figure 4d, in the same order. This is now **Table S6**. Scores, SDs, calls and flags are unchanged. All 15 screened conditions remain in the archived CSV; the text explicitly identifies this as a selected display, not the complete screen.
- Changed original Table S11 to describe the **used inventory**, without institutional shorthand. It is now **Table S8**.
- Removed institutional shorthand in prose and figure labels. Correct author affiliations and the corresponding author's actual university email are retained.
- Removed decision-counter and voting-related content from the displayed council extracts. These are explicitly described as edited extracts; substantive criticisms, numerical errors, uncertain assumptions and the original unedited records are retained.
- Restored the major development timeline in **Section 9, Table S12**. Snapshot dates are evidence anchors, not claims of controlled performance improvements.
- Renumbered the ESI to Figures S1-S27 and Tables S1-S12. Updated main-text and ESI references and the contents page. `citation_audit.csv` records first citations; `numbering_map.json` maps retained source numbers to revised numbers.

## Earlier Meeting Requests Included

- Re-captured the actual GUI front page and repositioned the external annotations. The screenshot uses an unsaved inventory copy with neutral display labels; equipment identifiers and numerical capabilities are unchanged. No design-generation call or saved-profile overwrite occurred.
- Re-laid out the inventory screenshot panels with more padding. The underlying screenshots are unchanged.
- Inserted the supplied `esi_fig` and `esi_fig2` as Figures S8 and S9 and cited them in the main manuscript. Their historical heuristic QA scores are explicitly distinguished from the fixed-criteria benchmark scores. Prompt lengths are not claimed to measure actual human effort.
- Restored the original experimental vector schemes and comparison tables as Figures S20 and S22, retaining the supplied chemical artwork and embedded source objects. They replace the duplicate reconstructed Tables S15 and S16. Setup photographs, spectra and all six proposed topologies remain.
- Expanded the Figure 5/6 narrative around candidate review, objective-dependent choices, stage coupling and feedback from implementation. Researcher-directed oxygen selection is not represented as an autonomous model discovery.
- Condensed repeated explanatory and provenance prose without deleting the actual inputs, rubric, detailed council assessments, or experimental evidence.
- Retained scientific subscripts/superscripts, the original document styles and page settings. Caption explanations are regular weight; labels and in-text figure/table references are bold.
- Main Figures 1-4 remain unchanged. Figures 5-6 retain their layout and numerical data; only the institution-specific labels were generalized.

## Verification Record

`quality_audit.json` records the final package, cross-reference, data-preservation and rendered-page checks. `contents_page_audit.json` records contents-page locations. `page_previews/` contains full-document review sheets and figure/table pages. The scientific-typesetting test suite passed all 12 focused tests.

## Scientific Details Still Awaiting Confirmation

These were not resolved by editorial changes or by inventing values:

- The original photochemical experimental comparison gives 180 min for the literature batch first step, whereas the supplied agent input gives 4 h. Both source-specific statements are identified explicitly.
- The experimental graphic's rounded second-stage index of 181 min corresponds to 181.82 min using the reported inlet-reference flows. It is not a measured operating-pressure residence time.
- The photochemical literature batch entry marked 95% is an isolated yield. It is not relabeled as NMR yield. Main Figure 5 instead compares the 90% literature flow NMR yield.
- Original nominal tubing labels and independently specified inch dimensions differ slightly; the distinction is recorded rather than silently altering the original artwork.
- MFC reference conditions, pressure-reference details, startup/check-valve documentation, quantitative-NMR worksheets, collection windows and replicate information remain as specified in ESI Section 8.7.
- The complete public data deposit, tagged manuscript release and final author contribution approval still require completion.

## Source Records

- Full module screen: `manuscript/Submission/overhaul_20260930/source_data/tableS10_all15_module_conditions.csv`.
- Original experimental document: `manuscript/Supplementary Information (KRICT)_final (1).docx`.
- Unabridged council report: `manuscript/council_discussions_figures5_6_20261002/FlowPilot_Figures5_6_full_council_reports.docx`.
- Historical timeline: `manuscript/Submission/overhaul_20260930/companion_archive/pipeline_history.txt`.

Original filenames in this audit record are provenance references; institutional shorthand has been removed from the manuscript/ESI narrative, not falsified in archived source records.
