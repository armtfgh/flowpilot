# Text-only manuscript revision

Start with `manuscript_text_revised.docx`. It was made directly from the original `manuscript.docx`, not from the discarded revision.

## Preserved

- Original Figure 1, Figure 2, Figure 3, and Figure 4 image bytes and Word drawing properties.
- Original document styles, fonts, sizes, numbering, page settings, margins, headers, and footers. No global formatting pass was applied; existing highlights remain as in the source.
- The complete original `esi.docx`, including every figure and table. No ESI revision was made.
- Original source files. The rejected `revision_20260917` folder and its four generation scripts were deleted.

## Text changes

- Aligned the introduction, Results, Discussion, conclusion, and Methods with the existing architecture, retrieval, and ablation figures.
- Separated Figure 4's fixed-criteria LLM-judge benchmark from Figure S12's historical engineering radar and model-matrix study.
- Explained the score, critical-flag counts, different SD definitions, and the scope of the five-model ESI cohort. Figure 4's additional GPT-5.4 bars were preserved, not removed; the caption explicitly distinguishes their scope.
- Added references to the ESI's existing intake, inventory, actual GUI, failure-example, and council-discussion material. All 19 ESI figure numbers and 28 table numbers are cited, in ascending first-appearance order.
- Replaced the obsolete Figure 5/6 result narratives with one short paragraph per chemistry and simple diagram placeholders. No wet-lab results were invented. The originals remain in the unchanged source manuscript.
- Replaced reference 43 with the DPDTC paper and corrected the Zeng citation to reference 36, retaining the bibliography's existing formatting. DPDTC bibliographic details were checked against [the ACS article](https://pubs.acs.org/doi/10.1021/acssuschemeng.5c00914).

## Existing issues deliberately not redrawn or changed

1. Figure 1's illustrative recipe card labels BPR with a time unit (`12 min`). The caption now makes clear that the card is schematic, but the artwork was not altered.
2. Figure 2(c) displays a material denominator of 431, whereas the ESI's categorical export uses 464. The revised text avoids deriving percentages from that conflicting denominator. The figure needs an author decision before submission; it was not silently replaced.
3. The ESI Table S10 caption says all 15 conditions, but the actual Word table contains eight. The main text describes the displayed summaries without claiming that this unchanged table contains all 15.

These issues cannot be removed through prose alone while also preserving the figures and ESI exactly. They are recorded here rather than used as a reason to redesign either document.

`text_revision_checks/preservation_checks.json` records the package-level checks. Only `word/document.xml` and the two Figure 5/6 placeholder image entries differ inside the revised DOCX. `text_changes.json` and `esi_cross_references.csv` provide the text-change and citation records. PDF renders in the check folder are for layout review; text edits necessarily change pagination even though page and paragraph formatting are preserved.
