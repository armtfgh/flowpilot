# Submission revision, 29 September 2026

Original Submission files are unchanged. Review the clean DOCX files; use *_marked.docx for yellow-highlighted text revisions.

## Collaborator comments
### Experimental evidence: Addressed with remaining metadata
Replaced prospective yield placeholders with KHU-reported measurements. Added ESI Section 12, Figures S23–S31, and Tables S37–S39: lamp photographs/spectra, actual setups, original comparison schemes, procedures, product characterization, four NMR spectra, and implementation/feed reconciliation. No new experiment or model run was represented as having been performed.

### Author information: Addressed; roles await author approval
Confirmed author list and affiliation assignments used consistently. Corresponding authors: Boyoung Y. Park and Gwang-Noh Ahn. Asterisks moved after names; both email addresses included. Equal contribution of Amirreza Mottafegh and Mincheol Park retained. KHU NRF grant RS-2026-25594673 added.

### Numbers, labels, and cohorts: Addressed
Figure 1 BPR unit corrected from min to bar; illustrative value retained. Figure 4b/c rebuilt from the same five-model cohort as Tables S11–S12. Table S10 restored to all 15 archived conditions (45 outcomes), rather than labeling an eight-row subset as complete. Nine underlying campaigns per model/architecture were used to independently check score means and repeat-level SDs.

### Figures 5 and 6: Addressed
New readable chemistry/process/conditions/yield layouts. Proposed configuration is labeled separately from reported operation. NMR and isolated yields are distinguished. Shared-set entries are not called independent repeats. Full archived GUI topologies are retained in Figures S20–S21; original KHU artwork and actual photographs are included separately.

### Gas time and pressure conventions: Addressed; metrology awaits KHU
Figure 5 R2 uses the explicit nominal inlet-reference index 20/(0.020 + 0.090) = 181.82 min. This is not operating-pressure residence time. Proposal STP is 273.15 K and 1 atm; actual MFC reference conditions are unconfirmed. The 7 bar cartridge BPR is distinguished from measured system-pressure ranges.

### Literature comparison: Addressed
Added main-text Table 1, condensed from Table S36, covering design task/output, equipment and stage scope, and reported validation for six related systems and FlowPilot. The comparison is descriptive, not a head-to-head performance ranking. Full Table S36 retained and updated with laboratory results.

### Repetition and traceability: Addressed
Replaced the old empty reporting templates, consolidated experimental procedures in Section 12, preserved source records, and clarified finite sample-loop operation. Existing figures and tables were retained except the explicitly targeted main-text revisions; all original embedded media remain archived in the packages.

### Code and data links: Addressed; data release incomplete
Added the confirmed public GitHub repository, benchmark-code link, and data-access index. Public repository/API access was checked on 29 September 2026. There are currently no release tags; ablation_results contains an index README rather than the complete outcome archive. No unprovided data DOI or release was invented.

## Outstanding confirmations
1. **Quantitative analytical records (KHU)**: Supply raw quantitative-NMR files and integration worksheets, the amount and purity of 1,3-benzodioxole, dilution/calculation details, and sample identifiers. The supplied spectra are product-characterization spectra, not the complete yield-calculation records. Original NMR/LRMS exports and final approval of reported assignments are also needed.

2. **Independent repeats (KHU)**: Confirm independent run counts and individual yields. Figure 5 Sets 1/3 and Figure 6 Sets 2/3 are grouped in the supplied result tables. No experimental SD or significance claim can be made from those grouped entries alone.

3. **Sample-loop and collection timing (KHU)**: Give sample injection and collection windows, carrier/stock transitions, dispersion handling, and benzylamine-feed start/stop timing and total volume. The 2 mL loop corresponds to 100 min injection for Figure 5, and 12.50 or 7.14 min for Figure 6; these are not the reactor residence times.

4. **Gas/pressure and backflow records (KHU)**: Confirm the MFC reference temperature/pressure and calibration; pressure-sensor position and gauge/absolute convention; check-valve branch/orientation; oxygen-free feed preparation; and startup/backflow observations. The proposal STP calculation must not be confused with a measured physical residence time or safety validation.

5. **Protocol and dimensional confirmation (KHU)**: Confirm the 95 °C water-bath description and actual tubing IDs (0.04 in = 1.016 mm; 0.093 in = 2.3622 mm). The original literature comparison uses 180 min for Giese Stage 1 versus 240 min in the supplied design prompt; amidation literature batch DPDTC is 1.10 equiv versus 1.05 in the input/current experiments. These differences are explicitly retained, not silently reconciled.

6. **Submission administration (All authors / corresponding authors)**: Approve the final CRediT roles, author spelling, affiliations, correspondence emails, funding, and any conflict-of-interest statement. Publish a fixed software release and complete data archive (or documented reviewer-access package), with a persistent identifier and updated availability text. These items cannot be completed by inference.

## Audit
Automated checks: 65/65. See verification.json, cross_reference_audit.csv, rendered_page_map.json, and visual_review/.
No new model benchmark or wet-lab experiment was performed during this revision.
