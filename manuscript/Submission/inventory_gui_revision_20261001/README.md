# Inventory GUI and ESI Revision

Date: 1 October 2026

## Deliverables

- `../esi_submission_inventory_gui_20261001.docx`: clean revised ESI.
- `../esi_submission_inventory_gui_20261001_marked.docx`: yellow-highlighted revisions.
- `../manuscript_submission_inventory_gui_20261001.docx`: matching main manuscript. Only the inventory-method description, GUI paragraph and affected supplementary figure references were updated.
- Matching PDFs are alongside the clean Word documents.
- `figures/`: the three consolidated GUI figures.
- `screenshots/`, `evidence/`, `browser_trace.zip`, `capture_manifest.json`: real browser captures, inputs, exported inventory and capture provenance.
- `citation_audit.csv`, `figure_numbering_map.csv`, `verification.json`: reference mapping and document checks.
- `api_tests.xml`, `inventory_regression_tests.xml`, `playwright_report.json`, `browser_tests/`: test results and browser evidence.

Original documents, experimental data, benchmarks and saved KHU profiles were not overwritten. All six main-text figure images and all 25 ESI tables retain their original data. The literature comparison remains Table S1. ESI fonts, table styles and page geometry are preserved; revised text is Times New Roman 11, and only the caption labels are bold.

## Using the Inventory Editor

Open http://10.13.24.95:8510 (or http://localhost:8510 on the server), then select **Inventory**.

1. Select **New profile**, **Import JSON**, or **Open profile** beside a saved profile.
2. Enter the profile name and laboratory.
3. Select an equipment category and choose **Add equipment**, or use an existing item's pencil button. Apply the prepared fields. Additional identifiers, compatibility and limits are expandable.
4. Use **Constraints** to declare unavailable equipment and permitted alternatives. Advanced limits and shared-resource maps remain editable without discarding existing metadata.
5. Select **Validate profile** and review errors and warnings.
6. **Export JSON** downloads the complete profile. **Save profile** creates a new stored version. **Use in design** binds the validated draft to intake; it does not launch a design automatically.

The editor covers all 15 equipment categories in the pipeline schema. Optional unknown values remain unknown. Pump minimum flow and setting increment are separate. Committed edits invalidate previous validation, and unapplied advanced edits cannot be bypassed during validation/export. JSON import uses the structured parser, not an LLM. Equipment/service status and source provenance are retained.

For a later server restart, run `./flowpilot_webapp/run_dev.sh` from the project root after ensuring port 8510 is free. The currently running server is not configured as a boot-time service.

## ESI Changes

- **Figure S12** combines protocol entry, a fixed-ID follow-up question and the answered/frozen-package state.
- **Figure S13** combines category navigation, prepared pump fields, constraints, and validation/export controls.
- **Figure S14** combines the stored process topology, final stage table and an expanded council-record excerpt.

These replace the five former GUI figures S12-S16. Later ESI figures are renumbered by two; the main manuscript references and ESI contents were updated. The ESI now has 21 figures and 72 pages, compared with 23 figures and 75 pages previously. The main manuscript remains 31 pages.

The GUI demonstration intentionally uses KHU inventory version 4 and archived DPDTC run `20260907_175308_three_protocol_scientific`, matching the existing ESI example. It does not replace the later confirmed KHU experimental inventory or operating conditions. New screenshots are software-interaction evidence, not new chemistry results or benchmark outcomes. Textarea/council excerpts are identified as excerpts; uncropped screenshots and full JSON records remain available.

## Verification

- Frontend TypeScript/Vite production build passed.
- 38 webapp/API tests passed, including 8 new inventory-editor tests.
- 19 core inventory/profile/intake/document-ingestion regression tests passed.
- 28 Playwright browser tests passed across desktop and mobile; no failures. Six of these directly exercise the new editor. Sixteen optional archived-result tests were skipped because their run-specific environment variables were not supplied.
- The live capture separately reopened the actual archived topology, stage tables and council records without mocking those responses.
- KHU JSON export/re-import preserved equipment, compatibility, constraints and provenance; three identical deterministic intake requests returned the same question IDs and hash.
- 168 document checks passed, including sequential figure/table references, preserved main figures and table data, refreshed contents, clean/marked agreement, and figure captions sharing pages with their artwork. The new GUI figure pages were visually inspected after PDF rendering.

Two older inventory-resolution tests expected a historical incomplete gas-stage answer to be bypassed after equipment confirmation. Their expectations were updated to answer the now-required Q-GAS-003 explicitly, with LLM extraction disabled. No production intake gate was weakened to pass the tests.

Profile validity is not a guarantee of laboratory safety, complete equipment availability or chemical performance. No new full-pipeline chemistry generation or wet-lab experiment was performed for this revision.
