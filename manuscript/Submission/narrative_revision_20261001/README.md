# FlowPilot Narrative Revision

The revised manuscript presents FlowPilot as an agentic research assistant for
laboratory-constrained, end-to-end flow-process design. Batch protocols are the
chemical starting information, not the scope of the contribution. End-to-end
refers to the design workflow, not autonomous physical execution.

## Files to Read

- `../manuscript_submission_narrative_20261001.docx`: clean revised manuscript.
- `../manuscript_submission_narrative_20261001_marked.docx`: the same manuscript,
  with revised paragraphs highlighted yellow.
- `../esi_submission_narrative_20261001.docx`: ESI aligned with the revised framing.
- `../esi_submission_narrative_20261001_marked.docx`: highlighted ESI changes.
- `../FlowPilot_paragraph_map_20261001.docx`: two-page reading guide, one short
  bullet for each of the 56 scientific paragraphs, including Methods. Captions,
  author information and bibliography are outside this paragraph map.
- Corresponding clean PDFs are provided beside the Word files after verification.

## Main Changes

- Reworked the title, abstract and Introduction around the researcher's design
  problem, then explained FlowPilot's purpose, integrated mechanism and evidence
  strategy across several paragraphs.
- Gave each Results subsection a role in the argument: coordinating decisions,
  grounding candidate choices, retrieving suitable precedents, evaluating the
  architecture, and implementing connected chemistry.
- Expanded the architecture comparison to explain the case and model selection,
  matched information, fixed criteria, evaluator calls and statistical units.
  Interpretation now addresses differences between models, process-integration
  criteria, remaining defects, cost trade-offs and internal ablations.
- Retained unfavorable or qualifying evidence. Some simplified council settings
  slightly exceed the full configuration, and more refinement is not presented
  as uniformly beneficial. The five-model aggregate is explicitly distinguished
  from the additional archived generator visible in the preserved Figure 4.
- Added a shared experimental introduction explaining the three response sets:
  integrated final yield, conversion-focused initial timing and greater emphasis
  on throughput/compactness. Explained the convergence to two reported operating
  points per chemistry and distinguished these from experimental replicates.
- Rewrote both case narratives to connect chemistry, hardware, feed accounting,
  researcher-directed changes and measured outcomes. Main-text prose and captions
  now describe the experimental work collectively, rather than as an external
  KHU contribution. Affiliations and source/inventory identities remain intact.
- Rewrote Discussion and Conclusion as a synthesis of computational and
  experimental evidence. Detailed Methods remain in place; exact prompts,
  archived conversations, tables and laboratory records remain in the ESI.

## What Was Preserved

All six main-text figures and all 21 ESI figures are retained, including their
image bytes, display order, dimensions and cropping. All 25 ESI tables, numerical
results, bibliographies, author information, affiliations, document styles,
numbering definitions and page geometry are unchanged. The original source
documents were not overwritten. Existing institutional labels inside artwork
were not redrawn; the collective-author revision applies to manuscript prose and
captions.

Changed text uses Times New Roman 11. Main-text figure/table references remain
bold. ESI caption bodies remain regular weight. Pagination was adjusted to keep
Figure 1 with its complete caption and section headings with their following
paragraphs, without resizing images or changing the document's style definitions.

## Verification and Traceability

- `verification.json`: preservation, typography, citations, numerical evidence
  and rendered-document checks, with pass/fail results.
- `evidence_audit.json`: source CSV hashes and rechecked benchmark aggregates,
  model effects, repeat-level SDs, gas calculations and stage times.
- `citation_audit.csv`: manuscript/ESI first references and matching captions.
- `paragraph_changes.json` / `.csv`: before/after text and editorial rationale.
- `paragraph_map.json` / `.csv`: each outline bullet linked to the actual revised
  manuscript paragraph and full text.
- `source_manifest.json`: hashes of the unchanged starting documents.
- `contents_page_audit.json`: refreshed ESI contents-page targets.
- `visual_review/`: rendered page contact sheets and enlarged inspection pages.
- `editorial_strategy.md`: the narrative plan and scientific claim boundaries.

The benchmark summaries were recalculated from the archived CSVs; no scores were
changed and no new benchmark or laboratory experiments were run for this edit.
The literature comparison continues to use the existing cited sources. The
primary reports for LLM-RDF and CeProAgents were additionally consulted during
the narrative review:

- Ruan et al.: https://www.nature.com/articles/s41467-024-54457-x
- CeProAgents preprint: https://arxiv.org/abs/2603.01654

## Remaining Scientific Confirmations

This editorial revision does not supply absent measurements. The previously
identified gas-controller reference state, exact startup/pressure observations,
sample collection details, analytical calculation records, contributor-role
approval and complete archived-data release remain to be confirmed as described
in the manuscript and ESI. Neither numerical closure nor LLM judging establishes
experimental safety, optimized kinetics or universal model superiority.
