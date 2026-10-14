Source-declared component alias repair — 14 October 2026

Status
The complete practical Claude benchmark was finished and verified before this
production repair was applied. This repair is UNSCORED. It did not alter frozen
generation results, judge packets, judge results, or benchmark summaries. No API
calls or experimental measurements were made for this repair.

Observed defect and cause
The fresh hydrogenolysis result contained DMAOL and
N-diphenylmethylazetidin-3-ol (DMAOL) as separate liquid-feed entries. Its original
protocol explicitly declares these names to identify one compound. The literal
PREP-A instruction therefore charged 1.26 mmol twice into 10 mL, giving 2.52 mmol
instead of the intended 1.26 mmol. Both entries came from one feed and one source
charge, not independent portions. The defect was recorded in the independent
manual audit before judging.

The chemistry reconciliation compared literal names when restoring missing
initial-charge species. An abbreviated feed name and a full plan name therefore
looked like different species. The artifact compiler retained both names and
assigned each the intended substrate concentration.

Repair
flora_translate/source_component_aliases.py obtains identity authority only from
an exact full-name (ACRONYM) declaration in the original isolated protocol. It
does not use inventory descriptions or a model-invented acronym as authority.
Chemistry reconciliation normalizes source aliases before restoring missing
initial-charge species. Conflicting plan role/loading declarations remain intact.
The final artifact compiler collapses matching alias descriptions only within the
same feed, with matching quantities and compatible roles. It retains explicit
portions, independent feeds, conflicting quantities, and conflicting roles. Every
collapse carries the exact source quotation and the discarded description as
provenance. Ambiguous or unverified alias assertions remain visible and block
release. The topology builder now passes original protocol context into the same
component compiler, so pump labels and preparation instructions use one identity.

Files
- source_declared_alias_fix.patch contains only this repair's production changes
  and dedicated regression tests. It has already been applied to live source.
- patch_source_hashes.json records before/after hashes for the isolated patch.
  Other agents subsequently made independent production changes, so these are
  patch-specific hashes, not a claim that all live files still match them.
- staging/ is the isolated overlay used before the source freeze ended.
  Its __init__.py is test scaffolding and is deliberately absent from the patch.
- verify_saved_recipe.py reproduces the saved hydrogenolysis example read-only.
- saved_hydrogenolysis_recipe_proof.json records the before/after instructions,
  component provenance, unchanged archive/proposal hashes, and reconciled feeds.

Verification
The live focused suite passed 131 tests, including 20 new alias tests, covering
source provenance, model-only assertions, ambiguous acronyms, inventory leakage,
different substituents, omitted conflicting source declarations, explicit portions
or aliquots on either description, separate feeds,
conflicting roles/loading/quantities, full release blocking, and raw-proposal
preservation. The saved-example proof confirms one 1.26 mmol charge after repair,
with the original result and raw proposal unchanged.

Run the proof from the repository root with
/home/amirreza/anaconda3/envs/flent/bin/python output/practical_benchmark_tools/alias_repair/verify_saved_recipe.py

This is a narrow deterministic identity repair. It does not infer arbitrary
synonyms or assess molecular structures, predict yield, or establish laboratory
readiness. The independent full publication replay is stored separately by the
input-boundary agent and should be consulted for whole-artifact consistency.
