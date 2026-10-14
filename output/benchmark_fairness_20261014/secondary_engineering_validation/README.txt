SECONDARY ENGINEERING VALIDATION — UNSCORED, NO MODEL CALLS

Preregistration is in preregistration.json. The fixed population is all three new FlowPilot outputs from claude_fair_bridge_20261014. The paid benchmark, original grades, generated physical decisions, inventories and one-shot outputs remain unchanged.

replay_isolated_pump_patch.py recompiles each saved reviewed proposal twice. The baseline uses unchanged production functions. The candidate uses only the isolated candidate_main._build_translate_topology and candidate_inventory_allocator.InventoryAllocator in the sibling pump_material_repair directory. The helper executes the ordinary realization predicates, topology allocation, disposition gate, final contract and publication. It does not re-run chemistry reconciliation or realization, which could change the reviewed proposal.

The normal disposition gate can update derived release metadata, such as BLOCK to SCREEN. Exact changes to that metadata are logged. Physical proposal values, council records, chemistry plan, source batch protocol, inventory, original realization decisions and engineering metrics are asserted unchanged. Original source files and artifacts are hashed before/after. A separate negative control confirms a failed pump check still blocks each output.

Independent arithmetic observations are reported separately from normal software-gate outcomes. The H2 diagnostic reproduces the original block with baseline code and releases an executable/SCREEN contract with the isolated patch. It still records an existing top-level concentration of 0.1 M versus an explicit feed concentration of 0.126 M. The patch does not repair that discrepancy. Passing software gates must not be described as proving every chemical or engineering choice correct.

The diagnostic H2 run completed without network access or live source edits. latest_diagnostic.json points to its full result, baseline/candidate comparison, diagrams and artifact manifest. Each replay stores exact candidate-code hashes so later candidate edits cannot silently change its provenance.

The all-three authoritative replay is intentionally deferred until all 63 fixed primary judgments are complete, including explicitly documented N/A-format recovery where applicable. The helper checks the complete format_recovery/summary_all.json and its referenced immutable hashes before running. Invoke from the project root after primary completion:

/home/amirreza/anaconda3/envs/flent/bin/python output/benchmark_fairness_20261014/secondary_engineering_validation/replay_isolated_pump_patch.py

The existing H2-only diagnostic can be repeated with --diagnostic-case hydrogenolysis. Diagnostic mode is explicitly marked partial and cannot become an all-three result. Full results will be stored in a timestamped runs/all_three_* directory and referenced by latest_all_three.json.

No derivative judging has been authorized. These artifacts are secondary engineering validation, not fresh generations or revised benchmark scores.

Authoritative all-three replay completed 2026-10-14 03:38:50 UTC.
Run: runs/all_three_20261014T033850_412623Z (latest_all_three.json pins summary).
The timestamped gate amendment allows all fixed primary cells to have settled, including one exhausted invalid Qwen response cell; the primary valid panel remains 62/63 and is not imputed.

Hydrogenolysis: baseline blocked; isolated candidate executable/SCREEN, 28/28 normal checks. The independent audit retains top-level 0.100 M versus single-feed 0.126 M.
Photochemical oxidation: baseline and isolated candidate executable/SCREEN, 28/28 normal checks. This gate result does not repair the missing wavelength caused upstream by schema-envelope loss.
CuAAC: baseline and isolated candidate blocked, 26/28 normal checks. COUNCIL-NO-APPROVED-CANDIDATE and DESIGN_REALIZATION_COMPLETE remain. The independent audit retains top-level 0.100 M versus single-feed 0.250 M.

All physical proposals, council records, chemistry plans, inventory and engineering metrics remained unchanged. A synthetic failed pump check still blocks each case. All 1,091 primary archive files were unchanged. The concurrent shared spending ledger was explicitly excluded from the immutable archive set and captured separately; its before/after hash was also unchanged during this run. All live and isolated candidate code hashes were unchanged. No model/API calls, new designs, new judge scores, or laboratory experiments were performed. This replay is an unscored software correction test, not a replacement benchmark or complete physical validation.
