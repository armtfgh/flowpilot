UNAPPLIED CANDIDATE REPAIR — frozen benchmark sources and outcomes are untouched.

The fresh H2 FlowPilot run was released as blocked. Its sole inventory-assignment failure was pump_a at 0.1 mL/min. The allocator's availability, requested ID, flow interval, setting grid, pressure, quantity and system predicates pass; material-name matching alone fails. See h2_allocator_predicates.json.

Root cause: proposal.streams[0].contents explicitly includes Methanol, while the separate solvent field is empty. Main._build_translate_topology canonicalizes the contents to solute-only labels and excludes the already identified solvent component. The result supplied to inventory allocation no longer contains either inventory allowlist spelling, methanol or DMAOL. Canonical substrate naming alone is not a material-compatibility decision.

The proposed patch preserves explicit same-feed solvent components alongside canonical solute labels. A sole explicit solvent becomes the operation solvent with source text provenance only when that feed's solvent field is empty. It never inherits batch solvent or another feed's solvent and never overrides an explicit solvent. Multiple explicit solvent names are retained without inferring a ratio. The allocator checks every explicitly declared solvent identity, using the existing conventional alias matcher declared_solvent_member. A known matching solute or solvent cannot authorize another incompatible or unknown solvent. The explicit structured solvent and the same-feed canonical solvent components are both checked, so contradictory declarations remain conservative.

Files:
- preserve_feed_solvent.patch — unapplied two-file production diff.
- candidate_main.py and candidate_inventory_allocator.py — isolated source copies used by tests.
- build_isolated_patch.py — constructs these copies and diff, preserving live files.
- test_isolated_patch.py — twelve focused offline tests.
- production_*_sha256.txt — production source hashes at construction.

Validation command:
/home/amirreza/anaconda3/envs/flent/bin/python -m pytest -q output/benchmark_fairness_20261014/pump_material_repair/test_isolated_patch.py
Result: 12 passed. The actual saved H2 proposal rebuild has complete allocation using isolated candidate code. Its raw proposal and original result remain byte-for-byte unchanged. Other tests cover MeOH/methanol equivalence, DCM rejection for an otherwise unmatched feed, unknown-medium no fallback, different solvents in separate streams, preservation of an explicit solvent and gas stream, and unchanged production hashes.

Limits: this preserves declared information; it does not independently validate solvent compatibility, authorize a laboratory experiment, or resolve conflicting explicit solvent statements. For feeds with explicit medium information, every declared medium member must be present in the pump compatibility list (conventional aliases allowed). For feeds with no explicit medium information the old lexical fallback is retained; this narrow patch does not redesign the legacy inventory schema or prove compatibility of every solute. The H2 generated output also contains a separate top-level-versus-feed concentration discrepancy (0.1 versus 0.126 M), which this patch does not conceal. The blocked result remains part of this frozen benchmark and no new score is inferred from this offline patch test.
