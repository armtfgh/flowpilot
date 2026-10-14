UNAPPLIED official benchmark harness repair

scope_final_output.patch modifies only ablation_test/scripts/run_fair_practical_benchmark.py. It transfers the final-output scoping behavior tested by the separate follow-up into the reusable official driver. It preserves the public/judge case and all scientific requirements while withholding the final JSON-envelope and field-list prefix from intermediate production objectives.

The wrapper captures a native-function marker at import to avoid recursive wrapping after configure/reload. configure installs the same wrapper on repeated calls. Unexpected objectives are rejected, rather than silently dropping extra user constraints. Every run records public/internal objectives and exact removed/retained text. Future campaign freezes include the internal objective and a manifest policy.

After the current frozen campaign and follow-up are fully verified, root may apply scope_final_output.patch and copy test_fair_objective_routing.py into ablation_test/tests/. Frozen historical source copies and current campaign artifacts must remain unchanged.

Validation before applying
- git apply --check succeeded.
- The five new production-facing tests plus all sixteen existing official-driver tests passed against the isolated candidate, 21 total.
- An actual second module import after configure preserved the unwrapped native function, and repeated configure did not stack wrappers.
- Live official source SHA stayed equal to production_source_sha256.txt.
- No model calls, paid calls, or production modifications were made.

check_candidate.py is an output-only test adapter for reviewing the unapplied candidate. Only the regression test file should be copied into the core test suite; candidate source and check helper stay in output.
