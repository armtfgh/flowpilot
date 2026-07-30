# Stage 1 Gate Report

Study: `fair_architecture_benchmark_v1_20260730`

Run directory: `stage1_smoke_run_20260730_attempt02`

## Decision

**STAGE_2_CLEARED**

Stage 1 gate passed: `True`

## Gate checks

- providers_and_models_available: `PASS`
- all_cells_recorded: `PASS`
- all_cells_completed_without_adapter_error: `PASS`
- no_systemic_truncation: `PASS`
- complete_artifact_sets: `PASS`
- identical_input_hashes_per_scenario: `PASS`
- oracle_witness_pass_rate_100pct: `PASS`

## Smoke observations

- Planned and recorded cells: `8`
- Oracle witness pass rate: `100.0%`
- Observed disposition accuracy: `75.0%`
- Model accuracy was not used as a stage gate.

## Scope

This smoke run uses one feasible/infeasible photochemical pair across four
conditions. It validates adapters, frozen inputs, deterministic scoring, and
artifact capture. It does not estimate comparative model performance.
