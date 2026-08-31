# FlowPilot Ablation Test

This directory is the code-only, reproducible evaluation package for
FlowPilot. It deliberately excludes the THQ case study. Generated artifacts
are stored separately in `../ablation_results/`, which is ignored by Git.

## Directory boundary

Keep these files in Git:

- `src/`, `scripts/`, and `tests/`: implementation and tests.
- `configs/`, `protocols/`, and `benchmarks/`: frozen benchmark inputs.
- `sources/`: source-record manifest used by the audit.

Do not place generated JSON, logs, figures, tables, reports, or archives in
this directory. They belong in `../ablation_results/`. To put large results on
another disk, set `FLOWPILOT_ABLATION_RESULTS_DIR` to an absolute directory.

## Evaluation arms

Architecture ablation uses a fixed model bundle:

- `general_one_shot`: general LLM with the batch protocol only.
- `structured_single_agent`: one structured FlowProposal call.
- `no_retrieval`: FlowPilot with literature retrieval disabled.
- `no_engineering`: structured LLM output without deterministic calculations.
- `no_council`: FlowPilot upstream proposal before council deliberation.
- `no_inventory`: full council with an intentionally permissive inventory.
- `full`: complete FlowPilot pipeline.

Model portability is a separate experiment and does not get mixed with the
architecture ablation. The harness supports:

- Qwen3.6-27B and Qwen3.8-27B.
- GPT-4o.
- Claude Sonnet 4.6 and Claude Opus 4.6.
- Matched one-shot and FlowPilot execution for each retained generator.

The current primary architecture result is a completed, matched GPT-4o matrix:
seven variants across six protocols (42 cells). It is stored under
`../ablation_results/runs/pilot_matched_full_gpt4o_v3_20260728/`.

The main architecture evidence is:

- `../ablation_results/figures/19_agent_component_calls.*`: recorded upstream and named council calls.
- `../ablation_results/figures/20_ablation_architecture_map.*`: module presence/removal by variant.
- `../ablation_results/figures/21_architecture_execution_coverage.*`: matched cell completion.
- `../ablation_results/figures/22_full_vs_ablation_matched.*`: matched automated score comparison.
- `../ablation_results/figures/23_full_flowpilot_pipeline.*`: full system dataflow.
- `../ablation_results/figures/24_schema_validity_and_cost.*`: formal validity and execution cost.
- `../ablation_results/figures/25_quality_assurance_score_v2.*`: architecture-blind quality ranking.
- `../ablation_results/figures/26_quality_dimensions_v2.*`: score decomposition by measured dimension.
- `../ablation_results/figures/27_full_pairwise_advantage_v2.*`: paired Full-versus-ablation effects.
- `../ablation_results/figures/28_weight_sensitivity_v2.*`: winner stability across plausible weights.
- `../ablation_results/figures/29_deployment_readiness_v2.*`: hard-gated experimental readiness.
- `../ablation_results/figures/30_deployment_gate_rates_v2.*`: reasons designs are not deployment-ready.
- `../ablation_results/reports/agent_trace_summary.md`: run-level council call sequence.
- `../ablation_results/reports/quality_score_v2_methodology.md`: candidate scoring specification.
- `../ablation_results/tables/agent_call_events.csv`: call-level prompt/response hashes.

The automated composite intentionally remains a screening metric. It excludes
formal schema validity and council rejection/failure-handling benefits, so a
high one-shot score must not be interpreted as proof of scientific superiority.
Quality and Assurance Score v2 is an exploratory post-hoc reanalysis. Its
specification must be frozen and evaluated on new holdout cases before it is
used for a confirmatory superiority claim.

## Data boundaries

The files under `protocols/` contain a batch-only prompt and a hidden reference
flow result. The reference is used only by the external metric calculator. The
corresponding source record is excluded from retrieval. `scripts/audit.py`
checks for THQ text and hidden-answer contamination.

The local literature records are machine-extracted and must be checked against
the original papers before publication. They are suitable for engineering the
evaluation harness, not a substitute for human source verification.

## Run

```bash
python -m ablation_test.scripts.audit
python -m ablation_test.scripts.run --profile smoke
python -m ablation_test.scripts.run --profile pilot
python -m ablation_test.scripts.analyze
python -m pytest ablation_test/tests -q
```

Every run gets a timestamped directory under `../ablation_results/runs/` with:

- full LLM prompts and completions;
- stage and error logs;
- parsed intermediate JSON snapshots;
- final output and deterministic metrics;
- environment and model endpoint manifests;
- SHA-256 checksums.

Aggregated CSV/JSON tables, expert-review sheets, figures, and the generated
report are written under `../ablation_results/` in `tables/`, `figures/`,
`expert_scoring/`, and `reports/`.

## Current cross-model benchmark

NewGen 2.0 compares Qwen3.6-27B, Qwen3.8-27B, GPT-4o, Claude Sonnet 4.6,
and Claude Opus 4.6. Every generator is evaluated in matched one-shot and
FlowPilot conditions on the same three cases with three generation repeats.

```bash
python -m ablation_test.scripts.run_manuscript_three_model_repeated_benchmark --phase all
python -m ablation_test.scripts.run_qwen38_three_repeat_benchmark --phase all
python -m ablation_test.scripts.run_alternative_frontier_three_repeat_benchmark --phase all
python -m ablation_test.scripts.build_all_models_three_repeat_summary
```
