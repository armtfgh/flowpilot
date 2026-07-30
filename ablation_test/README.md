# FlowPilot Ablation Test

This directory is a self-contained, reproducible evaluation package for
FlowPilot. It deliberately excludes the THQ case study.

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

- Claude upstream and Claude council.
- Qwen upstream and Qwen council.
- Gemma upstream and Gemma council.
- Claude upstream and Qwen council.

The current primary architecture result is a completed, matched GPT-4o matrix:
seven variants across six protocols (42 cells). It is stored under
`runs/pilot_matched_full_gpt4o_v3_20260728/`.

The main architecture evidence is:

- `figures/19_agent_component_calls.*`: recorded upstream and named council calls.
- `figures/20_ablation_architecture_map.*`: module presence/removal by variant.
- `figures/21_architecture_execution_coverage.*`: matched cell completion.
- `figures/22_full_vs_ablation_matched.*`: matched automated score comparison.
- `figures/23_full_flowpilot_pipeline.*`: full system dataflow.
- `figures/24_schema_validity_and_cost.*`: formal validity and execution cost.
- `figures/25_quality_assurance_score_v2.*`: architecture-blind quality ranking.
- `figures/26_quality_dimensions_v2.*`: score decomposition by measured dimension.
- `figures/27_full_pairwise_advantage_v2.*`: paired Full-versus-ablation effects.
- `figures/28_weight_sensitivity_v2.*`: winner stability across plausible weights.
- `figures/29_deployment_readiness_v2.*`: hard-gated experimental readiness.
- `figures/30_deployment_gate_rates_v2.*`: reasons designs are not deployment-ready.
- `reports/agent_trace_summary.md`: run-level council call sequence.
- `reports/quality_score_v2_methodology.md`: candidate scoring specification.
- `tables/agent_call_events.csv`: call-level prompt/response hashes.

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

Every run gets a timestamped directory under `runs/` with:

- full LLM prompts and completions;
- stage and error logs;
- parsed intermediate JSON snapshots;
- final output and deterministic metrics;
- environment and model endpoint manifests;
- SHA-256 checksums.

Aggregated CSV/JSON tables, expert-review sheets, figures, and the generated
report are written to `tables/`, `figures/`, `expert_scoring/`, and `reports/`.

## Cross-model benchmark

The matched Qwen/frontier experiment compares Qwen 27B one-shot, Qwen 27B with
Full FlowPilot, GPT-4o one-shot, Claude Sonnet 4.6 one-shot, and GPT-4o with
Full FlowPilot. It uses the same 12 non-THQ protocols and the frozen v2 scorer.

```bash
python -m ablation_test.scripts.run_cross_model --profile smoke \
  --condition qwen_full
python -m ablation_test.scripts.run_cross_model --profile publication \
  --workers 2 --output-id qwen_frontier_20260729
python -m ablation_test.scripts.cross_model_report \
  --experiment ablation_test/runs/cross_model_publication_qwen_frontier_20260729 \
  --output ablation_test/cross_model_package_20260729
```

The collaborator-facing report adds score decomposition, schema-neutral
sensitivity, protocol-clustered uncertainty, parameter repeatability, functional
reproducibility, and a figure interpretation guide:

```bash
python -m ablation_test.scripts.coworker_report \
  --benchmark-package ablation_test/cross_model_package_20260729 \
  --experiment ablation_test/runs/cross_model_publication_qwen_frontier_20260729 \
  --output ablation_test/coworker_ablation_package_20260729
```
