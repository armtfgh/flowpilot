# Alternative OpenAI and Claude NewGen 2.0 Benchmark

## Frozen design

This one-round extension uses the same three held-out protocols, authoritative inventories, two architecture conditions, unchanged 14-criterion rubric, and three-judge panel as the prior manuscript campaign. It contains 12 generated outcomes and 36 blinded judgments. No result was selected or replaced based on its score.

## Main results

- **Claude Opus 4.6:** FlowPilot 0.921 vs one-shot 0.892; paired delta +0.029; wins/ties/losses 2/0/1; generator-family-excluded delta +0.047.
- **GPT-4o:** FlowPilot 0.909 vs one-shot 0.675; paired delta +0.234; wins/ties/losses 3/0/0; generator-family-excluded delta +0.155.

The pooled matched delta was **+0.132**; the generator-family-excluded sensitivity was **+0.101**. All six FlowPilot outcomes closed as executable.

## Critical-error review

- **Claude Opus 4.6:** one-shot had 1/3 candidates and 2 judge flags; FlowPilot had 0/3 candidates and 0 flags.
- **GPT-4o:** one-shot had 3/3 candidates and 14 judge flags; FlowPilot had 0/3 candidates and 0 flags.

These are judge-reported critical-error flags, not deterministic arithmetic-error counts. Deterministic formal validity is reported separately in `outcome_contract_summary.csv`.

## Interpretation boundary

The GPT-4o result is directionally strong across all three cases. The Claude Opus result is positive on average but small, includes one case-level loss, and its three-pair confidence interval crosses zero. This one-round campaign demonstrates model-dependent architecture effects; it does not establish population-level superiority across flow chemistry. Wet-lab validation remains outside this benchmark.

## Reproducibility

See `CAMPAIGN_AUDIT.json`, `RAW_FILE_CHECKSUMS.csv`, `frozen/`, `tables/`, and `figures/`. The complete prompts, raw responses, telemetry, retries, normalized outcomes, and judge records are stored in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/alternative_frontier_one_round_20260821`.
