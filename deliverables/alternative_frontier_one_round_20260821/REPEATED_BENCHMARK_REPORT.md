# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: Claude Opus 4.6, GPT-4o.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- 1 fresh generation call per case/model/architecture: 12 outcomes.
- Blinded judges evaluate every outcome: 36 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Claude Opus 4.6: FlowPilot 0.921 vs one-shot 0.892; matched delta +0.029 (95% CI -0.110 to +0.168); wins/ties/losses 2/0/1.
- GPT-4o: FlowPilot 0.909 vs one-shot 0.675; matched delta +0.234 (95% CI +0.109 to +0.359); wins/ties/losses 3/0/0.

Overall matched delta: **+0.132**. Generator-family-excluded sensitivity delta: **+0.101**.

FlowPilot final-contract closure in this campaign:

- Claude Opus 4.6: 3/3 executable, 0/3 blocked.
- GPT-4o: 3/3 executable, 0/3 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 28.3% exact and 61.8% within one rubric point; mean pairwise absolute difference was 0.803/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the 3 case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/alternative_frontier_one_round_20260821`.
