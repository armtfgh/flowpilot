# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: Qwen3.8-27B.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- 1 fresh generation call per case/model/architecture: 6 outcomes.
- Blinded judges evaluate every outcome: 18 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Qwen3.8-27B: FlowPilot 0.924 vs one-shot 0.742; matched delta +0.182 (95% CI -0.001 to +0.365); wins/ties/losses 3/0/0.

Overall matched delta: **+0.182**. Generator-family-excluded sensitivity delta: **+0.227**.

FlowPilot final-contract closure in this campaign:

- Qwen3.8-27B: 3/3 executable, 0/3 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 28.9% exact and 59.2% within one rubric point; mean pairwise absolute difference was 0.798/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the 3 case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/qwen38_one_round_coverage_repair_20260821`.
