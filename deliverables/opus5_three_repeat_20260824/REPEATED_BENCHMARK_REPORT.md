# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: Claude Opus 5.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- 3 fresh generation calls per case/model/architecture: 18 outcomes.
- Blinded judges evaluate every outcome: 54 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Claude Opus 5: FlowPilot 0.772 vs one-shot 0.936; matched delta -0.164 (95% CI -0.312 to -0.016); wins/ties/losses 2/0/7.

Overall matched delta: **-0.164**. Generator-family-excluded sensitivity delta: **-0.157**.

FlowPilot final-contract closure in this campaign:

- Claude Opus 5: 4/9 executable, 5/9 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 43.0% exact and 79.4% within one rubric point; mean pairwise absolute difference was 0.573/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the 9 case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/opus5_three_repeat_20260824`.
