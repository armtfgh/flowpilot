# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: Claude Opus 4.6, GPT-4o.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- 3 fresh generation calls per case/model/architecture: 36 outcomes.
- Blinded judges evaluate every outcome: 108 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Claude Opus 4.6: FlowPilot 0.932 vs one-shot 0.883; matched delta +0.049 (95% CI +0.016 to +0.082); wins/ties/losses 8/0/1.
- GPT-4o: FlowPilot 0.910 vs one-shot 0.684; matched delta +0.226 (95% CI +0.187 to +0.265); wins/ties/losses 9/0/0.

Overall matched delta: **+0.138**. Generator-family-excluded sensitivity delta: **+0.120**.

FlowPilot final-contract closure in this campaign:

- Claude Opus 4.6: 9/9 executable, 0/9 blocked.
- GPT-4o: 9/9 executable, 0/9 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 30.3% exact and 66.4% within one rubric point; mean pairwise absolute difference was 0.746/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the 9 case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/alternative_frontier_three_repeat_20260824`.
