# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: Qwen3.8-27B.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- 3 fresh generation calls per case/model/architecture: 18 outcomes.
- Blinded judges evaluate every outcome: 54 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Qwen3.8-27B: FlowPilot 0.915 vs one-shot 0.747; matched delta +0.168 (95% CI +0.101 to +0.235); wins/ties/losses 9/0/0.

Overall matched delta: **+0.168**. Generator-family-excluded sensitivity delta: **+0.205**.

FlowPilot final-contract closure in this campaign:

- Qwen3.8-27B: 9/9 executable, 0/9 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 25.4% exact and 65.4% within one rubric point; mean pairwise absolute difference was 0.766/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the 9 case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/qwen38_three_repeat_20260824`.
