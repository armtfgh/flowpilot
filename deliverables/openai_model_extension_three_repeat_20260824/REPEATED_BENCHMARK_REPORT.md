# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: GPT-5.2, GPT-5.4 mini, GPT-5.5.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- 3 fresh generation calls per case/model/architecture: 54 outcomes.
- Blinded judges evaluate every outcome: 162 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- GPT-5.2: FlowPilot 0.825 vs one-shot 0.909; matched delta -0.084 (95% CI -0.319 to +0.151); wins/ties/losses 4/1/4.
- GPT-5.4 mini: FlowPilot 0.895 vs one-shot 0.835; matched delta +0.060 (95% CI +0.003 to +0.118); wins/ties/losses 6/2/1.
- GPT-5.5: FlowPilot 0.649 vs one-shot 0.962; matched delta -0.313 (95% CI -0.534 to -0.092); wins/ties/losses 1/0/8.

Overall matched delta: **-0.112**. Generator-family-excluded sensitivity delta: **-0.124**.

FlowPilot final-contract closure in this campaign:

- GPT-5.2: 8/9 executable, 0/9 blocked, 1/9 generation failed.
- GPT-5.4 mini: 9/9 executable, 0/9 blocked, 0/9 generation failed.
- GPT-5.5: 4/9 executable, 5/9 blocked, 0/9 generation failed.

Blocked and generation-failed outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 40.5% exact and 81.1% within one rubric point; mean pairwise absolute difference was 0.557/4.

## Interpretation

The paired case-repeat is the unit of comparison. Confidence intervals describe variation across the 9 case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/openai_model_extension_three_repeat_20260824`.
