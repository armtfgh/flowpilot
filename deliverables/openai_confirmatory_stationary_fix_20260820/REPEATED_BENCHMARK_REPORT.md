# Repeated Architecture Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Generator families: GPT-5.4.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- Three fresh generation calls per case/model/architecture: 18 outcomes.
- Blinded judges evaluate every outcome: 54 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- GPT-5.4: FlowPilot 0.926 vs one-shot 0.909; matched delta +0.017 (95% CI -0.012 to +0.045); wins/ties/losses 5/1/3.

Overall matched delta: **+0.017**. Generator-family-excluded sensitivity delta: **+0.011**.

FlowPilot final-contract closure in this campaign:

- GPT-5.4: 9/9 executable, 0/9 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 46.5% exact and 86.4% within one rubric point; mean pairwise absolute difference was 0.450/4.

## Interpretation

The paired repeat is the unit of comparison. Confidence intervals describe variation across the nine case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/openai_confirmatory_stationary_fix_20260820`.
