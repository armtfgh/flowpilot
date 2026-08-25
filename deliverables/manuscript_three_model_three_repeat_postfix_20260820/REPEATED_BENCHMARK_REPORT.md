# Three-Model Repeated Benchmark

> **Post-benchmark audit:** Two GPT-5.4/FlowPilot CuAAC outputs duplicated the
> packed-bed Cu/C catalyst in the pumped feed. The frozen scores below are
> retained unchanged. See `POST_FIX_BENCHMARK_AUDIT.md` for the defect analysis,
> corrected deterministic invariant, and interpretation limits.

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Three generator families: Qwen3.6-27B, GPT-5.4, and Claude Sonnet 4.6.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- Three fresh generation calls per case/model/architecture: 54 outcomes.
- Three blinded judges (Qwen, OpenAI, Claude) evaluate every outcome: 162 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Claude Sonnet 4.6: FlowPilot 0.924 vs one-shot 0.890; matched delta +0.034 (95% CI +0.005 to +0.063); wins/ties/losses 7/1/1.
- GPT-5.4: FlowPilot 0.896 vs one-shot 0.923; matched delta -0.028 (95% CI -0.076 to +0.020); wins/ties/losses 3/2/4.
- Qwen3.6-27B: FlowPilot 0.904 vs one-shot 0.765; matched delta +0.139 (95% CI +0.071 to +0.207); wins/ties/losses 9/0/0.

Overall matched delta: **+0.048**. Generator-family-excluded sensitivity delta: **+0.085**.

FlowPilot final-contract closure in this campaign:

- Claude Sonnet 4.6: 9/9 executable, 0/9 blocked.
- GPT-5.4: 9/9 executable, 0/9 blocked.
- Qwen3.6-27B: 9/9 executable, 0/9 blocked.

Blocked outcomes remain in the primary score; the report does not substitute results from earlier campaigns.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 34.2% exact and 73.8% within one rubric point; mean pairwise absolute difference was 0.654/4.

## Interpretation

The paired repeat is the unit of comparison. Confidence intervals describe variation across the nine case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/manuscript_three_model_three_repeat_postfix_20260820`.
