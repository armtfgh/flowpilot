# Three-Model Repeated Benchmark

## Frozen Design

- Three held-out chemistries selected before this campaign: Hydrogenolysis, Photochemical oxidation, CuAAC.
- Three generator families: Qwen3.6-27B, GPT-5.4, and Claude Sonnet 4.6.
- Two matched architectures: direct one-shot and the full FlowPilot pipeline.
- Three fresh generation calls per case/model/architecture: 54 outcomes.
- Three blinded judges (Qwen, OpenAI, Claude) evaluate every outcome: 162 judgments.
- The unchanged NewGen 2.0 rubric contains 14 equally weighted, architecture-neutral criteria.
- Temperature is 0.2; each call records a distinct seed. Anthropic does not guarantee seed control.

## Results

- Claude Sonnet 4.6: FlowPilot 0.843 vs one-shot 0.888; matched delta -0.045 (95% CI -0.172 to +0.083); wins/ties/losses 5/0/4.
- GPT-5.4: FlowPilot 0.930 vs one-shot 0.905; matched delta +0.025 (95% CI -0.004 to +0.054); wins/ties/losses 6/1/2.
- Qwen3.6-27B: FlowPilot 0.898 vs one-shot 0.763; matched delta +0.135 (95% CI +0.055 to +0.215); wins/ties/losses 8/0/1.

Overall matched delta: **+0.038**. Generator-family-excluded sensitivity delta: **+0.059**.

FlowPilot final-contract closure was 9/9 for Qwen, 9/9 for GPT-5.4, and 7/9 for Claude Sonnet 4.6. The two Claude blocks were a CuAAC inventory-topology allocation failure and a photochemical pump/topology feasibility failure. These blocked outcomes remain in the primary score.

UO-09 (solids/slurry handling) is absent from the criterion-effect heatmap because it was not applicable to any of the three frozen cases; no applicable criterion observations were removed.

Judge agreement was 32.9% exact and 73.0% within one rubric point; mean pairwise absolute difference was 0.670/4.

## Interpretation

The paired repeat is the unit of comparison. Confidence intervals describe variation across the nine case-repeat pairs per model, not uncertainty over all flow chemistry. LLM judging remains secondary evidence and does not replace deterministic checks or wet-lab validation. All failures, retries, raw responses, prompts, seeds, checksums, and model metadata are retained in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/manuscript_three_model_three_repeat_20260819`.
