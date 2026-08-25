# Qwen3.8-27B One-Round Audit

## Endpoint

- Base URL: `http://10.13.24.104:8000/v1`
- Advertised model: `/models/Qwen3.8-27B`
- vLLM model discovery: reachable and model advertised
- Generation model: Qwen3.8-27B for both one-shot and FlowPilot arms

## Frozen Comparison

- Cases: Hydrogenolysis, Photochemical oxidation, and CuAAC
- Architectures: direct one-shot and full FlowPilot
- Repeats: one fresh generation per matched case/architecture cell
- Outputs: 6/6 generated and schema-valid
- Judges: Qwen3.6-27B, OpenAI, and Claude
- Judgments: 18/18 valid
- Rubric: 14 equally weighted NewGen 2.0 criteria, frozen before generation
- Inventory: the same case-specific strict inventory was supplied to both matched arms

## Results

| Case | One-shot | FlowPilot | Paired delta |
|---|---:|---:|---:|
| Hydrogenolysis | 0.622 | 0.878 | +0.256 |
| Photochemical oxidation | 0.833 | 0.942 | +0.109 |
| CuAAC | 0.771 | 0.951 | +0.181 |
| Mean | 0.742 | 0.924 | +0.182 |

FlowPilot won all three matched comparisons. Its deterministic final-design contract closed as executable in 3/3 cases, with no blocked or missing outcomes. The generator-family-excluded sensitivity delta was +0.227.

The largest rubric improvement was gas bookkeeping (`UO-08`, +0.458). Inventory feasibility (`UO-10`) improved from 0.833 to 0.944. The one-shot hydrogenolysis output received the only critical-error flags in the campaign; the FlowPilot outputs received none.

## Defect Found And Corrected

The first campaign exposed a council score-coverage boundary defect. With two candidates and a scoring batch size of two, the coverage-repair path was skipped because it was entered only when `batch_size < candidate_count`. Qwen3.8 scored candidate 1 but omitted candidate 2, causing the CuAAC FlowPilot cell to fail.

The condition now includes the equality boundary. Partial score responses trigger a targeted request for only the missing candidate IDs, merged scores are revalidated, and incomplete final coverage still fails strictly. A regression test reproduces the omitted-candidate response and verifies complete ordered coverage. The clean campaign reported here was generated from the corrected code; the failed pre-fix campaign remains retained separately as an audit trail.

## Interpretation

This is a connectivity and one-round architecture check, not the confirmatory manuscript campaign. The mean paired delta has a 95% confidence interval of -0.001 to +0.365 because there are only three case-level pairs. The judge agreement rate was 28.9% exact and 59.2% within one rubric point, so deterministic contract results and case-level evidence should be reported alongside the LLM consensus score.

For a stronger claim, repeat this frozen protocol three times without changing prompts, rubric, inventory, cases, or stopping rules.

## Artifact Locations

- Raw generations, prompts, judgments, telemetry, and logs: `ablation_results/manuscript_benchmark/qwen38_one_round_coverage_repair_20260821`
- Tables, figures, frozen manifests, and report: `deliverables/qwen38_one_round_coverage_repair_20260821`
- Failed pre-fix audit run: `ablation_results/manuscript_benchmark/qwen38_one_round_20260821`
