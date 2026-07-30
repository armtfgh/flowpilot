# FlowPilot Ablation and Repeatability Study

## Purpose

This package explains the completed FlowPilot ablation benchmark for a collaborator who did not build the software. It separates four questions:

1. Does the full architecture improve design quality over a general one-shot LLM?
2. Are the proposed designs internally consistent and compatible with laboratory hardware?
3. Are cautious, audited designs actually ready for immediate execution?
4. Does the same protocol produce functionally similar results across repeated runs?

The benchmark evaluates engineering decision support. It does not establish superior wet-lab yield prediction.

## Study Design

- 12 literature-derived, non-THQ batch protocols.
- 5 model/architecture conditions.
- 3 independent runs per protocol and condition.
- 180 completed runs with no failed cells.
- Temperature zero and predetermined seeds.
- The hidden literature flow result was excluded from model prompts and retrieval.
- The frozen scorer did not receive the architecture label.
- Scorer SHA-256: `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`.

![Study design](figures/01_study_design.png)

## Conditions

| Condition | What it contains |
|---|---|
| Qwen 27B one-shot | One general prompt and one Qwen response |
| Qwen 27B + Full FlowPilot | Lightweight Qwen upstream adapter, retrieval, deterministic engineering, inventory enforcement, Qwen council, skeptic, revision, and validation |
| GPT-4o one-shot | One general prompt and one GPT-4o response |
| Claude Sonnet 4.6 one-shot | One general prompt and one Claude response |
| GPT-4o + Full FlowPilot | Full-schema GPT-4o upstream, retrieval, deterministic engineering, inventory enforcement, GPT-4o council, skeptic, revision, and validation |

Both full conditions used the same shared OpenAI embedding service for retrieval. Embeddings did not generate any design text.

## Main Results

| Condition | Quality | Readiness | Ready rate | Quality SD | Reliable translation |
|---|---:|---:|---:|---:|---:|
| Qwen 27B + Full FlowPilot | 0.921 | 0.719 | 22.2% | 0.0065 | 58.3% |
| GPT-4o + Full FlowPilot | 0.895 | 0.743 | 33.3% | 0.0129 | 58.3% |
| Claude Sonnet 4.6 one-shot | 0.588 | 0.350 | 0.0% | 0.0195 | 0.0% |
| GPT-4o one-shot | 0.550 | 0.350 | 0.0% | 0.0143 | 0.0% |
| Qwen 27B one-shot | 0.545 | 0.350 | 0.0% | 0.0088 | 0.0% |

![Quality comparison](figures/02_quality_clustered_ci.png)

### Primary interpretation

- Qwen Full achieved the highest mean quality (0.921).
- GPT-4o Full achieved the highest mean readiness (0.743) and ready rate (33.3%).
- Qwen Full beat Qwen one-shot on every protocol.
- Full FlowPilot improves traceability, structured validity, inventory compliance, gas bookkeeping, and independent review.
- Full FlowPilot should still be supervised by a chemist because many outputs correctly requested experimental screening.

## What The Score Measures

The Quality and Assurance Score v2 is a weighted output-based score:

```text
0.10 Formal validity
+ 0.25 Engineering integrity
+ 0.15 Process completeness
+ 0.15 Safety adequacy
+ 0.10 Evidence provenance
+ 0.20 Decision assurance
+ 0.05 Actionability and calibration
```

The first four dimensions contribute 65% and measure direct design quality. The remaining 35% measures evidence, independent review, and calibrated actionability.

![Score weights](figures/17_score_weights.png)

![Score dimensions](figures/04_quality_dimensions.png)

### Direct quality versus assurance

One-shot systems received substantial engineering, process, and safety credit. They were not assigned zero. Their main deficits were strict machine-contract validity, retrieved-source provenance, deterministic calculation traces, inventory provenance, council deliberation, and final audit.

![Direct versus assurance](figures/03_direct_quality_vs_assurance.png)

## Paired Architecture Effects

| Comparison | Mean difference | 95% case-bootstrap interval | W-L-T |
|---|---:|---:|---:|
| Qwen 27B + Full FlowPilot minus Qwen 27B one-shot | +0.375 | [+0.348, +0.401] | 12-0-0 |
| Qwen 27B + Full FlowPilot minus GPT-4o one-shot | +0.371 | [+0.343, +0.400] | 12-0-0 |
| Qwen 27B + Full FlowPilot minus Claude Sonnet 4.6 one-shot | +0.332 | [+0.303, +0.362] | 12-0-0 |
| GPT-4o + Full FlowPilot minus GPT-4o one-shot | +0.346 | [+0.310, +0.383] | 12-0-0 |
| GPT-4o + Full FlowPilot minus Qwen 27B + Full FlowPilot | -0.025 | [-0.040, -0.011] | 1-9-2 |

![Paired effects](figures/05_paired_quality_effects.png)

Paired differences were calculated after averaging the three repetitions within each protocol. Confidence intervals used 20,000 protocol-level bootstrap samples, avoiding treatment of repeated runs as independent chemistries.

## Schema-Neutral Sensitivity

Every general one-shot output was parseable JSON but failed at least one strict FlowProposal field type. This set formal validity to zero and capped readiness at 0.35. To test whether the conclusion depended on that penalty, we recalculated quality without formal validity and readiness without the schema cap.

![Schema sensitivity](figures/06_schema_neutral_sensitivity.png)

Without the formal-validity component, Qwen Full remained at 0.912, compared with 0.606 for Qwen one-shot, 0.611 for GPT-4o one-shot, and 0.654 for Claude one-shot.

The architecture advantage therefore does not depend only on schema compliance. Nevertheless, a future confirmatory benchmark should include an exact-schema structured one-shot baseline.

## Readiness Is Not Quality

Quality rewards a well-supported and correctly cautious recommendation. Readiness asks whether the design can be executed immediately. Hard defects cap readiness even when the output is otherwise well documented.

![Readiness](figures/07_readiness_and_ready_rate.png)

![Hard checks](figures/08_hard_check_rates.png)

![Deployment gates](figures/09_deployment_gate_rates.png)

A screened design is not an experimental failure. It is a recommendation that additional validation is required before deployment.

## Protocol-Level Behavior

![Case quality](figures/10_case_quality_heatmap.png)

![Qwen gain](figures/16_qwen_gain_by_protocol.png)

Qwen Full improved quality over Qwen one-shot for all 12 protocols. The magnitude varied by chemistry, which is expected because gas-liquid, heterogeneous, hazardous, and multistep cases exercise different modules.

## Repeatability And Reproducibility

The current three repeated runs measure run-to-run repeatability under the same software and provider environment. Reproducibility across another server, software version, or date requires a separate rerun.

A repeated answer is not automatically useful. A one-shot model can repeat the same non-deployable design. We therefore report several layers:

- Numeric repeatability: all relevant parameters remain inside predefined tolerances.
- Functional reproducibility: numeric repeatability plus geometry, pump, tubing, exact reactor inventory, gas, topology, and safety checks pass in every repeat.
- Strict reproducibility: functional reproducibility plus valid structured output.
- Reliable translation: strict reproducibility plus quality at least 0.70 in every repeat.

Parameter tolerances were residence time, liquid flow, concentration, reactor volume, and gas quantities within 10%; temperature within 2 C; BPR within 0.5 bar; and tubing ID within 0.10 mm. Gas-column percentages use only the protocols that require a gas feed.

![Quality repeatability](figures/11_quality_repeatability.png)

![Parameter repeatability](figures/12_parameter_repeatability.png)

![Functional reproducibility](figures/13_functional_reproducibility.png)

![Tradeoff](figures/14_quality_repeatability_tradeoff.png)

### Repeatability interpretation

- Qwen Full had the lowest mean within-protocol quality SD (0.0065).
- Qwen one-shot had the highest all-parameter numeric repeatability (83.3%), showing that exact numeric stability does not by itself establish design quality.
- Qwen Full deployment decisions agreed across all three repeats for 91.7% of protocols.
- GPT-4o Full decision agreement was 75.0%.
- Three repetitions are preliminary. A strong reproducibility claim should use at least 10 repeats on a stratified protocol subset.

## Execution Cost

![Execution cost](figures/15_execution_cost.png)

Full FlowPilot requires substantially more calls and runtime because it performs retrieval, candidate generation, specialist scoring, skeptical audit, revision, and final validation. The Qwen local pipeline reduces commercial API dependence but is slower on the tested local server.

## What Can Be Claimed

> Under the frozen Quality and Assurance Score v2, Full FlowPilot produced more structured, engineering-consistent, inventory-aware, traceable, and safety-audited batch-to-flow designs than general one-shot prompting across 12 matched protocols and three repetitions.

## What Cannot Yet Be Claimed

- Superior wet-lab yield prediction.
- Identification of globally optimal operating conditions.
- Safe autonomous operation without chemist review.
- General superiority over every possible structured frontier-model prompt.
- Cross-server or cross-version reproducibility from the current three-repeat study.

## Recommended Next Validation

1. Add an exact-schema structured one-shot baseline.
2. Select six stratified protocols and run 10 repeats per condition.
3. Repeat the frozen benchmark on another date or model deployment.
4. Conduct blinded expert scoring without architecture labels.
5. Compare predicted conditions with wet-lab conversion, yield, operability, and safety observations.

## Files In This Package

- `tables/condition_overview.csv`: headline quality, readiness, sensitivity, cost, and repeatability metrics.
- `tables/reproducibility_by_case.csv`: protocol-condition repeatability outcomes.
- `tables/parameter_repeatability.csv`: parameter-level spread and tolerance results.
- `tables/quality_dimensions.csv`: score decomposition.
- `tables/paired_comparisons.csv`: predefined paired effects.
- `tables/model_provenance.csv`: provider/model evidence from call logs.
- `tables/data_dictionary.csv`: definitions for derived variables.
- `figures/`: PNG and vector PDF versions of every figure.
- Raw run directory: `ablation_test/runs/cross_model_publication_qwen_frontier_20260729`.
