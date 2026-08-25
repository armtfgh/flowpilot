# OpenAI Confirmatory Campaign Audit

## Frozen comparison

- Generator: GPT-5.4 for both architectures.
- Architectures: direct one-shot versus the full FlowPilot pipeline.
- Cases: hydrogenolysis, photochemical oxidation, and CuAAC.
- Repeats: three per case and architecture, giving 18 generated outcomes.
- Evaluation: 54 blinded judgments from Claude, OpenAI, and Qwen using 14 equal-weight, architecture-neutral criteria.
- The acceptance criteria and source checksums were frozen before generation.

## Primary result

| Measure | One-shot | FlowPilot |
|---|---:|---:|
| Mean score | 0.909 | 0.926 |
| Required-schema validity | 7/9 | 9/9 |
| Executable FlowPilot contract | n/a | 9/9 |
| Critical judge flags | 2 | 0 |

The matched mean delta was **+0.017** with a 95% interval of **-0.012 to +0.045**. FlowPilot won, tied, and lost 5/1/3 matched pairs. The generator-family-excluded sensitivity delta was **+0.011**.

This is a modest positive result, not proof of universal superiority: the interval crosses zero and only three chemistries were tested.

## Judge sensitivity

- Claude: -0.007
- OpenAI: +0.028
- Qwen: +0.030

Judge-family disagreement is retained as uncertainty. Exact agreement was 46.5%; agreement within one rubric point was 86.4%.

## What improved

- Gas bookkeeping was the largest gain (`UO-08`, +0.250).
- Residence-time/geometry closure improved (`UO-07`, +0.065).
- Stoichiometry and liquid material balance both improved (`UO-03` and `UO-06`, +0.046 each).
- FlowPilot produced the required structured result in 9/9 outcomes and incurred no critical judge flags.

## Remaining weaknesses

- Safety controls (`UO-12`, -0.056) need more explicit instantiated controls, especially H2 check valves, separator vent routing, and shutdown purging.
- Operating procedure (`UO-13`, -0.046) needs quantitative priming/flush instructions, shutdown order, depressurization, and chemistry-specific waste handling.
- Transport plausibility (`UO-11`, -0.019) needs clearer packed-bed wetting, liquid-holdup/contact-time, two-phase, and post-BPR flash assumptions.
- CuAAC was the weakest case-level comparison; its procedure detail, rather than catalyst placement, remains the main deficit.

## Defects found and corrected after scoring

The frozen benchmark outputs and scores were not replaced. Inspection exposed two architecture-level defects:

1. A stationary packed-bed catalyst could be repeated in pumped-feed composition. Final realization and contract validation now prohibit this conflict.
2. Evidence-first translation still inherited a model-derived intensification ceiling, allowing conservative residence-time revisions to be clipped to sub-minute values. The ceiling now applies only when the explicit policy is `intensify`.

The separate three-repeat OpenAI CuAAC regression passed all acceptance checks: all three contracts were executable, all selected 3.0 min, and none put Cu/C in a pumped feed. These regression results validate the repair but are not included in the frozen architecture score.

## Artifact map

- Frozen benchmark data: `/home/amirreza/Documents/codes/Flow Agent/ablation_results/manuscript_benchmark/openai_confirmatory_stationary_fix_20260820`
- Focused post-fix regression: `/home/amirreza/Documents/codes/Flow Agent/ablation_results/regression/openai_cuaac_tau_policy_20260820`
- Main summary: `summary.json`
- Pair-level data: `tables/repeat_level_paired_comparisons.csv`
- Criterion effects: `tables/criterion_architecture_effect.csv`
- New figures: `figures/fig08_openai_confirmatory_summary.png`, `figures/fig09_openai_criterion_effects.png`, and `figures/fig10_openai_contract_and_errors.png`
