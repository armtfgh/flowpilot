# FlowPilot five-case confirmatory result

Date: 2026-08-19

## Frozen stopping rule

The attempt was accepted only if all conditions were met:

1. All ten FlowPilot candidates close deterministically.
2. Overall mean paired FlowPilot minus one-shot score is positive.
3. The mean comparison is non-negative for both generator families.
4. Generator-family-excluded comparison is non-negative for both families.
5. FlowPilot does not have more critical flags than the corresponding one-shot baseline.

All conditions were met. No further score-driven rerun was performed.

## Results

| Generator | FlowPilot | One-shot | Difference | FlowPilot critical flags | One-shot critical flags |
|---|---:|---:|---:|---:|---:|
| GPT-5.4 | 0.9204 | 0.9085 | +0.0119 | 0 | 0 |
| Qwen3.6-27B | 0.8912 | 0.7436 | +0.1476 | 0 | 12 |

- Overall mean paired difference: +0.0797.
- Generator-family-excluded mean paired difference: +0.1303.
- Qwen FlowPilot beat Qwen one-shot in all five cases.
- GPT FlowPilot beat GPT one-shot in three cases, tied in one, and was lower by
  0.0104 in exothermic dinitration.
- Inter-judge exact agreement: 36.9%.
- Inter-judge within-one-point agreement: 69.4%.
- Mean judge absolute difference: 0.988/4.

## Interpretation

This is evidence that the FlowPilot architecture improves the smaller Qwen
model substantially and is competitive with, and slightly better on mean than,
GPT-5.4 one-shot under this five-case rubric. It is not evidence of universal
superiority because the campaign has five cases, one generation repeat per
cell, and LLM rather than human judges.

## Remaining recurring revisions

No critical errors remained. The strict OpenAI judge still requested stronger:

- explicit packed-bed holdup/contact measurement language;
- hydrogen check-valve and safe-vent connectivity in the visual topology;
- peroxide/nitration quench details supplied by a chemist;
- two-outlet routing for liquid-liquid separators;
- phase/pressure justification for atmospheric heated peroxide or nitration service;
- startup, shutdown, and flush quantities for gas processes.

These are retained as visible limitations. Chemistry-specific quench media or
endpoints must not be invented solely to improve a benchmark score.

## Files

- `summary.json`: complete numerical summary.
- `tables/criterion_judgments.csv`: all 560 criterion judgments.
- `tables/paired_comparisons.csv`: paired case-level comparison.
- `tables/judge_agreement.json`: judge agreement statistics.
- `figures/`: eight generated benchmark figures.
- Raw candidates, packets, prompts, judgments, and logs are stored in
  `ablation_results/manuscript_benchmark/manuscript_five_case_v4_latest_compiler_20260819/`.
