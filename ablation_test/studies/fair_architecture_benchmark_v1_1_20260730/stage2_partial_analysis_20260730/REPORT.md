# Stage 2 Partial Matched Benchmark Report

Study: `fair_architecture_benchmark_v1_1_20260730`

Status: **PARTIAL - 90/120 cells complete**

Claude Sonnet 5 one-shot is absent because Anthropic rejected generation
with an explicit insufficient-credit error. The 30 Claude cells remain pending
and are not counted as scientific failures.

## Primary result

The available results do **not** support superiority of the current full
FlowPilot pipeline. Joint success requires both the correct `SCREEN/BLOCK`
decision and satisfaction of all critical deterministic constraints.

| Condition | Joint success | Feasible design | Infeasible block | Violations | Median runtime |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Terra one-shot | 25/30 (83.3%) | 66.7% | 100.0% | 0 | 33.7 s |
| Qwen 27B + FlowPilot | 15/30 (50.0%) | 100.0% | 0.0% | 9 | 439.4 s |
| Qwen 27B one-shot | 27/30 (90.0%) | 80.0% | 100.0% | 5 | 18.4 s |

Full FlowPilot succeeded on 100.0% of
feasible cells, outperforming Qwen one-shot
(80.0%) and GPT-5.6 Terra
(66.7%) on that subset. However, it
returned `SCREEN` for every infeasible cell, yielding
0.0% correct blocks. Both one-shot
baselines correctly blocked every infeasible cell.

This is a fundamental pipeline defect: detected council violations are not
being promoted into the final top-level disposition. It is not evidence that
Qwen 27B lacks chemistry capability.

## Paired scenario analysis

The uncertainty intervals resample the ten scenario clusters, not the 30
repeat cells, avoiding treatment of repeats as independent protocols.

| Comparison | Mean difference | 95% cluster bootstrap | Better / tied / worse clusters |
|---|---:|---:|---:|
| FlowPilot - GPT-5.6 Terra one-shot | -33.3 pp | [-76.7, +10.0] pp | 3 / 2 / 5 |
| FlowPilot - Qwen 27B one-shot | -40.0 pp | [-80.0, +0.0] pp | 2 / 3 / 5 |

## Repeatability

Disposition agreement is not sufficient evidence of quality: FlowPilot is
highly consistent because it always returns `SCREEN`, including when `BLOCK`
is required. The stricter measure is the fraction of scenario clusters where
all three repeats jointly succeed.

- GPT-5.6 Terra one-shot: 70.0% disposition agreement; 70.0% all-repeat joint success.
- Qwen 27B + FlowPilot: 100.0% disposition agreement; 50.0% all-repeat joint success.
- Qwen 27B one-shot: 80.0% disposition agreement; 80.0% all-repeat joint success.

## Compute burden

Full FlowPilot used a mean of 16.2 LLM calls and
64012 tokens per cell, versus one call and
1654 tokens for Qwen one-shot. Its median runtime
was 439.4 seconds versus
18.4 seconds. The architecture must therefore
deliver materially better decisions to justify its current cost; this partial
run does not show that benefit.

## Required correction before confirmatory rerun

1. Add a deterministic final disposition gate after council validation.
2. Emit `BLOCK` when no candidate satisfies a hard inventory constraint.
3. Prevent downstream formatting from converting council critical failures
   into a generic `SCREEN` result.
4. Add regression tests for all five infeasible perturbations.
5. Freeze a new pipeline version, then rerun the same untouched benchmark as
   a separately labeled confirmatory study. Do not overwrite this run.

## Figures

![Joint success](figures/01_joint_success.png)

![Feasible and infeasible split](figures/02_feasible_vs_infeasible.png)

![Paired effects](figures/03_paired_architecture_effect.png)

![Compute cost](figures/04_compute_cost.png)

![Repeatability](figures/05_repeatability.png)

## Integrity

- Run state: `PARTIAL_RESUMABLE`.
- Completed cells: `90`.
- Remaining Claude cells: `30`.
- All run checksums verified after the preserved transport retry.
- Full test suite for `ablation_test/tests`: 31 passed.
