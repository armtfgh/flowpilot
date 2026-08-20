# NewGen 2.0 Benchmark

## Purpose

NewGen 2.0 compares final batch-to-flow design outcomes using one frozen, universal rubric. It does not award points for councils, validators, agent count, logs, or architecture-specific fields.

## Design

- 20 matched candidates: 5 held-out chemistries x 2 generator families x 2 architectures.
- Independent judge panel: qwen, openai (2 judges).
- Generator and architecture labels are withheld. Perfect architecture blinding is not claimed because output style may reveal provenance.
- Every judge sees the same protocol, objective, inventory, held-out source facts, normalized final outcome, deterministic verification sheet, and rubric.
- The reference flow procedure is evidence, not the only acceptable solution.

## Score

Each applicable criterion receives an anchored integer from 0 to 4. All criteria have equal weight. For candidate *c* and criterion *k*:

`criterion_consensus(c,k) = mean(score from every selected judge)`

`candidate_score(c) = mean(all applicable criterion_consensus) / 4`

There are no hand-tuned 0.1/0.2 weights and no architecture bonus. Only gas bookkeeping (UO-08) and multistage closure (UO-09) may be not applicable. Missing required information is scored 0 or 1.

## Criteria

UO-01 transformation fidelity; UO-02 required materials; UO-03 stoichiometry/feed chemistry; UO-04 condition/stage mapping; UO-05 executable topology; UO-06 liquid material balance; UO-07 residence-time/geometry closure; UO-08 gas bookkeeping; UO-09 multistage closure; UO-10 inventory feasibility; UO-11 transport plausibility; UO-12 hazards/controls; UO-13 operating procedure/work-up; UO-14 evidence/uncertainty calibration.

## Results

- GPT-5.4 / FlowPilot: 0.867 (family-excluded 1.000)
- GPT-5.4 / One-shot: 0.908 (family-excluded 0.996)
- Qwen3.6-27B / FlowPilot: 0.863 (family-excluded 0.726)
- Qwen3.6-27B / One-shot: 0.744 (family-excluded 0.526)

Mean matched FlowPilot-minus-one-shot delta: **+0.039**. Generator-family-excluded sensitivity delta: **+0.102**.

Judge agreement: exact 31.7%; within one point 61.9%; mean pairwise absolute difference 1.151/4.

Judge mean candidate scores: qwen 0.989; openai 0.701. This calibration difference is why both consensus and judge-specific tables are retained.

## Interpretation Limits

This is a small matched benchmark, not a universal performance claim. LLM judges can share biases and are not substitutes for wet-lab validation. Critical-error flags are reported separately and never used to manipulate or cap the numerical score. With two judges, the generator-family-excluded result is a single cross-family judgment, not an independent consensus; it is a directional sensitivity check only. Inter-judge agreement is low by the predeclared within-one-point diagnostic. The result should be described as preliminary until repeated generation and stronger independent evaluation are added.
