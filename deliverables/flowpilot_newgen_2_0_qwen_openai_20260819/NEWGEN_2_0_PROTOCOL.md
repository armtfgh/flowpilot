# NewGen 2.0 Benchmark Protocol

## Research Question

Given identical batch protocols, objectives, inventories, and held-out source evidence, does the final design outcome differ between a single-pass model and the same model family used within the FlowPilot pipeline?

## Frozen Design

- Cases: CuAAC, gas-liquid hydrogenolysis, and two-stage oxidative amidation.
- Generators: Qwen3.6-27B and GPT-5.4.
- Architectures: one-shot and FlowPilot.
- Candidates: 12 matched outputs.
- Judges: Qwen3.6-27B, GPT-5.4, and Claude Sonnet 4.6.
- Repeats: one deterministic judgment per judge/candidate pair.
- Primary score: equal-weight mean of 14 universal criterion scores, normalized to 0-1.
- Secondary outcomes: critical-error flags, judge dispersion, pairwise architecture delta, and generator-family-excluded sensitivity.

## Fairness Controls

1. The rubric is frozen before judgment.
2. Generator and architecture labels are withheld from judges.
3. Internal councils, validation traces, logs, and agent counts are excluded.
4. Both architectures are mapped into the same final-outcome section names.
5. The same protocol, inventory, source facts, and deterministic arithmetic sheet accompany both candidates in a matched case.
6. Published flow conditions are evidence, not the only acceptable answer.
7. No criterion-specific weights, architecture bonuses, score caps, or post-hoc rubric changes are permitted.
8. Only gas bookkeeping and multistage closure can be not applicable.
9. Missing required information receives a low score rather than not-applicable status.
10. Every raw response, malformed response, retry, prompt, schema, seed, and telemetry record is retained.

## Fixed Criteria

| ID | Criterion | Domain |
|---|---|---|
| UO-01 | Transformation fidelity | Chemistry |
| UO-02 | Required materials | Chemistry |
| UO-03 | Stoichiometry and feed chemistry | Chemistry |
| UO-04 | Condition and stage mapping | Chemistry |
| UO-05 | Executable topology | Process |
| UO-06 | Liquid material balance | Engineering |
| UO-07 | Residence-time and geometry closure | Engineering |
| UO-08 | Gas bookkeeping | Engineering |
| UO-09 | Multistage closure | Engineering |
| UO-10 | Inventory feasibility | Inventory |
| UO-11 | Physical and transport plausibility | Engineering |
| UO-12 | Hazards and controls | Safety |
| UO-13 | Operating procedure and work-up | Operations |
| UO-14 | Evidence and uncertainty calibration | Evidence |

## Anchors

- `4`: correct, complete, executable, and supported; no material defect.
- `3`: correct overall with a minor defect unlikely to change execution or interpretation.
- `2`: partially correct with one material but recoverable defect requiring revision.
- `1`: major or multiple material deficiencies; substantial redesign required.
- `0`: fundamentally wrong, unsafe, chemically invalid, or physically impossible.

## Calculation

For candidate `c`, criterion `k`, and valid judges `j`:

`criterion_consensus(c,k) = mean_j(score(c,k,j))`

`candidate_score(c) = mean_k(criterion_consensus(c,k)) / 4`

The primary architecture comparison is the paired difference:

`delta(model,case) = score(FlowPilot) - score(one-shot)`

The generator-family-excluded analysis recalculates each candidate after removing a judge from the same model family as the generator. Critical errors remain separate categorical outcomes and do not cap the score.

## Interpretation

This small matched study estimates comparative design quality on the selected cases. It does not establish universal superiority, and LLM judgments do not replace independent flow-chemist review or wet-lab validation. Architecture labels are withheld, but perfect architecture blinding is not claimed because differences in output style may reveal provenance.
