# NewGen Three-LLM Judge Benchmark

> **WITHDRAWN FOR COMPARATIVE CLAIMS:** A retrospective validity audit found non-equivalent evidence packets, architecture leakage, an assurance extraction bug, order-sensitive pairwise counting, low judge agreement, same-family bias, and unstable seed bookkeeping. Do not use the scores or winner figures as performance evidence. See `BENCHMARK_REAUDIT.md`.

## Scope

This benchmark independently evaluates the same 12 frozen batch-to-flow outputs with Qwen 27B, GPT-5.4, and Claude Sonnet 4.6. Candidate identity, generator family, and architecture are blinded. No judge is part of the generation pipeline.

Outcome quality and assurance quality are reported separately. Scores are integer 0-4 criterion judgments; Python applies the predeclared weights and scales results to 0-100. There is no manual score adjustment.

## Completion

- Absolute judgments: 216/216 valid.
- Pairwise judgments: 72/72 valid.
- Absolute design: 3 judges x 3 repeats x 12 candidates x 2 tracks.
- Pairwise design: 3 judges x 2 reversed A/B orders x 6 matched pairs x 2 tracks.

## Absolute Results

```text
generator_model  architecture  track    
GPT-5.4          FlowPilot     assurance    38.8
                               outcome      67.5
                 One-shot      assurance    52.9
                               outcome      73.8
Qwen3.6-27B      FlowPilot     assurance    32.9
                               outcome      56.7
                 One-shot      assurance    16.7
                               outcome      57.9
```

FlowPilot minus One-shot score differences:

```text
architecture               FlowPilot_minus_One-shot
generator_model track                              
GPT-5.4         assurance                     -14.2
                outcome                        -6.2
Qwen3.6-27B     assurance                      16.2
                outcome                        -1.2
```

## Pairwise Results

Overall preference votes across judges and reversed orders:

```text
generator_model     track  FlowPilot  One-shot  TIE  total
        GPT-5.4 assurance          8         9    1     18
        GPT-5.4   outcome          2        16    0     18
    Qwen3.6-27B assurance         18         0    0     18
    Qwen3.6-27B   outcome          8        10    0     18
```

## Main Finding

This campaign does **not** establish general FlowPilot superiority. GPT-5.4 one-shot leads GPT-5.4 FlowPilot on both absolute tracks and receives 16/18 direct outcome votes. Qwen FlowPilot strongly improves assurance over Qwen one-shot, but Qwen outcome is approximately tied in absolute scoring and slightly loses the total direct outcome vote (8/18 versus 10/18).

The case audit shows that the judges are responding to substantive residual inconsistencies rather than only output length. The GPT-5.4 multistep pipeline candidate is the clearest failure: its realized inventory volume and engineering trace disagree, and it loses all six overall pairwise votes on both tracks. Consequently, this benchmark should be used as a pipeline defect-discovery result and a preregistered baseline, not as a superiority figure.

## Reliability

Within-judge exact repeatability:

```text
judge   track    
claude  assurance    0.722
        outcome      0.847
openai  assurance    0.667
        outcome      0.778
qwen    assurance    0.750
        outcome      0.847
```

Reversed-order consistency:

```text
judge   track    
claude  assurance    0.833
        outcome      0.619
openai  assurance    0.833
        outcome      0.714
qwen    assurance    0.833
        outcome      0.548
```

Inter-judge diagnostics:

```text
    track  targets  judge_count  icc_2_1_absolute  exact_agreement_rate  mean_pairwise_spearman
assurance       72            3             0.374                 0.097                   0.514
  outcome       72            3             0.373                 0.083                   0.517
```

Mean own-family minus cross-family score difference (positive indicates favorable same-family scoring):

```text
generator_model  track    
GPT-5.4          assurance    -4.62
                 outcome      -2.67
Qwen3.6-27B      assurance     9.97
                 outcome      31.88
```

## Interpretation Rules

- Treat outcome score and pairwise outcome preference as primary endpoints.
- Treat assurance, agreement, repeatability, order consistency, and family-bias sensitivity as secondary diagnostics.
- LLM consensus is an automated evaluation, not a substitute for wet-lab validation or blinded flow-chemist review.
- The benchmark contains three chemistry cases and one frozen generated output per model-architecture-case cell; generalization beyond these cases requires a larger preregistered campaign.
- Pairwise judgments control for score calibration differences, while reversed order measures position sensitivity.

## Traceability

Every prompt, raw response, parsed response, validation result, timing record, frozen packet, blinding key, and source hash is retained in the campaign directory. Tables in `tables/` are mechanically generated from valid call records.
