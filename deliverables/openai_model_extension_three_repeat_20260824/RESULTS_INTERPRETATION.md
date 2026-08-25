# OpenAI Model Extension: Results Interpretation

## Campaign

- Exact generator snapshots: `gpt-5.5-2026-04-23`, `gpt-5.4-mini-2026-03-17`, and `gpt-5.2-2025-12-11`.
- Three frozen chemistries, two matched architectures, and three repeats: 54 generation cells.
- Three blinded judges (Qwen, GPT-5.4, Claude Sonnet 4.6): 162 valid judgments.
- The earlier GPT-5.4 campaign was not overwritten or pooled into this extension.

## Primary Results

| Generator | One-shot | FlowPilot | Matched delta | 95% CI | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| GPT-5.2 | 0.909 | 0.825 | -0.084 | -0.319 to +0.151 | 4/1/4 |
| GPT-5.4 mini | 0.835 | 0.895 | +0.060 | +0.003 to +0.118 | 6/2/1 |
| GPT-5.5 | 0.962 | 0.649 | -0.313 | -0.534 to -0.092 | 1/0/8 |

The prior frozen GPT-5.4 campaign remains the relevant same-model baseline: one-shot 0.909, FlowPilot 0.926, matched delta +0.017.

## What Drives The Results

### GPT-5.4 mini

FlowPilot closed all 9/9 designs and improved the matched score in six pairs, tied twice, and lost once. This is the cleanest positive architecture result in the extension and supports the claim that the pipeline can lift a smaller model.

### GPT-5.2

Eight FlowPilot designs were executable. One photochemical repeat delivered no schema-valid proposal after all three frozen attempts and received a score near zero; it remains in the primary result. Removing that failed pair only as a diagnostic sensitivity calculation changes the mean matched delta from -0.084 to approximately +0.017 across the eight delivered pairs. The primary reported result remains -0.084.

### GPT-5.5

Only 4/9 FlowPilot outcomes closed as executable; five were deterministically blocked. All three hydrogenolysis repeats and two CuAAC repeats were blocked after the model's stream decomposition required two liquid pumps while the frozen inventory declared one. The executable GPT-5.5 FlowPilot pairs were much closer to one-shot than the aggregate suggests, but blocked outcomes correctly remain in the primary score.

This is not evidence that GPT-5.5 lacks chemistry capability: its one-shot score was the highest in the campaign. It is evidence of a model-pipeline interface problem. The same protocol and inventory can be decomposed into different stream counts by different generators, and the current capability preflight treats those decompositions as hard equipment requirements. That behavior should be investigated before using GPT-5.5 as the default FlowPilot generator.

## Benchmark Reliability Notes

- One failed generation cell was converted to an explicit blinded `GENERATION_FAILED` packet instead of being deleted or selectively rerun.
- Two Qwen judge responses were initially truncated while serializing the 14-row JSON. Their immutable attempts are retained; concise schema-repair retries produced valid complete judgments without changing the rubric.
- Judge agreement was 40.5% exact, 81.1% within one rubric point, and 0.557/4 mean pairwise absolute difference.
- These are three-case estimates. They support model-selection and defect diagnosis, not a universal ranking over flow chemistry.
