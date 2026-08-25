# NewGen 2.0: All-model three-repeat summary

This package compares the same three protocols, two architectures, three generation repeats, and three blinded judge families for every included model. No incomplete model is admitted.

## Statistical unit

For each model and architecture, each repeat first averages the three case scores. The reported mean and sample SD are then calculated across those three repeat-level campaign means (n=3, ddof=1). Judge disagreement is retained separately in the candidate table and is not substituted for generation-repeat variability. Error bars show sample SD, not confidence intervals.

## Summary

| Model | One-shot mean ± SD | FlowPilot mean ± SD | Paired delta mean ± SD |
|---|---:|---:|---:|
| Qwen3.6-27B | 0.765 ± 0.014 | 0.904 ± 0.029 | +0.139 ± 0.042 |
| Qwen3.8-27B | 0.747 ± 0.019 | 0.915 ± 0.011 | +0.168 ± 0.016 |
| GPT-4o | 0.684 ± 0.011 | 0.910 ± 0.002 | +0.226 ± 0.012 |
| GPT-5.4 | 0.909 ± 0.021 | 0.926 ± 0.008 | +0.017 ± 0.027 |
| Claude Sonnet 4.6 | 0.890 ± 0.014 | 0.924 ± 0.024 | +0.034 ± 0.036 |
| Claude Opus 4.6 | 0.883 ± 0.015 | 0.932 ± 0.010 | +0.049 ± 0.019 |
| Claude Opus 5 | 0.936 ± 0.023 | 0.772 ± 0.087 | -0.164 ± 0.064 |

## Interpretation boundary

These are outcome-quality and repeatability measurements for the frozen cases and inventory. They support matched architecture comparisons but do not establish universal superiority over all chemistry domains. With n=3, uncertainty remains substantial; the paired case-repeat table must accompany aggregate claims.
