# Qwen Architecture Outcome Benchmark v1

This benchmark compares the same Qwen3.6-27B model used in two architectures:

1. Direct one-shot batch-to-flow translation.
2. The complete FlowPilot pipeline.

The comparison reuses the frozen public inputs, seeds, and 60 saved outputs from
the paired confirmatory study. It does not call either model again.

The scoring specification is deterministic and intentionally separates:

- executable design rate,
- numerical and inventory quality,
- paired win/tie/loss,
- infeasible-case blocking,
- reproducibility.

A design is not assigned zero merely because it came from the one-shot
condition. Every structured field is evaluated by the same rules. Invalid
designs are shown explicitly as hard failures rather than hidden inside a
single aggregate score.

This is a retrospective pilot because the model outputs existed before the
richer evaluator was written. A future confirmatory benchmark should freeze the
same evaluator before generating outputs on unseen protocols.
