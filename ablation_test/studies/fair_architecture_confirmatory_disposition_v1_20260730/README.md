# FlowPilot Disposition-Gate Confirmatory Benchmark

Study ID: `fair_architecture_confirmatory_disposition_v1_20260730`

This is a separately labeled post-fix confirmatory run. It does not overwrite
the negative pre-fix study at
`fair_architecture_benchmark_v1_1_20260730/stage2_matched_run_20260730`.

The ten scenario inputs are byte-for-byte identical to the corrected v1.1
benchmark. The only FlowPilot intervention is the new deterministic
post-council disposition gate and its supporting structured inventory checks.
The one-shot prompts and oracle are unchanged.

## Conditions

1. Qwen 27B one-shot.
2. Qwen 27B inside full FlowPilot.
3. GPT-5.6 Terra one-shot.

Each condition runs ten scenarios with three repeats, for 90 planned cells.
Claude is intentionally excluded from this confirmatory study at the user's
request.

## Primary Endpoint

Joint success requires:

1. Correct `SCREEN` versus `BLOCK` disposition.
2. Every critical deterministic engineering constraint passing.

Feasible-design success and infeasible-block success are reported separately
to prevent an always-block or always-screen policy from appearing strong.
