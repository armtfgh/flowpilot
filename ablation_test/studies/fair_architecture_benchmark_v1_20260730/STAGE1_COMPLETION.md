# Stage 1 Completion Record

Study ID: `fair_architecture_benchmark_v1_20260730`

Completion date: 2026-07-30

Original gate decision: **STAGE_2_CLEARED**

Post-gate scientific audit: **SUPERSEDED - DO NOT USE FOR STAGE 2**

The infrastructure gate was valid, but subsequent proposal-level review found
that the nominally feasible photochemical inventory omitted gas-delivery and
gas-liquid-mixing hardware even though the objective required a controlled
gas-liquid experiment. Consequently, GPT-5.6 Terra's `BLOCK` response was
scientifically defensible and the expected `SCREEN` label was ambiguous.

No result was deleted or rescored. This study version is retained as an audit
record. The corrected benchmark is
`fair_architecture_benchmark_v1_1_20260730`, where both photochemical scenarios
contain identical gas hardware and differ only in available wavelength.

## Frozen benchmark definition

- Benchmark config SHA256:
  `9882f3088462fb9a7bae587e81088a466d8086bd9d81ed4e354384fb5ce11503`
- Paired scenarios SHA256:
  `b3a144ebc8e70d17441bd439c3a3c528695df3fd0c3ac689464137247fa4fce6`
- Deterministic Stage 1 oracle SHA256:
  `c1e6b5e50393b68f34a9e4fb718cc655c5a1e5f8595eb27cdbdc009d98dadf4d`
- Secondary quality scorer SHA256:
  `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`

## Scenario set

Five protocol pairs were defined:

1. Suzuki-Miyaura coupling: feasible 50 C hardware vs 40 C hardware ceiling.
2. Photochemical aerobic oxidation: matched 420 nm source vs 525 nm-only source.
3. Packed-bed hydrogenolysis: certified hydrogen service vs hydrogen prohibited.
4. Dinitration: approved acid-compatible path vs laboratory prohibition.
5. Two-step oxidative amidation: interstage hardware available vs unavailable.

Each pair differs by one controlled inventory or operating-limit change. The
smoke profile used the feasible/infeasible photochemical pair.

## Run history

### Attempt 01

Directory: `stage1_smoke_run_20260730_attempt01`

Result: **HOLD**

The live smoke test discovered that Anthropic structured output rejects object
schemas with `additionalProperties: true`. Both Claude cells were retained as
failed adapter records. No output was deleted or rewritten.

### Attempt 02

Directory: `stage1_smoke_run_20260730_attempt02`

Result: **PASS**

The common one-shot contract used a strict provider-native envelope:

- `recommended_disposition`
- `disposition_rationale`
- `proposal_json`

All eight cells completed. All providers and configured models were available,
no condition showed systemic truncation, all required artifacts were present,
scenario input hashes matched across all four conditions, and all ten oracle
witnesses passed.

## Smoke observations

Disposition accuracy was 6/8 (75%). This was not a Stage 1 gate.

- Qwen 27B one-shot: 2/2 correct.
- Qwen 27B + full FlowPilot: 1/2 correct; the infeasible wavelength case was
  classified as `SCREEN` instead of `BLOCK`.
- Claude Sonnet 5 one-shot: 2/2 correct.
- GPT-5.6 Terra one-shot: 1/2 correct; the feasible case was classified as
  `BLOCK` instead of `SCREEN`.

These two outcomes are retained without benchmark tuning. Stage 1 is an
infrastructure and data-integrity smoke test, not a comparative performance
estimate.

## Verification

- Stage 1 focused and compatibility tests: 14 passed.
- Complete `ablation_test/tests` suite: 27 passed.
- Attempt-2 checksum verification: 146 files passed.
- Independent artifact audit confirmed that the byte-identical public input and
  inventory files were used across all four conditions and that the exact
  canonical design text appeared in each condition's recorded prompt stream.
- Accepted gate record:
  `stage1_smoke_run_20260730_attempt02/gate_results.json`
