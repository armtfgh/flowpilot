# Stage 2 Partial Status

Study ID: `fair_architecture_benchmark_v1_1_20260730`

Run: `stage2_matched_run_20260730`

Analysis: `stage2_partial_analysis_20260730`

Status: **PARTIAL_RESUMABLE - 90/120 CELLS COMPLETE**

## Completion

- Qwen 27B one-shot: 30/30 completed.
- Qwen 27B + full FlowPilot: 30/30 completed.
- GPT-5.6 Terra one-shot: 30/30 completed.
- Claude Sonnet 5 one-shot: 0/30 pending.

Anthropic rejected generation with an explicit insufficient-credit error. The
Claude cells are pending external service restoration and are not scored as
scientific failures.

One full-FlowPilot cell encountered an OpenAI embeddings transport timeout. The
failed attempt is retained under `failed_attempts/`, and the frozen cell was
rerun successfully with identical input and seed.

## Current Result

Joint success requires both the correct design disposition and satisfaction of
all critical deterministic engineering constraints.

| Condition | Joint success | Feasible design | Infeasible block |
|---|---:|---:|---:|
| Qwen 27B one-shot | 27/30 (90.0%) | 80.0% | 100.0% |
| Qwen 27B + full FlowPilot | 15/30 (50.0%) | 100.0% | 0.0% |
| GPT-5.6 Terra one-shot | 25/30 (83.3%) | 66.7% | 100.0% |

The current full pipeline does not demonstrate overall superiority. It produces
valid designs consistently for feasible scenarios, but it returns `SCREEN` for
every infeasible scenario instead of promoting hard-constraint failure to
`BLOCK`. This is a final-disposition architecture defect and must be fixed in a
new version before a separately labeled confirmatory rerun.

## Integrity

- The run contains 90 completed cell records and no unresolved failed cell.
- The original transport failure is preserved as provenance.
- Run checksums pass.
- All 31 ablation tests pass.
- The report uses scenario-cluster bootstrap intervals so three repeats are not
  treated as 30 independent protocols.

See `stage2_partial_analysis_20260730/REPORT.md` for tables and figures.
