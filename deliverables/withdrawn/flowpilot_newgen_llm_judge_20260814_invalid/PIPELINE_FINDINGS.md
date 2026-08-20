# FlowPilot Findings From the Three-Judge Benchmark

## Bottom line

This benchmark does not show general FlowPilot superiority. It establishes a reproducible baseline and identifies where the current pipeline still loses to a one-shot model.

## Outcome results

| Generator | FlowPilot | One-shot | Difference |
|---|---:|---:|---:|
| GPT-5.4 | 67.5 | 73.8 | -6.3 |
| Qwen3.6-27B | 56.7 | 57.9 | -1.3 |

Direct outcome votes were 2 FlowPilot versus 16 one-shot for GPT-5.4, and 8 versus 10 for Qwen. The Qwen result is case-dependent: FlowPilot wins hydrogenolysis, ties at the pair-consensus level for CuAAC, and loses two-stage amidation.

## Assurance results

| Generator | FlowPilot | One-shot | Difference |
|---|---:|---:|---:|
| GPT-5.4 | 38.8 | 52.9 | -14.2 |
| Qwen3.6-27B | 32.9 | 16.7 | +16.3 |

Qwen FlowPilot provides materially better assurance than Qwen one-shot. GPT-5.4 FlowPilot does not, primarily because its richer record exposes unresolved contradictions that the judges penalize.

## Main technical failure

The final contract and supporting traces are not always the same design. The GPT-5.4 two-stage candidate is the clearest example: its final inventory reactor volume is 15.04 mL while an engineering trace reports 15.69 mL. Similar stale or non-authoritative intermediate values reduce numerical consistency, auditability, and actionability. A larger record is harmful when it contains conflicting versions.

## Required pipeline changes

1. Make the executable final-design contract the only numerical authority after realization.
2. Regenerate calculations, stream tables, topology, equipment assignments, recipe, and GUI fields from that contract.
3. Remove stale pre-realization numbers from the published assurance record, or label them explicitly as rejected candidates.
4. Add cross-artifact equality gates for reactor volume, flow, residence time, pressure, temperature, concentration, stage totals, and equipment IDs.
5. Block publication of a final design when any displayed trace disagrees with the final contract, even if each artifact passes its own local schema.
6. Improve chemistry-specific safety and execution details; judges consistently scored these below chemistry preservation and inventory feasibility.

## Evaluation cautions

Inter-judge ICC is approximately 0.37 on both tracks, so model judges have only modest absolute agreement. Qwen also shows substantial favorable same-family bias. The leave-own-family-out analysis must accompany headline scores. The benchmark should be rerun unchanged only after assigning a new pipeline version and recording the code changes; this baseline must remain immutable.
