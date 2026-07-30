# FlowPilot Fair Architecture Benchmark v1.1

Study ID: `fair_architecture_benchmark_v1_1_20260730`

This version supersedes `fair_architecture_benchmark_v1_20260730`.

The v1 post-gate audit found that the feasible photochemical inventory omitted
the gas-delivery and gas-liquid-mixing hardware required by its objective. In
v1.1, both photochemical scenarios contain the same explicit gas hardware and
differ only in the controlled wavelength perturbation: 420 nm versus 525 nm.
No model or FlowPilot behavior was changed in response to the smoke outputs.

This directory is the immutable study namespace for the fair comparison of:

1. Qwen 27B one-shot
2. Qwen 27B inside full FlowPilot
3. Claude Sonnet 5 one-shot
4. GPT-5.6 Terra one-shot

All systems receive the same external design-input text: raw protocol, objective,
hard constraints, and complete inventory. The full architecture may use its
internal retrieval, deterministic engineering, and council because those are the
system components under study.

## Stage 1

Stage 1 defines five protocol pairs. Each pair contains one feasible scenario and
one infeasible scenario created by one controlled inventory or operating-limit
change. The smoke run uses the photochemical wavelength pair, producing eight
cells: two scenarios by four conditions by one repeat.

The Stage 1 gate tests infrastructure and data integrity. Model accuracy is
reported but is deliberately not a gate, preventing outcome-dependent benchmark
changes.

## Frozen items

- `benchmark_config.json`: conditions, models, seed, and gate.
- `protocol_scenarios.json`: public inputs, hidden expected dispositions, and
  deterministic oracle witnesses.
- `ablation_test/src/stage1_oracle.py`: architecture-independent checker.
- `ablation_test/src/metrics.py`: legacy quality scorer retained as a secondary
  measure.

Hashes are recorded in the smoke run execution plan and package checksum file.
