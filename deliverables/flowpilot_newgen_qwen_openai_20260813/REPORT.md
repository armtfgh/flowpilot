# FlowPilot NewGen Qwen/OpenAI Benchmark

## Headline

The full architecture improved the universal quality-assurance score in all six matched model-by-case comparisons.

| Model | One-shot mean QA | FlowPilot mean QA | Uplift | Full executable designs |
|---|---:|---:|---:|---:|
| Qwen3.6-27B | 0.6737 | 0.9580 | +0.2843 | 3/3 |
| GPT-5.4 | 0.7417 | 0.9533 | +0.2116 | 3/3 |

Qwen FlowPilot (0.9580) and GPT FlowPilot (0.9533) are nearly equal on this small suite. This supports an architecture-efficiency argument, not model equivalence in general.

## Design

- Three cases: CuAAC packed-bed chemistry, gas-liquid-solid hydrogenolysis, and two-stage oxidative amidation.
- Same protocol, objective, inventory, candidate budget, scorer, and held-out-source policy within each pair.
- Qwen condition: `/models/Qwen3.6-27B` for one-shot and every FlowPilot LLM role.
- OpenAI condition: `gpt-5.4-2026-03-05` for one-shot and every FlowPilot LLM role.
- FlowPilot adds chemistry parsing, held-out-safe retrieval, deterministic calculations, specialist council, design realization, inventory allocation, safety validation, and executable topology generation.

## Scoring

The fixed QA score combines formal validity (10%), engineering integrity (25%), process completeness (15%), safety adequacy (15%), evidence provenance (10%), decision assurance (20%), and actionability calibration (5%). Deployment gates cap readiness for schema, gas-bookkeeping, geometry, pump/tubing, topology, and critical-safety failures.

## Interpretation

The benchmark supports the statement that FlowPilot improves design assurance relative to an identical-model one-shot baseline on these cases. It does not establish statistical significance, universal superiority, or wet-lab yield accuracy. Every full design is labeled `SCREEN` pending experimental validation.
