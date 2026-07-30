# Reduced Frontier Benchmark Package

## Included Study

- Five non-THQ protocols selected across catalytic thermal, photochemical
  gas-liquid, packed-bed hydrogenation, hazardous exothermic, and multistep
  chemistry.
- Three conditions: Qwen 27B with full FlowPilot, Claude Sonnet 5 one-shot,
  and GPT-5.6 Terra one-shot.
- Three independent provider calls per protocol and condition.
- Forty-five matched scored observations.

## Raw Data

Raw run directory:
`ablation_test/runs/cross_model_publication_reduced_frontier_5x3x3_20260730`

Each cell retains its protocol, hidden reference, metadata, prompt, raw model
response, LLM event log, normalized result, metrics, run summary, and checksum
manifest. Full FlowPilot cells additionally retain council and intermediate
snapshots.

Five Claude calls reached `max_tokens` and returned malformed or empty JSON.
They were not retried. Their original responses are retained and scored as
`schema_valid=false`.

## Validation

- Planned and matched observations: 45/45.
- Leave-one-source-out check: 45/45 passed.
- Qwen provenance: 402 events from `/models/Qwen3.6-27B`.
- Claude provenance: 15 events from `claude-sonnet-5`.
- OpenAI provenance: 15 events from `gpt-5.6-terra`.
- Frozen scorer SHA-256:
  `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`.
- Complete ablation-test suite: 21 passed.
- Per-run checksum verification: passed.

## Interpretation

This is a system-level comparison. Qwen Full receives FlowPilot's intake,
retrieval, deterministic engineering, inventory enforcement, specialist
council, validation, and formatting architecture. Claude and GPT are general
one-shot baselines. The benchmark supports an architecture-suitability claim,
not a claim that Qwen 27B is generally more capable than the commercial base
models or that FlowPilot improves wet-lab yield.
