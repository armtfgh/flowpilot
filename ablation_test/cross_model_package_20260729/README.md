# FlowPilot Qwen 27B versus Frontier One-Shot Benchmark

## Design

- Twelve non-THQ batch-to-flow protocols.
- Three repetitions per condition unless the execution plan states otherwise.
- Temperature 0 with matched protocol text and fixed model-specific condition.
- Qwen Full routes upstream, translation, council, revision, and formatting to Qwen 27B.
- Qwen uses FlowPilot's lightweight local-model upstream adapter; GPT-4o uses the full-schema upstream path.
- Both full conditions use the same shared OpenAI embedding backend for retrieval; embeddings do not generate designs.
- Quality and Assurance Score v2 was frozen before this run and does not receive the condition label.
- Frozen scorer SHA-256: `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`.

## Results

| Condition | Quality v2 | Readiness v2 | Ready rate | Schema valid | Runtime (s) | Calls | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 27B + Full FlowPilot | 0.921 | 0.719 | 22.2% | 100.0% | 474.2 | 23.2 | 36 |
| GPT-4o + Full FlowPilot | 0.895 | 0.743 | 33.3% | 100.0% | 216.7 | 39.7 | 36 |
| Claude Sonnet 4.6 one-shot | 0.588 | 0.350 | 0.0% | 0.0% | 80.2 | 1.0 | 36 |
| GPT-4o one-shot | 0.550 | 0.350 | 0.0% | 0.0% | 12.2 | 1.0 | 36 |
| Qwen 27B one-shot | 0.545 | 0.350 | 0.0% | 0.0% | 27.4 | 1.0 | 36 |

## Predefined Paired Comparisons

| Comparison | Mean delta | 95% case-bootstrap CI | W-L-T |
|---|---:|---:|---:|
| Qwen 27B + Full FlowPilot minus Qwen 27B one-shot | +0.375 | [+0.348, +0.401] | 12-0-0 |
| Qwen 27B + Full FlowPilot minus GPT-4o one-shot | +0.371 | [+0.343, +0.400] | 12-0-0 |
| Qwen 27B + Full FlowPilot minus Claude Sonnet 4.6 one-shot | +0.332 | [+0.303, +0.362] | 12-0-0 |
| GPT-4o + Full FlowPilot minus GPT-4o one-shot | +0.346 | [+0.310, +0.383] | 12-0-0 |
| GPT-4o + Full FlowPilot minus Qwen 27B + Full FlowPilot | -0.025 | [-0.040, -0.011] | 1-9-2 |

## Interpretation Rules

- Qwen Full versus Qwen one-shot estimates the Qwen architecture gain.
- Qwen Full versus GPT-4o or Claude one-shot is a system-level comparison, not a pure model comparison.
- GPT-4o Full versus Qwen Full compares provider-specific full-pipeline profiles; the upstream adapters differ.
- Quality and deployment readiness are separate outcomes; a screened design is not immediately executable.
- The score rewards visible calculation, provenance, and independent audit evidence. This is appropriate for assurance evaluation but structurally favors systems that emit such evidence.
- One-shot responses were parseable JSON but failed strict FlowProposal field typing; their 0.35 readiness cap measures contract noncompliance, not automatic chemical invalidity.

## Limitations

- Twelve protocols remain a small benchmark and do not establish wet-lab yield superiority.
- Provider execution may vary even at temperature zero.
- Qwen and GPT use provider-specific structured-output handling and different upstream adapters.
- Local-model throughput and frontier API latency are hardware/provider dependent.
- Hidden literature references are machine-extracted and reference agreement is secondary.

Raw prompts, completions, snapshots, metrics, and checksums: `ablation_test/runs/cross_model_publication_qwen_frontier_20260729`.
