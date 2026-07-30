# FlowPilot Reduced Frontier Benchmark

## Design

- 5 stratified non-THQ batch-to-flow protocols.
- 3 conditions with 3 repetitions each.
- Matched protocol text and fixed condition assignment are used across models.
- Temperature 0 is used where accepted. Claude Sonnet 5 and GPT-5.6 Terra require provider-default temperature; GPT-5.6 receives the fixed benchmark seed.
- Qwen Full routes upstream, translation, engineering calculations, council, revision, and formatting through the complete FlowPilot workflow.
- Commercial comparators are general one-shot systems and do not receive FlowPilot's deterministic engineering or council modules.
- Shared OpenAI embeddings used by FlowPilot retrieval do not generate designs.
- Quality and Assurance Score v2 was frozen before this run and does not receive the condition label.
- Frozen scorer SHA-256: `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`.

## Results

| Condition | Quality v2 | Schema-neutral | Readiness v2 | Ready rate | Valid JSON | Schema valid | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 27B + Full FlowPilot | 0.910 | 0.901 | 0.696 | 20.0% | 100.0% | 100.0% | 15 |
| GPT-5.6 Terra one-shot | 0.589 | 0.655 | 0.350 | 0.0% | 100.0% | 0.0% | 15 |
| Claude Sonnet 5 one-shot | 0.392 | 0.435 | 0.252 | 0.0% | 66.7% | 0.0% | 15 |

## Predefined Paired Comparisons

| Comparison | Mean delta | 95% case-bootstrap CI | W-L-T |
|---|---:|---:|---:|
| Qwen 27B + Full FlowPilot minus Claude Sonnet 5 one-shot | +0.519 | [+0.385, +0.657] | 5-0-0 |
| Qwen 27B + Full FlowPilot minus GPT-5.6 Terra one-shot | +0.321 | [+0.262, +0.380] | 5-0-0 |
| Claude Sonnet 5 one-shot minus GPT-5.6 Terra one-shot | -0.197 | [-0.291, -0.108] | 0-5-0 |

## Schema-Neutral Sensitivity

The formal-validity dimension (10% weight) is removed and the remaining dimensions are renormalized.

| Comparison | Mean delta | 95% case-bootstrap CI | W-L-T |
|---|---:|---:|---:|
| Qwen 27B + Full FlowPilot minus Claude Sonnet 5 one-shot | +0.465 | [+0.317, +0.618] | 5-0-0 |
| Qwen 27B + Full FlowPilot minus GPT-5.6 Terra one-shot | +0.246 | [+0.180, +0.311] | 5-0-0 |
| Claude Sonnet 5 one-shot minus GPT-5.6 Terra one-shot | -0.219 | [-0.324, -0.120] | 0-5-0 |

## Interpretation Rules

- Qwen Full versus either commercial one-shot condition is a system-level comparison, not a pure base-model comparison.
- Claude Sonnet 5 versus GPT-5.6 Terra compares one-shot model behavior under the same prompt contract.
- Quality and deployment readiness are separate outcomes; a screened design is not immediately executable.
- The score rewards visible calculation, provenance, and independent audit evidence. This is appropriate for assurance evaluation but structurally favors systems that emit such evidence.
- Five Claude Sonnet 5 responses reached the output-token limit with malformed or empty JSON and are scored as formal-validity failures without retrying.
- Valid one-shot JSON still failed strict FlowProposal field typing; the resulting readiness cap measures contract noncompliance, not automatic chemical invalidity.
- The schema-neutral sensitivity removes the formal-validity dimension; it does not remove engineering, safety, evidence, or assurance requirements.

## Limitations

- 5 protocols are deliberately cost-controlled and do not establish wet-lab yield superiority.
- Provider execution may vary because current Claude and GPT models do not accept temperature zero.
- Qwen Full receives architecture and deterministic tooling unavailable to the commercial one-shot baselines.
- Local-model throughput and frontier API latency are hardware/provider dependent.
- Hidden literature references are machine-extracted and reference agreement is secondary.

Raw prompts, completions, snapshots, metrics, and checksums: `ablation_test/runs/cross_model_publication_reduced_frontier_5x3x3_20260730`.
