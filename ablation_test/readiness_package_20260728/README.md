# FlowPilot Frozen-v2 Readiness Benchmark

## Scope

- Development set: six protocols used for implementation diagnosis.
- Untouched holdout: six distinct protocols evaluated after the v2 score was frozen.
- Seven matched architectures per split, one run per protocol and architecture.
- Primary score uses output evidence only; the architecture label is not an input.
- Frozen scorer SHA-256: `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`.

## Mean Results

| Split | Architecture | Quality v2 | Readiness v2 | Ready rate | Cases |
|---|---|---:|---:|---:|---:|
| development | Full FlowPilot | 0.920 | 0.763 | 33.3% | 6 |
| development | No council | 0.871 | 0.871 | 100.0% | 6 |
| development | No inventory | 0.853 | 0.600 | 0.0% | 6 |
| development | No retrieval | 0.852 | 0.779 | 50.0% | 6 |
| development | No engineering | 0.747 | 0.693 | 66.7% | 6 |
| development | General one-shot | 0.550 | 0.350 | 0.0% | 6 |
| development | Structured single agent | 0.538 | 0.350 | 0.0% | 6 |
| holdout | Full FlowPilot | 0.858 | 0.684 | 16.7% | 6 |
| holdout | No council | 0.793 | 0.773 | 83.3% | 6 |
| holdout | No retrieval | 0.778 | 0.642 | 0.0% | 6 |
| holdout | No inventory | 0.774 | 0.600 | 0.0% | 6 |
| holdout | No engineering | 0.682 | 0.676 | 66.7% | 6 |
| holdout | Structured single agent | 0.536 | 0.350 | 0.0% | 6 |
| holdout | General one-shot | 0.520 | 0.350 | 0.0% | 6 |

## Interpretation

Quality and deployment readiness are reported separately. A high-quality, well-audited refusal can score well for assurance but is not counted as an immediately executable design. Conversely, a numerically complete answer is capped when gas bookkeeping, geometry, hardware feasibility, topology, safety, or screening status triggers a deployment gate.

On the untouched holdout, the highest mean quality was **Full FlowPilot (0.858)**.
Full FlowPilot scored 0.858 quality and 0.684 deployment readiness, with 16.7% of cases immediately executable.

The primary quality result and deployment result must not be conflated. Full FlowPilot leads the architecture-blind quality score because it provides calculation, provenance, safety, and independent audit evidence. It does not lead immediate executability because five of six holdout cases were conservatively marked for screening.

## Paired Full-System Comparisons

| Split | Comparator | Mean delta | 95% bootstrap interval | W-L-T |
|---|---|---:|---:|---:|
| development | No council | +0.049 | [+0.021, +0.075] | 5-1-0 |
| development | No inventory | +0.068 | [+0.055, +0.079] | 6-0-0 |
| development | No retrieval | +0.068 | [+0.040, +0.095] | 6-0-0 |
| development | No engineering | +0.173 | [+0.125, +0.222] | 6-0-0 |
| development | General one-shot | +0.370 | [+0.308, +0.443] | 6-0-0 |
| development | Structured single agent | +0.382 | [+0.355, +0.406] | 6-0-0 |
| holdout | No council | +0.065 | [+0.040, +0.090] | 6-0-0 |
| holdout | No retrieval | +0.080 | [+0.060, +0.105] | 6-0-0 |
| holdout | No inventory | +0.084 | [+0.053, +0.117] | 6-0-0 |
| holdout | No engineering | +0.175 | [+0.141, +0.209] | 6-0-0 |
| holdout | Structured single agent | +0.322 | [+0.276, +0.368] | 6-0-0 |
| holdout | General one-shot | +0.338 | [+0.283, +0.405] | 6-0-0 |

## Limitations

- Six holdout protocols provide prospective evidence but remain a small sample.
- LLM outputs are stochastic even at temperature zero because provider execution can vary.
- Reference agreement is secondary; the primary v2 score emphasizes engineering integrity, safety, evidence, and auditability.
- Wet-lab yield prediction is not established by this benchmark.
- The peroxide holdout exposed a council-framing error: tert-butyl hydroperoxide chemistry was labeled as gas-liquid O2 chemistry. The deterministic final calculation correctly emitted no gas stream, but the mistaken framing contributed to a screened result.
- Several council audits produced implausibly high pressure-floor estimates. These LLM-derived values are retained as audit evidence and must be replaced or bounded by deterministic vapor-pressure calculations before wet-lab deployment.
- The seven-step holdout publicly specifies only its first operation while requesting an integrated sequence. The result appropriately requires screening, but this case cannot establish complete multistep design accuracy without all seven stage protocols.

Raw JSON, prompts, model events, council snapshots, metrics, and checksums remain in the two run directories.
