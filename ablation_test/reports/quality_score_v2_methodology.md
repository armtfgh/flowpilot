# FlowPilot Quality and Assurance Score v2

## Status

This metric is an exploratory reanalysis designed after the current matched outputs were available. It is a candidate specification to freeze before prospective holdout validation; it is not preregistered evidence for the current dataset.

## Principles

- Score observable outputs and run artifacts, never the architecture label.
- Keep direct design quality as the majority of the score.
- Reward evidence provenance and auditable independent review.
- Treat calibrated refusal as safer than confident failure, but not as an executable design.
- Apply hard deployment caps independently of the weighted quality score.
- Exclude the hidden literature reference from the primary score.

## Primary Weights

| Dimension | Weight |
|:--|--:|
| Formal validity | 10% |
| Engineering integrity | 25% |
| Process completeness | 15% |
| Safety adequacy | 15% |
| Evidence provenance | 10% |
| Decision assurance | 20% |
| Actionability & calibration | 5% |

Direct design quality is formal validity, engineering integrity, process completeness, and safety adequacy, totaling 65%.

## Engineering Integrity

| Check | Subweight |
|:--|--:|
| Numeric Completeness | 10% |
| Geometry Consistency | 25% |
| Pump Feasibility | 15% |
| Tubing Feasibility | 15% |
| Reactor Match | 10% |
| Gas Bookkeeping | 25% |

## Evidence And Assurance

- Evidence provenance: 70% verified, contamination-free retrieved records and 30% field-level reasoning coverage.
- Decision assurance: calculation trace 25%, deliberation trace 20%, safety review 15%, final audit 15%, real-inventory provenance 15%, documented confidence 10%.
- Actionability/calibration: 60% executable output and 40% confidence calibrated to detected defects.

## Deployment Gates

| Gate | Maximum readiness score |
|:--|--:|
| Schema Invalid | 0.35 |
| Required Gas Bookkeeping Incomplete | 0.55 |
| Geometry Inconsistent | 0.60 |
| Pump Infeasible | 0.60 |
| Tubing Infeasible | 0.60 |
| Critical Topology Omission | 0.60 |
| Critical Safety Omission | 0.60 |
| Screen Required | 0.65 |

## Statistical Analysis

- Architecture comparison uses the same model and exact six-case intersection.
- Pairwise uncertainty uses 20,000 case-level bootstrap resamples.
- Weight sensitivity uses 20,000 uniformly sampled plausible weight sets, normalized to sum to one.
- Case wins and a two-sided sign test are reported without treating the six cases as a large sample.

## Required Prospective Test

Freeze this specification, evaluate new blinded protocols, complete blinded expert review, and report quality, deployment readiness, cost, and wet-lab outcomes separately. Do not claim overall superiority from the current post-hoc reanalysis alone.
