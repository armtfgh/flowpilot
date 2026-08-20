# Retrospective Validity Audit

## Disposition

**INVALID FOR COMPARATIVE PERFORMANCE CLAIMS.**

The raw prompts and responses remain useful for debugging and qualitative failure analysis. The absolute scores, pairwise vote totals, architecture rankings, and superiority conclusions must not be used in a manuscript or presentation.

## What Passed

- Protocol, objective, hard-constraint, and inventory hashes are identical across all four model-architecture cells within each chemistry case.
- Candidate-to-generator mapping is correct.
- Criterion IDs and 0-4 response validation are enforced.
- Recalculation reproduces every reported weighted score to floating-point precision.
- Raw prompts, responses, statuses, and hashes are retained.

These checks establish data traceability, not benchmark validity.

## Critical Failures

### 1. Architecture was not actually blinded

Assurance packets identify architecture by structure and size even after removing names. Every one-shot assurance view is 0.8-3.0 kB and contains only field reasoning and minimal metadata. Every FlowPilot assurance view is 32.0-42.1 kB and contains calculation, inventory, safety, deliberation, source, and validation structures. A simple packet-size threshold classifies architecture with 100% accuracy.

This violates the claim that judges were blinded to architecture and introduces length, complexity, and familiarity effects.

### 2. Candidate evidence was not equivalent

FlowPilot packets include internal and sometimes stale intermediate records, while one-shot packets contain a concise final proposal. Judges therefore compared a final answer against a final answer plus internal development history. Contradictory rejected candidates could penalize FlowPilot even when they were not part of the final executable design.

All six FlowPilot assurance packets contain truncation markers; several contain 6-39 depth-limit markers. No one-shot packet is truncated or depth-limited. Evidence loss is therefore architecture-dependent.

### 3. The assurance extractor contains a confirmed factual bug

`council_rounds` is stored as an integer in the source results. The packet builder counts rounds only when the value is a list, so every FlowPilot packet reports `round_count: 0` despite two completed rounds. Judges explicitly penalized the resulting contradiction between zero rounds, multiple messages, and a populated decision log.

The final-audit path is also absent for three of six FlowPilot candidates, creating inconsistent assurance evidence within the same architecture.

### 4. Pairwise headline votes count order-sensitive duplicates

The report counts both A/B presentations as separate votes. They are repeated measurements from the same judge and pair, not independent observations.

At the overall-decision level, reversed order changes the answer in:

- GPT-5.4 assurance: 4/9 judge-case decisions.
- GPT-5.4 outcome: 2/9.
- Qwen assurance: 0/9.
- Qwen outcome: 4/9.

For Qwen CuAAC outcome, all three judges reverse their preference when candidate order is reversed. Such cells must be classified as order-unstable/abstain, not counted once for each side.

### 5. Judge agreement is inadequate for pooled ranking

Inter-judge ICC(2,1) is approximately 0.37 for both tracks. Exact inter-judge agreement is only 8-10%. This is insufficient to treat the pooled median as a stable ground-truth score.

Qwen also gives Qwen-generated outcome candidates approximately 31.9 points more than cross-family judges on average. Same-family bias materially affects the pooled result.

### 6. Seed bookkeeping is not stable under sharding

Seeds are calculated from loop positions after candidate/judge/track slicing. Sharding resets these positions. The absolute campaign has only 37 unique recorded seeds for 216 cells, with some seed values reused ten times. The pairwise campaign has 24 unique seeds for 72 cells.

Two of seven child attempts use a different recorded seed from their base call despite the amendment claiming the same seed. This does not necessarily change providers that ignore seed, but it invalidates the stated reproducibility protocol.

## Additional Limitations

- Three chemistry cases are too few for broad architecture claims.
- Three repeats of one judge are technical repeats, not independent judges.
- The rubric was not calibrated against known-good and deliberately corrupted designs.
- The held-out published reference may anchor judges toward one implementation even when other designs are physically valid.
- Outcome and assurance packets do not enforce equal information budgets.
- LLM judges are not a substitute for deterministic conservation, inventory, unit, and source checks.

## What Remains Usable

- Input-parity proof.
- Raw candidate designs and judge comments.
- Qualitative leads about stale values, gas-flow bases, residence-time definitions, stoichiometry, inventory assignments, and safety omissions.
- The campaign as a negative-control example showing why LLM judge benchmarking requires calibration and order controls.

The numerical rankings and winner figures are not usable.

## Requirements for Replacement Benchmark

1. Make deterministic, source-grounded error counts the primary endpoint.
2. Generate one canonical final-design record per candidate with the same schema and field budget.
3. Evaluate final outcomes separately from process assurance; do not mix internal rejected values into the outcome packet.
4. Fix round extraction and add packet parity tests, size limits, and zero-truncation gates.
5. Derive every seed from stable IDs: judge, track, candidate/pair, repeat, and order.
6. Use pairwise preference only when both reversed orders agree; otherwise record abstention.
7. Count one order-robust result per judge-pair, not two order presentations as independent votes.
8. Make cross-family judging the primary analysis and same-family judging sensitivity-only.
9. Calibrate judges using hidden designs with known injected errors before scoring real candidates.
10. Predeclare failure thresholds and refuse ranking when agreement or order consistency is inadequate.

