# Methods: NewGen Independent LLM-as-Judge Benchmark

## Experimental unit

The experimental unit is one frozen generated batch-to-flow design. The campaign contains 12 designs: two generator families, two architectures, and three chemistry cases. Generation is complete before judging; judges cannot revise candidates.

## Blinding

Candidate IDs are deterministic hashes. Generator model, provider, architecture, run path, and deliberation-system names are removed from judge packets. The confidential key is retained only for aggregation. Pairwise candidate A/B order is deterministically randomized, then reversed in the second presentation.

## Judges

Three independent model families evaluate every candidate: Qwen3.6-27B, GPT-5.4, and Claude Sonnet 4.6. Temperature is 0.0 and each absolute repeat uses a recorded seed. The models receive the same frozen rubric and case evidence. They do not receive scores from other judges.

## Tracks

Outcome quality evaluates the final proposed experiment: chemistry preservation, numerical consistency, process completeness, inventory feasibility, safety adequacy, and experimental actionability.

Assurance quality evaluates the evidence record: calculation traceability, evidence provenance, decision justification, uncertainty calibration, auditability, and reproducibility. Assurance is deliberately separate because missing logs do not prove that an experimental design is chemically wrong.

## Absolute scoring

Each criterion is scored as an integer from 0 to 4 using frozen anchors. Each candidate is scored three times by each judge on each track. For each candidate and criterion, the primary consensus statistic is the median of all valid judge-repeat ratings. Python applies the preregistered criterion weights and computes:

`track score = 100 * sum(median criterion score * weight) / (4 * sum(weights))`

The LLM never calculates the reported 0-100 aggregate. No manual score changes are permitted.

## Pairwise scoring

For each matched generator-model/case pair, judges compare the single-pass and pipeline designs directly on every criterion and overall. Choices are A, B, or TIE. Each pair is shown twice with A/B reversed. Aggregation maps blinded A/B choices back to architecture and reports vote counts and rates.

## Reliability and sensitivity

Within-judge repeatability is the proportion of candidate-criterion cells with three identical scores, accompanied by score range and standard deviation. Inter-judge agreement is calculated after averaging each judge's three repeats per candidate-criterion target; reported diagnostics are ICC(2,1) absolute agreement, exact agreement, and mean pairwise Spearman correlation. Position sensitivity is the proportion of pairwise decisions unchanged after A/B reversal.

Same-family sensitivity excludes the judge belonging to the candidate generator's model family and recomputes consensus scores. Same-family bias is the own-family judge score minus the mean cross-family judge score.

## Invalid calls

Every request has a status record. A malformed or failed original response is immutable. A deterministic schema-repair retry may be written only to a numbered child attempt folder using the same frozen prompt, schema, and seed; the original remains available and the call-status table reports initial status and attempt count. No score is manually repaired or imputed. The publication report is generated only when all 288 planned cells have a valid resolved response; otherwise aggregation stops with an error.

## Interpretation

Outcome and pairwise outcome are primary. Assurance and reliability diagnostics are secondary. Automated LLM judging measures rubric-aligned evaluation, not wet-lab truth. Three chemistry cases do not establish universal superiority; broader claims require more frozen cases and external validation.
