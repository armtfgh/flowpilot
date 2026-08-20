# NewGen Holistic Error Audit v2: Review

## Decision

The former score-based LLM-judge benchmark is withdrawn. This replacement contains no composite score and no winner ranking. It is a diagnostic error audit, not publication-ready evidence of architectural superiority.

All 12 paired candidate outputs were audited in six fixed domains by Qwen3.6-27B, GPT-5.4, and Claude Sonnet 4.6. All 216 candidate-domain-judge calls passed the strict response contract after immutable retries. Generator identity was withheld, all substantive source keys were assigned to an audit domain, and no packet contained a truncation marker.

## What Was Audited

The judges did not inspect only the final answer. Across the six domain calls they received protocol and held-out source facts, inventory, proposal and pre-council records, chemistry plan, numerical calculations, council/deliberation records, topology, equipment allocation, safety, validation, disposition, and final design.

Qwen's 32k context required a documented hierarchical evidence pass for oversized records. Evidence-specific sections were supplied verbatim; duplicated sections were represented by Qwen's own validated findings from the other five full-record domain audits. No raw text was silently truncated.

## Raw Outcome

Across 348 candidate-criterion records:

| Consensus state | Count |
|---|---:|
| Cross-family LLM error claim | 155 |
| Cross-family pass | 90 |
| Judge disagreement | 100 |
| Not assessable | 3 |
| Incomplete | 0 |

The 155 error claims split as follows:

| Generator | Architecture | Error claims |
|---|---|---:|
| Qwen3.6-27B | One-shot | 55 |
| Qwen3.6-27B | FlowPilot | 70 |
| GPT-5.4 | One-shot | 5 |
| GPT-5.4 | FlowPilot | 25 |

These counts do **not** show FlowPilot superiority. Under the holistic LLM audit, FlowPilot exposed more error claims than one-shot generation for both model families.

## Why This Is Diagnostic

Two independent LLM judges agreeing is stronger than one judge, but it is not chemical ground truth. The first technical review found four possible tolerance overcalls, including treating `0.1111` and `0.11111 mL/min` as a material contradiction. All 95 FlowPilot claims therefore remain `NEEDS_TECHNICAL_ADJUDICATION`.

The deterministic evaluator produced a different result. Across its 20 fixed checks, all six FlowPilot outputs had zero failures, while Qwen one-shot had eight failures and GPT one-shot had four. None of the 95 FlowPilot LLM claims was corroborated by that narrower deterministic evaluator. This does not prove the LLM claims false: many concern unsupported assumptions, stale narrative values, operational completeness, evidence provenance, or council traceability that the deterministic evaluator does not test. It does prove that the two evidence layers must not be merged into one score.

The 100 disagreements, 28.7% of all candidate-criterion records, are another reason not to present raw LLM claims as established errors.

## Actionable Findings

The GPT-5.4 FlowPilot outputs generated 25 repair candidates. The recurring issues are:

1. Stale or phantom values survive in `explanation`, council, or calculation records after the final inventory-bound design changes.
2. Hydrogen pressure basis, STP-to-channel flow, gas equivalents, gas holdup, and residence time do not always close to one authoritative basis.
3. Global transport calculations can use a phantom or aggregate reactor geometry instead of stage-specific inventory geometry.
4. Kinetic constants, heat-transfer properties, conversion predictions, and analogies are sometimes presented with inadequate assumption labels.
5. Final validation may report readiness while warnings, uncertainty, or superseded values remain elsewhere in the record.
6. Process topology can mis-type gas streams or retain equipment/settings that conflict with the final allocation.

The Qwen FlowPilot outputs generated 70 repair candidates and show the same classes more frequently, plus incomplete startup/shutdown and quench/waste specifications.

## Use Of Files

- `tables/judge_findings.csv`: every judge finding with exact paths, observed values, expected correction, and source basis.
- `tables/llm_consensus_error_claims.csv`: cross-family unanimous claims before technical adjudication.
- `tables/model_fix_tickets.csv`: 95 FlowPilot repair candidates with triage flags.
- `deterministic_evidence/deterministic_criteria.csv`: machine-calculated PASS/FAIL checks.
- `deterministic_evidence/numeric_occurrences.csv`: every relevant numerical value and JSON path.
- `frozen/rubric.json`: the fixed 29-criterion rubric.

Raw prompts, responses, telemetry, retry histories, and statuses remain in `ablation_results/newgen_benchmark/newgen_holistic_error_audit_v2_20260814`.

## Publication Status

Do not use this package to claim that FlowPilot is better or worse than one-shot generation. It is suitable for model debugging and for designing a smaller, tolerance-explicit, source-adjudicated benchmark. Publication use requires deterministic adjudication of numerical claims and chemistry-expert review of chemistry, safety, and evidence claims.
