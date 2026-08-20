# NewGen Holistic Error Audit v2

This package contains no composite quality score and no winner ranking. Each of three independent LLM judges audits all six domains. The primary result is a fixed-criterion, cross-family unanimous LLM error claim. `NOT_ASSESSABLE` and disagreement are reported separately. Machine-calculated checks are retained as independent corroboration.

LLM consensus is not ground truth. Repair candidates remain `NEEDS_TECHNICAL_ADJUDICATION`; rounding-only and intermediate-record concerns are flagged rather than silently counted as established defects.

Use `tables/model_fix_tickets.csv` for FlowPilot repair triage, `tables/judge_findings.csv` for exact evidence, and `deterministic_evidence/numeric_occurrences.csv` to trace numerical values across the complete record.
