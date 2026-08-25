# Post-processing Amendment

This amendment was added after generation and before final aggregation. It does not alter any generated chemistry design, score, rubric criterion, case, architecture, model, or repeat.

## Failed generation outcome

One GPT-5.2 FlowPilot cell (`Photochemical oxidation`, `repeat_01`) exhausted the three frozen generation attempts because every response failed the `FlowProposal` output schema. The original packet builder stopped the whole campaign when a `result.json` was absent. To avoid deleting or selectively rerunning this failure, post-processing now emits a blinded `GENERATION_FAILED` outcome with empty delivered-design fields and an authoritative verification flag that no valid design was delivered. All three judges scored that failure, and all raw attempts remain archived.

## Qwen judge serialization repair

Two Qwen judgments repeatedly reached the 7,500-token output limit while serializing the fixed 14-row JSON object. The incomplete responses and statuses remain archived. Retries used the same packet, rubric, model, temperature, and scoring rules, with a larger local serialization allowance and an instruction to shorten evidence strings after a length-truncated response. No score content was manually repaired or imputed.

## Reporting

The report generator now represents `generation_failed` separately from `blocked` and `executable` outcomes. The failed outcome remains in all primary paired statistics.
