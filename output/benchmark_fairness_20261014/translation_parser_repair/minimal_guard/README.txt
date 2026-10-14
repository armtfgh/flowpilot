UNAPPLIED MINIMAL GUARD — supersedes the broader unwrapping proposal for possible post-benchmark integration.

reject_nonproposal_payload.patch changes only TranslationLLM._normalize_proposal_data, before any default mutation. It rejects nonobject/empty JSON, a nested final-output proposal envelope (including conflicting flat-plus-nested fields), and objects with no recognized FlowProposal fields. Valid flat normalization remains exactly unchanged. Existing generate retry and schema repair receive the original invalid response and validation error, so no wrapper disposition is silently discarded by an unwrapping adapter.

This is a response-boundary guard, not a new chemistry rubric, mandatory-value policy or proof of scientific completeness. Existing handling of partial flat proposals is intentionally unchanged. The retry explicitly requests the internal flat schema; the rejection message asks that blocking reasons remain in the flat safety/reasoning fields.

Validation: 13 offline tests passed.
- Three current frozen envelopes reject before mutation.
- Six previous practical/manuscript flat answers normalize exactly as before.
- Mocked wrapper then flat response uses existing retry and retains420nm,21C, liquidflow and both streams.
- Three bad responses exercise initial/retry/repair, then raise instead of returning a default design.
- Production translation_llm.py hash remains unchanged.

Command:
/home/amirreza/anaconda3/envs/flent/bin/python -m pytest -q output/benchmark_fairness_20261014/translation_parser_repair/minimal_guard/test_minimal_guard.py

No model calls, scoring, campaign edits or live source changes.
