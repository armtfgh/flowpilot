# Claude Judge Schema Adapter Amendment

Date: 2026-08-20 (Asia/Seoul)

The first Claude judge requests failed before inference with HTTP 400. Anthropic
reported that `minItems: 14` is unsupported because its structured-output
implementation accepts only `minItems` values of 0 or 1.

The next pre-inference smoke request reported that Anthropic also does not
accept numeric `minimum` or `maximum` keywords. Those provider-only keywords
were removed as part of the same adapter amendment.

Before any Claude judgment was produced or inspected, the provider adapter was
changed to remove `minItems` from the schema sent to Anthropic. The benchmark's
local validator remains unchanged and still requires exactly 14 rows containing
UO-01 through UO-14 once each and integer scores in the range 0 through 4. The
rubric, prompts, candidate packets, model,
temperature, seeds, score aggregation, and stopping rule were not changed.

All original HTTP 400 attempts remain in the judgment directories. Claude
judging resumes into immutable `attempt_*` directories rather than replacing
those records.
