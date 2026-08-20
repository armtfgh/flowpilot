# Manuscript Five-Case Pilot Dataset

## Dataset

- Five held-out chemistry cases, two generator families, and two architectures: 20 matched outcomes.
- One generation repeat per cell; this is a case-selection pilot, not a confirmatory benchmark.
- Two blinded LLM judges scored 14 fixed equal-weight criteria: 40 valid judgments and 560 criterion rows.
- Every case used one identical frozen input hash across all four conditions.
- Held-out source IDs were excluded from retrieval and shown only to judges for fact checking.

## Fairness Repair

The first pass used a JSON string nested inside a JSON envelope for one-shot outputs. This caused serialization failures unrelated to chemistry. That pass is preserved at `/home/amirreza/Documents/codes/Flow Agent/deliverables/manuscript_five_case_v3_postcompiler_fix_20260819_attempt1_nested_string_contract` and under `archived_attempts/nested_string_contract_v1`. The reported dataset uses the direct JSON-object v2 baseline. FlowPilot generations were not rerun.

## Primary Findings

- Qwen3.6-27B: FlowPilot 0.863 vs one-shot 0.744; delta +0.119.
- GPT-5.4: FlowPilot 0.867 vs one-shot 0.908; delta -0.042.
- Across all ten pairs, the mean delta is +0.039.
- Judge agreement is low. LLM scores are secondary evidence, not the sole endpoint.

## Recommended Cases For Confirmatory Repeats

1. CuAAC: cleanest comparison; both model families show a small non-negative FlowPilot effect and both final contracts are executable.
2. Hydrogenolysis: technically discriminating gas-liquid-solid case; Qwen benefits, GPT-5.4 does not.
3. Exothermic dinitration: safety and heat-transfer case; Qwen benefits while GPT-5.4 one-shot remains stronger.

Photochemical oxidation and two-stage amidation should remain in ESI/failure analysis because both FlowPilot outputs were blocked. Run the selected three with at least three independent repeats per condition. Use deterministic error counts and contract closure as primary endpoints and blinded LLM judging as a secondary sensitivity analysis. Do not claim universal superiority from this single-repeat pilot.
