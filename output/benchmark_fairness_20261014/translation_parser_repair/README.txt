UNAPPLIED PROPOSAL — parser boundary repair; independent of pump/solvent patch.

Observed loss
The fresh bridge harness appends its final response envelope to case.objective (ablation_test/scripts/run_fair_practical_benchmark.py:52,66). This scientific objective is reused inside specialist prompts. All three fresh FlowPilot translator responses therefore use recommended_disposition/disposition_rationale/proposal envelopes. All three preceding practical-repair responses and three repeat_01 original manuscript Claude responses were flat FlowProposal objects. The bridge directive occurs twice in each fresh translator user prompt and zero times in these six comparison prompts.

The production path at flora_translate/translation_llm.py:279 calls FlowProposal(**_normalize_proposal_data(data)). Its normalizer at line44 reads only root scientific fields, writes default values for missing fields and never extracts nested proposal. The envelope therefore becomes flow=0, temperature=25, volume=0, BPR=0, no streams and no wavelength without raising a schema error. Retry/repair is skipped because this default object is schema-valid.

Photo example
Exact raw translator event index2, timestamp2026-10-14T03:15:31.296263+00:00, in generation/photochemical_oxidation/claude_flowpilot/repeat_01/llm_events.jsonl declares wavelength420nm, LED1.5W, temperature21C, liquidflow0.5mL/min and acetonitrile feed. Stage1 designer snapshot already holds null wavelength, empty light_setup,25C and reconstructed liquidflow0.01mL/min. Final delivery preserves the null wavelength and omits the lamp; it was not merely hidden by the judge packet adapter. Chemistry analysis and source inventory still retained420nm.

Evidence
- ../photo_light_loss_trace.json — exact saved raw-to-parser replay.
- ../wrapper_loss_comparison.json — nine-run comparison with paths, event indices, selected fields, stream counts, directive counts and response hashes.
- ../fresh_content_audit.json and .txt — all six original fresh deliveries, with diagnostic-only findings separated from executable deliverables.

Interpretation
The response-format instruction exposed a pre-existing permissive-parser bug. The new experiment is therefore confounded by benchmark prompt plumbing, rather than cleanly isolating architecture under practical information. This does not erase the actual blocked/defective deliveries. Preserve the campaign and label it accordingly. A prospectively registered harness-only follow-up should keep the scientific objective free of response-envelope instructions while preserving the same actual task facts, one-shot answers, rubric and judges. Serialization requirements belong at the outer response boundary, not inside every specialist's scientific task. No judge results were inspected for this audit.

Narrow future parser proposal
proposal_payload.py separates a single explicit envelope from its scientific payload before any numeric normalization. It preserves every raw scientific number and the entire disposition envelope, rejects ambiguous flat-plus-nested fields, rejects malformed/empty payloads rather than fabricating defaults, and never changes scientific quantities. It returns both payload and envelope so callers cannot pretend the envelope decision disappeared. A future production integration must explicitly retain/handle SCREEN/BLOCK disposition and its rationale; BLOCK must not become an implicit execute authorization. This artifact is deliberately not integrated into current generation or downstream safety gates.

Offline tests
/home/amirreza/anaconda3/envs/flent/bin/python -m pytest -q output/benchmark_fairness_20261014/translation_parser_repair/test_proposal_payload.py
13 passed. Covers all nine saved examples, scientific value preservation, source immutability, empty/malformed structures, duplicate definitions and BLOCK metadata retention. Existing numeric normalization remains a separate behavior; the extraction test does not claim that legacy gas/time arithmetic is independently validated.

No production source, archive, frozen prompt, outcome or judge packet was edited. No model/API calls were made.

Adversarial completeness review (2026-10-14, after the 13 boundary tests)
completeness_review.json records an additional limitation. Empty-shaped but nonempty containers (stage_parameters=[{}], streams=[{stream_label:'A'}]) and a payload containing only flow_rate_mL_min can still be accepted by the legacy normalizer/schema and filled with defaults. The 13 tests demonstrate wrapper separation and preservation; they do not establish a complete mandatory scientific-field contract. A future production integration needs a separate explicit completeness check and explicit downstream disposition handling. The helper has not been promoted into production.
