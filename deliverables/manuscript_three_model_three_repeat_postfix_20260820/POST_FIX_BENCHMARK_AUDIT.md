# Post-Fix Benchmark Audit

## Scope and Integrity

This is the completed, preregistered post-fix campaign from 20 August 2026. It
contains three frozen chemistries, three generator families, two matched
architectures, and three repeats per cell: 54 generated outcomes. Qwen, OpenAI,
and Claude judged every blinded outcome with the same 14-criterion NewGen 2.0
rubric, producing 162 valid judgments. No generation was replaced and no score
was changed after unblinding.

The source paper answers were held out from generation. Model and architecture
identity were withheld from judges. Every applicable criterion was weighted
equally, and the comparison unit was the matched chemistry-model-repeat pair.

## Primary Result

| Generator | One-shot | FlowPilot | Paired delta | 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 0.890 | 0.924 | +0.034 | +0.005 to +0.063 | 7/1/1 |
| GPT-5.4 | 0.923 | 0.896 | -0.028 | -0.076 to +0.020 | 3/2/4 |
| Qwen3.6-27B | 0.765 | 0.904 | +0.139 | +0.071 to +0.207 | 9/0/0 |

The pooled matched effect is +0.048. FlowPilot clearly improved Qwen, modestly
improved Claude, and did not improve GPT in this small three-case campaign. The
GPT confidence interval crosses zero. Excluding the judge from the same family
as the generator gives deltas of +0.069 for Claude, -0.002 for GPT, and +0.188
for Qwen; on that sensitivity analysis GPT is effectively tied.

## Deterministic Outcomes

All 27 FlowPilot runs completed and all 27 contracts were marked executable by
the code frozen at campaign start. Required structural schema validity was 9/9
for every FlowPilot model, compared with 2/9 for Claude one-shot, 6/9 for GPT
one-shot, and 9/9 for Qwen one-shot. One-shot responses remained assessable even
when they did not satisfy the requested machine-readable schema.

## Defect Found by the Benchmark

The GPT deficit is concentrated in CuAAC: the mean FlowPilot-minus-one-shot
effect was -0.088 there, versus -0.002 for hydrogenolysis and +0.006 for
photochemical oxidation. In CuAAC repeats 1 and 3, FlowPilot correctly selected
the Cu/C packed-bed cartridge but also serialized copper on carbon into the
pumped feed recipe. That creates an unsupported slurry, plugging risk, and a
duplicate catalyst assignment. The OpenAI judge penalized this strongly; Claude
penalized it moderately; the Qwen judge did not detect it.

This exposed a deterministic contract gap. Stationary-catalyst detection relied
on narrow wording such as `heterogeneous catalyst`; variants such as
`heterogeneous copper catalyst packed in cartridge` were missed. The defect was
fixed after the campaign at two independent layers:

1. Design realization now classifies immobilized, heterogeneous, supported,
   packed-bed, fixed-bed, and cartridge catalyst roles as stationary when the
   selected reactor is a packed bed, and removes them from pumped streams.
2. The final contract independently reconstructs stationary catalyst roles from
   the chemistry plan and blocks any remaining feed duplication with
   `FINAL-STATIONARY-COMPONENT-IN-FEED`.

The historical benchmark files and scores were not rewritten. Replaying the
three original GPT CuAAC artifacts against the corrected contract now blocks
repeats 1 and 3 and accepts repeat 2, exactly matching the audited defect.

## Judge Sensitivity

Judge agreement was moderate: 34.2% exact, 73.8% within one rubric point, and
mean pairwise absolute disagreement was 0.654 on the 0-4 scale. The Qwen judge
assigned perfect scores to all six GPT CuAAC candidates and therefore had no
discrimination for the duplicated-catalyst failure. LLM consensus scores should
therefore be presented as secondary evidence beside deterministic checks, not as
ground truth.

## Defensible Conclusion

This campaign supports the claim that the FlowPilot architecture substantially
improves the smaller Qwen model and modestly improves Claude under matched
inputs. It does not support a universal claim that FlowPilot always beats a
strong one-shot model: GPT was tied to slightly worse depending on judge-family
sensitivity, because two FlowPilot outputs passed a missing material-role gate.
The benchmark was useful precisely because it exposed that invariant, which is
now enforced and regression-tested.

The study remains limited to three chemistries, three repeats, LLM judges, and
no wet-lab outcome comparison. A confirmatory campaign must be registered and
run as a new campaign against the corrected code; it must not replace this one.
