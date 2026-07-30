# Engineering Findings From the Executed Ablation

## Evidence Base

- 20 frozen cases: 12 literature-derived and 8 adversarial.
- 121 preserved run summaries: 108 completed and 13 failed diagnostic cells.
- Primary matched architecture experiment: 42 completed GPT-4o cells covering
  seven variants and six identical protocols.
- The seven variants are general one-shot, structured single agent, no
  retrieval, no engineering, no council, no inventory, and full FlowPilot.
- Full-pipeline portability smoke runs are available for Claude, Qwen, and
  Gemma but are not a publication-level model ranking.
- THQ exclusion, source-record availability, and hidden-source audits passed.

## Findings

1. **The full architecture and its component removals are now directly tested.**

   Every architecture variant completed the same six protocols. The architecture
   map, execution-coverage matrix, full-pipeline schematic, role-classified
   model-call table, and raw prompt logs provide evidence for upstream chemistry,
   deterministic engineering, inventory filtering, and council execution.

2. **Formal output validity separates the pipeline from the one-shot baselines.**

   Full FlowPilot and every component-ablation arm produced schema-valid outputs
   in all six matched cases. General one-shot and structured single agent produced
   zero schema-valid outputs despite often emitting numerically complete text.
   Strict schema compliance must therefore be reported separately from semantic
   or numeric completeness.

3. **The original composite cannot establish architectural superiority.**

   Mean automated composite scores were 0.907 for no council, 0.900 for general
   one-shot, 0.893 for no engineering, 0.872 for structured single agent, 0.865
   for no retrieval, 0.852 for full FlowPilot, and 0.846 for no inventory. The
   composite omits schema validity and council rejection/failure-handling value,
   so this ordering is a screening result, not evidence that removing components
   improves scientific design.

4. **Quality and Assurance Score v2 ranks Full FlowPilot first.**

   The architecture-blind v2 score assigns 65% to direct design quality, 10% to
   evidence provenance, 20% to decision assurance, and 5% to actionability and
   uncertainty calibration. Full scored 0.880, followed by no inventory at 0.840
   and no council at 0.835. Full exceeded no inventory in all six paired cases;
   the mean difference was 0.040 with a case-bootstrap 95% interval of 0.030 to
   0.053. Full ranked first in 80.8% of 20,000 plausible weight sets.

   This metric was designed after the outputs were available. It is exploratory
   and must be frozen before prospective holdout validation.

5. **Quality assurance and deployment readiness disagree.**

   Full had a mean deployment-readiness score of 0.600 and zero completely
   ungated cases. No council reached 0.761 with four of six cases ungated. The
   difference comes from tubing-feasibility, gas-bookkeeping, and explicit
   screen-required gates. The defensible current claim is stronger assured
   decision quality, not superior immediately executable designs.

6. **Gas bookkeeping remains a system-level weakness.**

   None of the seven variants passed complete gas bookkeeping on the matched
   protocols requiring a gas reagent. Pressure-only hydrogen descriptions can
   still produce zero delivered equivalents, and aerobic cases can lose the O2
   stream during upstream normalization. A pressure/solubility/stoichiometry
   policy with a nonzero supply guard remains necessary.

7. **Council execution is real, but expensive and adaptive.**

   The matched full pipeline averaged 40.3 model calls and 154,252 tokens per
   case, compared with one call and roughly 1,300-1,400 tokens for the one-shot
   baselines. Captured full-pipeline prompts include problem framing, design
   strategy, chemistry, kinetics, fluidics, safety, chief selection, revision,
   and post-selection DFMEA roles.

8. **Weak candidate pools expose deterministic-design failure modes.**

   Several council runs invoked retries, redesign, revision, or deterministic
   chief fallback after candidates were rejected. Some multistep pressure-floor
   calculations became physically unreasonable and disqualified the entire pool.
   Candidate generation needs bounded pressure policies and better recovery when
   all candidates violate hard constraints.

9. **Retrieval and inventory did not improve the original automated composite.**

   Retrieval was active in the matched full run. Full FlowPilot scored 0.852
   versus 0.865 without retrieval; full scored 0.852 versus 0.846 without
   inventory. These small, mixed changes do not support a positive or negative
   manuscript claim without expert review and a metric that directly evaluates
   evidence use and hard-constraint enforcement.

10. **Model-portability observations remain preliminary.**

   Claude, Qwen, and Gemma each have one completed full-pipeline smoke case.
   Their different call counts, token accounting, schema behavior, and fallback
   paths make this useful integration evidence, but not a statistically valid
   model comparison.

11. **Expert evaluation is still required.**

   Automated topology and safety scores use controlled keyword screens.
   Blinded expert sheets and a confidential identity key are included. Manuscript
   claims should require completed expert scoring, source-paper verification, and
   a preregistered primary metric.

## Recommended Next Code Work

1. Add authoritative gas-specification models for STP flow, in-channel flow,
   pressure basis, and delivered equivalents.
2. Recompute volume, flow, residence time, pressure, and inventory feasibility
   atomically after every council revision.
3. Bound pressure-floor calculations and add an explicit all-candidates-rejected
   recovery path.
4. Enforce council call/token budgets and report fallback rates.
5. Freeze Quality and Assurance Score v2 before running any new holdout cases.
6. Complete blinded expert review on the matched six-case set, then broaden the
   matched matrix only after the scoring protocol is frozen.
