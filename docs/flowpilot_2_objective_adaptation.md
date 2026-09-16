# FlowPilot 2.0: objective-aware screening

Date: 2026-09-07. Private scientific preview only; legacy policy is unchanged.

## Why this change was needed

The two GUI runs `20260907_164042_webapp` and `20260907_165126_webapp`
received different objectives and hypotheses. Both stored the changed answers
and sent them to all six council participants. Both selected candidate 4,
with 33.5411 and 26.8328 min in two 5 mL coils at 80 C.

The previous generator used the same batch-relative screen regardless of the
objective. The Chief's instruction emphasized the most defensible first
experiment. This encouraged baseline similarity even when the chemist
explicitly prioritized final yield. These were decision-policy limitations,
not evidence of a cached image or lost answer.

## Implemented workflow

1. Freeze `screening_priority` alongside the objective in `DesignInputPackage`.
   Supported values: `auto`, `balanced`, `yield_priority`, `throughput_priority`.
   The GUI provides an explicit selector; the choice survives intake refresh,
   page reload and design submission.
2. Resolve a versioned, deterministic screening intent. An explicit chemist
   selection takes precedence. Auto mode recognizes a limited set of explicit
   optimization phrases, records the matching text, and defaults to balanced
   when wording is ambiguous or unrecognized. It is not a general-purpose
   natural-language intent classifier. For unrestricted wording, use the
   explicit selector. Hypotheses are not promoted to constraints.
3. Generate 12 complete, inventory-bound candidates. Retain common controls;
   change the exploration emphasis with the objective. Yield-focused screening
   covers more experiments with extended holds in both stages. Throughput-
   focused screening covers more shortened holds. Balanced screening retains
   the previous broad batch-relative grid.
4. Annotate each candidate with actual stage hold ratios, target holds and
   temperature deviations. These are exposure descriptors, not yield scores.
   All candidates still pass the existing inventory, stoichiometry, pressure,
   geometry, flow and topology checks before review.
5. Send the same resolved intent and original authority-labeled intake to the
   four domain reviewers, Skeptic and Chief. All four domain reviewers must
   cover all 12 candidates. The Chief must explain objective alignment and
   compare at least two other eligible candidates when available. No selected
   time, mandatory time increase, predicted yield or desired winner is imposed.
6. Preserve the selected physical design through finalization. Store objective
   interpretation, pool fingerprint, raw council requests/responses, candidate
   comparisons and answer bindings in the result JSON. The GUI displays the
   priority and comparisons in Scientific assessment, and binding plus labeled
   council assessments in Responses.

## Scientific limits

- The sampling multipliers are explicit experimental-design heuristics, not
  inferred kinetics or literature-derived acceleration factors. They are
  versioned policy choices, not universally validated optimum sampling grids.
- A longer hold does not guarantee higher yield: degradation, selectivity,
  intermediate stability and phase behavior can reverse the trend.
- Source batch time at 95 C does not establish adequate residence at the
  inventory-limited 80 C. Matching or exceeding the former is not validation.
- Identical or equivalent intentions can legitimately select the same design.
  Changed hypotheses may change uncertainty assessments without changing
  equipment. The application does not manufacture visible differences.
- Auto interpretation is deliberately limited and transparent. General semantic
  paraphrase invariance is not claimed; explicit priorities remove this ambiguity.
- Answer bindings show where information was used. The council's description
  of influence is an unvalidated assessment, not a causal sensitivity study.
- This preview's validated scope remains the two-stage homogeneous thermal
  DPDTC software case. No improved wet-lab yield is claimed.

## Verification artifacts

All new artifacts are under `outputs/flowpilot2/20260907_objective_adaptation/`.
Original GUI runs and earlier development artifacts remain intact.

- `regression_final.xml`: full Python regression, including objective resolution,
  negation, ambiguity, serialization, intake preservation, reproducible pools,
  physical feasibility and invalid Chief-output tests.
- `browser_01.json`: desktop/mobile compatibility check on an existing run and
  explicit-priority persistence/submission, before new live results exist.
- `controlled_ab_02/`: identical saved upstream chemistry, translation and
  retrieval; original A/B intake answers; fresh six-part council for each case;
  downstream finalization and icon diagrams. This isolates answer changes from
  upstream sampling but is not a fresh upstream-model benchmark.
- `fresh_pipeline_03/`: full fresh upstream and downstream evaluation using
  the second intake. This is separate from the controlled comparison.
- `throughput_probe/`: an explicitly synthetic objective-sensitivity test,
  with the same frozen upstream chemistry and a throughput-priority objective.
  This is not a third user-provided protocol or measured laboratory experiment.

The first attempts are retained but superseded: visual inspection exposed
that temperature deviation could refer to the proposed flow temperature
instead of the original batch hold. Source-linked temperatures and a regression
test now prevent that substitution. `ATTEMPTS.md` records the interrupted runs.
The second fresh attempt also exposed a numbered mixed-case solvent alias
parsing defect. Exact abbreviation/expansion matching now recognizes explicit
names without treating a different positional isomer as the same component.

Live outcomes and final GUI checks are recorded in the results report in that
artifact folder. One draw per condition does not establish statistical
reproducibility of model selection. Deterministic tests establish repeatability
of interpretation and candidate construction for a fixed physical input.

## Code map

- `flora_translate/scientific_objective.py`: resolved intent, screen policy,
  exposure descriptors, pool fingerprint and answer bindings.
- `flora_translate/schemas.py`, `intake_agent.py`: frozen priority and context.
- `flora_translate/engine/council_v4/scientific.py`: objective-driven screens,
  council prompts, comparisons and validation.
- `flowpilot_webapp/frontend/src/main.tsx`, `result_views.tsx`: selector,
  objective assessment, alternatives and answer effects.
- `scripts/check_scientific_objective_ab.py`: controlled live comparison.
- `scripts/revalidate_flowpilot2_run.py`: refuse saved council responses when
  the policy or intake changed, even if physical candidates happen to match.

## Running

The GUI is served at `http://localhost:8513`. Choose the private scientific
preview, answer intake, then choose a screening priority or use automatic
interpretation. Backend changes require a server restart; frontend changes
require a rebuild. Saved historical results remain historical and are not
rewritten to look like runs of the new policy.
