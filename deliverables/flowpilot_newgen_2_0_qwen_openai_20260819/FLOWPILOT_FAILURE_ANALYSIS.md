# FlowPilot Failure Analysis

## Purpose

This report uses the frozen NewGen 2.0 Qwen/OpenAI benchmark to identify where
FlowPilot still fails and to define controls that prevent those failures from
being published as executable designs. It is a defect analysis, not a claim
that the benchmark score proves experimental performance.

## Evidence Base

- Six FlowPilot outputs: three chemistries generated with Qwen3.6-27B and the
  same three generated with GPT-5.4.
- Three cases: CuAAC, gas-liquid-solid hydrogenolysis, and two-stage oxidative
  amidation.
- Two independent judge families: Qwen and OpenAI.
- Frozen candidate packets, raw pipeline results, deterministic validation
  records, and per-criterion judge rationales.

All six FlowPilot outputs passed the existing deterministic audit with zero
reported errors. However, the OpenAI judge found 36 criterion deductions,
including 13 scores of 2/4 and one critical score of 1/4. The Qwen judge gave
4/4 to every applicable FlowPilot criterion. The disagreement is important:
Qwen is too lenient to serve as the only release gate, and the deterministic
audit is too narrow to detect the confirmed semantic defects below.

## Confirmed Failure Modes

| ID | Failure mode | Benchmark evidence | Why the present gate missed it | Required prevention |
|---|---|---|---|---|
| FP-01 | Chemistry identity drift | Qwen hydrogenolysis was labeled `Hydrogenation of DMAOL to 3-azetidinol` instead of hydrogenolysis/debenzylation. OpenAI: UO-01 = 1, critical. | Reaction identity remains model-written free text and is not compared with a frozen bond-change/transformation record. | Freeze a typed chemistry identity at intake and reject any final artifact that changes substrate, product, transformation, or modality. |
| FP-02 | Gas and multiphase streams serialized as liquid | Both hydrogenolysis topologies encode the H2 inlet and gas-bearing downstream paths as `liquid`. OpenAI: UO-05 = 2 for both models. | `_connect()` and topology bypass construction hardcode `stream_type="liquid"`. | Derive stream phase from feed phase and unit-operation phase rules; validate every edge before rendering. |
| FP-03 | Control equipment represented as process-path equipment | Hydrogenolysis PID prose places the temperature controller after the reactor even though it is external/unconnected in the topology. | PID prose is assembled from unit-operation insertion order, not graph edges and control edges. | Add typed process edges and control/utility edges; compile PID prose and the diagram only from those edges. |
| FP-04 | Stale council values survive final realization | Qwen hydrogenolysis final liquid flow is 0.15789 mL/min, while its retained validation experiment specifies 0.138 mL/min. | DFMEA and experiments are generated before the deterministic final design and are not recompiled or invalidated afterward. | Generate validation experiments only after realization, or bind every numeric reference to canonical parameter IDs and reject stale hashes. |
| FP-05 | Validation recommendations contradict inventory/topology | The liquid-only two-stage case recommends a 3 bar BPR, gas-flow tests, and BPR blockage tests although no gas/BPR design exists. | Free-text council artifacts are outside inventory and final-topology validation. | Run every validation experiment through inventory, phase, topology, and final-parameter feasibility checks. |
| FP-06 | Safety completeness is a false positive | All six outputs are `executable`; five council safety reports contain zero checks and one contains one check, while the deterministic safety contract reports `complete=true`. | Current completeness means controls exist and a small hardware boolean set passed; it does not require hazard-specific operational controls. | Replace the boolean with a hazard-derived mandatory control matrix and fail closed when any required control lacks evidence. |
| FP-07 | Operating procedures are not executable | Missing combinations of priming, leak testing, startup order, steady-state discard, collection window, shutdown, depressurization, quench/workup, and waste handling. OpenAI repeatedly scored UO-13 = 2. | Procedure text is model-generated and has no topology/hazard-derived required-step contract. | Deterministically compile a required procedure checklist from topology and hazards, then require all steps and parameters before release. |
| FP-08 | Per-component stoichiometry is structurally incomplete | Two-stage stream C names morpholine, TBHP, and solvent but does not quantify TBHP concentration/equivalents. OpenAI: UO-03 = 2. | A stream has one optional concentration plus free-text contents; mixed-feed components have no typed amounts. | Introduce typed `StreamComponent` records with concentration, equivalents, role, and provenance for every reactive component. |
| FP-09 | Unsupported operations appear in instructions | Qwen CuAAC requests 0.22 micrometre filtration despite empty filter inventory; the two-stage case adds optional degassing without a process need. | Inventory validation checks major assigned equipment but does not parse procedure operations or accessories. | Give every executable instruction an operation code and inventory binding; reject unsupported optional and mandatory operations. |
| FP-10 | Residence-time basis is semantically ambiguous | Hydrogenolysis reports `nominal liquid empty-bed/contact time` while also reporting inlet-STP and in-channel apparent times. Empty-bed space time and liquid contact time are treated as one basis. | Numerical formulas close independently, but the basis vocabulary is not typed or mutually exclusive. | Use fixed basis enums and equations: liquid superficial space time, inlet-STP apparent time, in-channel total-volumetric apparent time, and measured RTD where available. |
| FP-11 | Mixed-phase equipment settings omit gas throughput | Hydrogenolysis mixer settings report only liquid throughput even though H2 enters the mixer. OpenAI: UO-11 = 2. | Mixer assignment inherits liquid-flow totals instead of phase-aware edge flows. | Calculate equipment load from incoming graph edges, including both inlet-STP and pressure-corrected gas values where relevant. |
| FP-12 | Unsupported intensification claims leak into final evidence | Outputs contain segmented-flow or fixed residence-time-reduction language that is not supported by the packed-bed/liquid-only topology or measured data. | Generic intensification targets are selected from a fixed map and survive as narrative hypotheses without a final evidence gate. | Default to `preserve`; permit an intensification factor only from cited/measured evidence and prohibit it from directly determining residence time. |

## Fundamental Cause

FlowPilot does not yet have one authoritative executable object. Chemistry,
proposal, deterministic realization, topology, safety report, validation
experiments, and operating procedure are produced or reconciled at different
times. The final numerical proposal becomes authoritative, but the other
artifacts are not fully regenerated and revalidated from it.

The result is locally valid but globally inconsistent output: an equation can
be correct, an inventory assignment can be valid, and a topology image can be
rendered while the complete package still contains the wrong chemistry label,
wrong stream phase, stale flow values, incomplete safety controls, or an
unexecutable procedure.

## Prevention Architecture

### 1. Canonical executable design

Create one typed `ExecutableDesignV2` after council selection. It must contain:

- immutable chemistry identity and provenance;
- typed components for every stream;
- equipment instances and inventory bindings;
- process, control, utility, and waste edges;
- pressure, temperature, phase, and flow state on every edge;
- fixed residence-time basis fields;
- hazard inventory and required controls;
- an ordered operating procedure with parameter bindings;
- validation experiments bound to canonical parameters.

All GUI tabs, JSON exports, diagrams, recipes, safety text, and reports must be
compiled from this object. Intermediate agent prose must never be presented as
the executable design.

### 2. Post-realization compilation order

Use this strict order:

1. Freeze chemistry identity and hard constraints.
2. Solve stream composition, stoichiometry, and flow.
3. Bind inventory and build the phase-aware topology.
4. Recalculate all equipment loads and residence-time bases.
5. Compile hazard controls from the final materials and topology.
6. Compile the operating procedure from the final topology and hazards.
7. Generate final-design validation experiments.
8. Run the final semantic and numerical contract.
9. Publish executable artifacts only when every critical gate passes.

Council DFMEA, procedures, and experiments produced before step 2 may be used
as suggestions, but must be discarded or regenerated before publication.

### 3. Mandatory final gates

Add deterministic gates for:

- `chemistry_identity_preserved`
- `component_stoichiometry_complete`
- `topology_phase_consistent`
- `control_edges_not_in_process_path`
- `equipment_loads_match_edge_flows`
- `residence_time_basis_unambiguous`
- `safety_controls_complete`
- `procedure_complete`
- `validation_experiments_current`
- `validation_experiments_inventory_feasible`
- `no_unsupported_operations`
- `all_published_artifacts_match_canonical_hash`

Any critical failure must change the result to `BLOCKED` or
`INVENTORY_CONFIRMATION_REQUIRED`. It must not publish run parameters or an
executable topology. A diagnostic topology may still show missing requirements.

### 4. LLM role after the change

The LLM may propose chemistry, interpret evidence, identify hazards, and add
optional controls. It must not be the authority for arithmetic closure,
inventory existence, stream phase, required safety controls, or whether the
package is executable. A second LLM can challenge the design, but cannot replace
the deterministic release contract.

## Regression Tests Required

The existing six benchmark outputs should become fixed failure fixtures. The
test suite must prove that:

1. Hydrogenolysis cannot be relabeled as hydrogenation.
2. H2 edges cannot be serialized as liquid.
3. External temperature control cannot appear in the process path.
4. A validation experiment containing 0.138 mL/min is rejected when the final
   design contains 0.15789 mL/min.
5. A liquid-only design rejects BPR and gas-flow experiments absent from the
   inventory and topology.
6. Safety cannot be complete with zero required checks.
7. Missing startup, steady-state, shutdown, or waste steps blocks execution.
8. A reactive component without concentration/equivalents blocks execution.
9. A filtration instruction fails when no filter is declared.
10. Every GUI tab and exported artifact resolves to the same canonical design
    hash and values.

## Implementation Order

### Phase A: Stop known false releases

- Correct phase propagation in `flora_translate/main.py` and
  `flora_translate/topology_polisher.py`.
- Move final DFMEA/validation-experiment generation after deterministic design
  realization in `flora_translate/engine/council_v4/chief.py` or explicitly
  invalidate its pre-realization numeric content.
- Replace the weak safety `complete` calculation in
  `flora_translate/design_realizer.py` with hazard-specific required controls.
- Extend `flora_translate/final_design_validator.py` with the critical semantic
  gates listed above.

Exit condition: every known benchmark defect is either corrected or causes a
blocked result. No known-invalid package may remain `executable`.

### Phase B: Remove the architectural cause

- Add `ExecutableDesignV2` and typed stream components in
  `flora_translate/schemas.py`.
- Compile topology, safety, procedure, validation experiments, and all display
  projections after final realization.
- Remove independent GUI fallbacks to proposal, calculations, chemistry-plan,
  or intermediate topology values when an executable design exists.
- Add a canonical hash to every exported projection.

Exit condition: changing one canonical parameter changes every dependent
artifact, and injecting a conflicting value into any artifact fails validation.

### Phase C: Prove the fix

- Convert all 12 registered defects into deterministic regression fixtures.
- Rerun the same frozen three-case/two-model campaign.
- Add mutation tests that intentionally alter reaction identity, phase, flow,
  equipment, safety steps, and procedure values after realization.
- Rerun the independent judges only after deterministic acceptance, as a
  secondary assessment rather than a release gate.

Exit condition: all release acceptance criteria below pass, including the
mutation tests and cross-artifact hash checks.

## Release Acceptance Criteria

Before calling the revised pipeline robust, require on the frozen cases:

- zero critical semantic-contract failures;
- 100% phase-consistent topology edges;
- zero stale numeric references across final artifacts;
- 100% required safety-control coverage;
- 100% required operating-step coverage;
- 100% quantified reactive stream components;
- zero procedure operations without inventory or an explicit manual capability;
- identical canonical design hashes for GUI, JSON, diagram manifest, and report;
- no regression in existing numerical and inventory closure tests.

LLM-judge scores can remain a secondary external assessment. They must not be
the release criterion, especially because one judge assigned perfect scores to
all FlowPilot criteria despite defects visible in the raw artifacts.

## What Can and Cannot Be Guaranteed

No software can guarantee that a proposed reaction condition will work without
experimental evidence. FlowPilot can, however, guarantee that a design is not
published as executable when its chemistry identity, arithmetic, inventory,
topology, safety controls, procedure, or cross-artifact consistency has not
closed. That is the appropriate meaning of making these errors stop.

## Audited Source Locations

- Frozen criterion judgments: `tables/criterion_judgments.csv`
- Machine-readable defect register: `tables/flowpilot_defect_register.csv`
- Candidate provenance:
  `../../ablation_results/newgen_benchmark/newgen_2_0_qwen_openai_20260818/frozen/candidate_key_confidential.json`
- Qwen hydrogenolysis raw result:
  `../../ablation_results/newgen_benchmark/newgen_production_three_case_20260813_150731/hydrogenolysis_qwen27b_full_flowpilot/result.json`
- GPT hydrogenolysis raw result:
  `../../ablation_results/newgen_benchmark/newgen_openai_repaired_cells_20260813_161126/hydrogenolysis_gpt54_full_flowpilot/result.json`
- Qwen two-stage raw result:
  `../../ablation_results/newgen_benchmark/newgen_production_multistep_repaired_20260813_151952/multistep_qwen27b_full_flowpilot/result.json`
