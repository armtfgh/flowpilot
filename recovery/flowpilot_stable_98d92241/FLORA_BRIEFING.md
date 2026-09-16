# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing

**Version:** 6.0  
**Updated:** 2026-05-27  
**Status:** Active development and benchmarking  
**Primary active path:** `flora_translate` batch-to-flow pipeline with ENGINE Council v4  

This document describes the current code status and maps the major behavior to
the Python files that implement it. It intentionally supersedes older briefing
notes that described pre-v4 council behavior, liquid-only sizing, or weak-model
rescue paths that are no longer the main source of truth.

---

## 1. Project Purpose

FLORA translates a batch chemistry protocol into a continuous-flow process
proposal. The current system combines:

- LLM parsing and chemistry planning.
- Plan-aware retrieval from the flow literature corpus.
- Deterministic engineering calculations.
- Deterministic design-space candidate generation.
- Multi-agent council evaluation.
- Gas-liquid, heat-transfer, pressure, BPR, stream, and topology checks.
- Streamlit GUI rendering and benchmark tooling.

The current design philosophy is:

```text
LLMs interpret chemistry and judge trade-offs.
Deterministic code computes and synchronizes engineering numbers.
```

The current source of truth is:

```text
final council winner
    -> deterministic calculator rerun
    -> synchronized proposal / streams / topology / GUI
```

---

## 2. Main Execution Path

The full batch-to-flow pipeline is orchestrated by
`flora_translate/main.py::translate`.

```text
User batch protocol
    |
    v
parse_batch_input()
    |
    v
analyze_batch_chemistry()
    |
    v
VectorRetriever + AnalogySelector
    |
    v
DesignCalculator().run()
    |
    v
DesignSpaceSearch().run()
    |
    v
TranslationLLM().generate()
    |
    v
override proposal geometry with top deterministic design-space candidate
    |
    v
CouncilV4().run()
    |
    v
final calculator rerun on winner
    |
    v
_reconcile_final_bpr()
_sync_final_stream_flowrates()
    |
    v
_build_singlestep_topology() or _build_multistep_topology()
    |
    v
FlowsheetBuilder().build()
    |
    v
OutputFormatter + Streamlit GUI
```

The LLM-generated proposal is not allowed to be the final numerical authority.
After the translation LLM returns a `FlowProposal`, the main path overwrites
`residence_time_min`, `flow_rate_mL_min`, `tubing_ID_mm`, and
`reactor_volume_mL` using the top deterministic design-space candidate before
the council starts.

---

## 3. Core Data Models

All main schemas live in `flora_translate/schemas.py`.

| Model | Role |
|---|---|
| `BatchRecord` | Structured batch protocol extracted from user text or JSON. |
| `ChemistryPlan` | Mechanism, stages, stream logic, sensitivities, quench/workup, retrieval hints, and intensification mandate. |
| `IntensificationMandate` | Target tau reduction and declared reason flow should improve batch. |
| `StreamLogic` | Chemistry-level stream plan generated upstream. |
| `ProcessStage` | Per-stage chemistry for multistep protocols. |
| `FlowProposal` | Flow design passed through council and final output. |
| `StreamAssignment` | Pump/MFC stream contents, phase, solvent, concentration, and flowrate. |
| `DesignCandidate` | Final packaged proposal plus council log/report. |
| `ProcessTopology` | Unit-operation graph for GUI and process diagram. |

Important current schema behavior:

- `StreamAssignment.solvent` accepts `None` from LLM JSON and normalizes it to `""`.
- `StreamAssignment.contents` accepts a single string and normalizes it to a list.
- Gas streams can carry `phase="gas"`, `gas_flow_sccm`, and `gas_flow_actual_mL_min`.
- `FlowProposal.multiphase_metrics` and `FlowProposal.heat_transfer_metrics` are deterministic annotations from the calculator.

Responsible file:

- `flora_translate/schemas.py`

---

## 4. Parsing and Upstream Chemistry

### 4.1 Input Parsing

Free-text batch protocols are converted to `BatchRecord`.

Responsible files:

| File | Responsibility |
|---|---|
| `flora_translate/input_parser.py` | Strong-model parser for user protocol to `BatchRecord`. |
| `flora_translate/batch_normalization.py` | Deterministic evidence extraction and correction for scale, time, concentration, and volume. |
| `flora_translate/lightweight_upstream.py` | Lightweight/local parser path and `EvidenceBackedInputParser`. |

The important correction layer is `batch_normalization.py`. It prevents errors
such as missing concentration from text like:

```text
0.5 mmol in 1 mL -> 0.5 M
```

### 4.2 Chemistry Planning

The chemistry planner produces `ChemistryPlan`.

Responsible files:

| File | Responsibility |
|---|---|
| `flora_translate/chemistry_agent.py` | Full chemistry planning for strong models. |
| `flora_translate/lightweight_upstream.py` | Compact/fair lightweight chemistry planning for weak/local models. |
| `flora_translate/intensification.py` | Ensures every plan has an `IntensificationMandate`. |

The chemistry plan identifies:

- Reaction class and mechanism.
- Stage sequence for multistep reactions.
- Gas/liquid involvement.
- O2 sensitivity vs O2 reagent use.
- Light requirement and wavelength.
- Stream logic.
- Quench vs real reaction stage.
- Retrieval hints.
- Intensification mandate.

### 4.3 Upstream Modes

Configured in `flora_translate/config.py`.

Relevant settings:

```python
LIGHTWEIGHT_UPSTREAM_MODE = "auto" | "always" | "v2" | "never"
LIGHTWEIGHT_UPSTREAM_WEAK_MODEL_MARKERS = (...)
```

Current intended behavior:

- Strong models use the full upstream path.
- Weak/local models can use lightweight upstream.
- Lightweight v2 is intended to reduce JSON/schema failures without secretly replacing the design answer with a strong-model anchor.

---

## 5. Literature Retrieval

Retrieval is plan-aware. The chemistry plan improves the query beyond raw batch
text.

Responsible files:

| File | Responsibility |
|---|---|
| `flora_translate/vector_store.py` | Vector storage over paper records. |
| `flora_translate/retriever.py` | Plan-aware retrieval. |
| `flora_translate/analogy_selector.py` | Selects final analogy set from retrieval results. |

The analogies influence:

- Residence-time estimates.
- Intensification factors.
- Reactor/material/BPR precedent.
- Similarity confidence.

If hard filters return no close matches, retrieval relaxes to soft filtering.
This is expected for niche chemistry.

---

## 6. Deterministic Design Calculator

The engineering calculator is implemented in
`flora_translate/design_calculator.py::DesignCalculator`.

It runs the 9-step calculation:

| Step | Responsibility |
|---|---|
| Step 1 | Parse batch quantities and target conversion. |
| Step 2 | Estimate residence time from class/analogy/temperature, or accept council-approved tau override. |
| Steps 3-5 | Reactor volume, tubing length, Reynolds number, pressure drop, pump adequacy. |
| Step 6 | Mixing and mass-transfer metrics. |
| Step 7 | Heat-transfer metrics: wall area, UA, heat generation/removal, `Da_th`, heat score. |
| Step 8 | BPR sizing. |
| Step 9 | Productivity, STY, startup waste, Pe, closure checks. |

Current gas-liquid behavior:

- Detects real reagent gases such as O2, air, H2, CO2, Cl2, O3, SO2, etc.
- Does not treat inert blanket/deoxygenation as gas-liquid reaction feed.
- Calculates MFC setpoint at STP and actual gas volume flow at reactor T/P.
- Calculates gas holdup, liquid holdup, total reactor volume, two-phase pressure multiplier, O2 supply, dissolved O2, kLa, and transfer sufficiency.
- Uses liquid residence time:

```text
tau_liquid = V_liquid / Q_liquid
V_total = V_liquid / (1 - gas_holdup)
```

Current BPR behavior:

- Gas-liquid designs require a BPR floor of 5 bar.
- Routine gas-liquid BPR ceiling is 10 bar.
- Calculator tries to resize tubing when gas-liquid pressure would force excessive BPR.
- Final post-processing no longer silently promotes stale pre-council BPR values.

Important functions:

- `DesignCalculator.run`
- `DesignCalculator.annotate_proposal_with_calculations`
- `DesignCalculator._estimate_gas_context`
- `DesignCalculator._step8`

---

## 7. Design-Space Search and Candidate Matrix

There are two deterministic candidate layers.

### 7.1 Initial Design-Space Search

Responsible file:

- `flora_translate/engine/design_space.py`

Used before translation/council. It finds a deterministic top starting point.
This is where cases such as `tau=93.8 min, Q=0.13397 mL/min, d=1.0 mm` can be
selected before council.

Key functions:

- `DesignSpaceSearch().run`
- `get_council_starting_point`
- `candidates_to_dicts`

### 7.2 Council Designer Candidate Matrix

Responsible files:

| File | Responsibility |
|---|---|
| `flora_translate/engine/council_v4/designer.py` | LLM-guided sampling strategy and v4 hard-gate flagging. |
| `flora_translate/engine/sampling.py` | Deterministic candidate enumeration, metric computation, and hard filtering. |
| `flora_translate/engine/flow_value.py` | Flow-sense/PVS features used by Designer/Skeptic/Chief. |

The Designer does not invent numbers directly. It chooses a strategy, then
deterministic code generates:

```text
tau samples x tubing ID samples x practical length samples
    -> Q, V, L, Re, deltaP, BPR, Pe, STY, X, gas metrics, heat metrics
```

Current criteria:

- Tau range is derived from calculator anchor, analogy, batch time, and intensification mandate.
- Tau is constrained by the intensification policy where appropriate.
- Photochemistry prefers small IDs for photon penetration.
- Gas-liquid photochemistry avoids 0.5 mm as a default because gas holdup can make length/pressure impractical.
- Reactor length is constrained by practical bench limits.
- Flow rate is derived from geometry and tau.
- Gas-liquid candidates are scored on total tube volume, liquid holdup, gas holdup, and two-phase pressure.
- Candidates requiring routine BPR above 10 bar are rejected/flagged.
- Candidates with poor flow value or batch-like behavior can be dropped or marked degraded.

Important functions:

- `generate_candidates`
- `sample_design_space`
- `compute_metrics`
- `hard_filter`
- `run_designer_v4`
- `_apply_v4_hard_gates`

---

## 8. Translation LLM

Responsible files:

| File | Responsibility |
|---|---|
| `flora_translate/prompt_builder.py` | Builds translation prompt from batch, analogies, chemistry plan, and calculator. |
| `flora_translate/translation_llm.py` | Calls the translation model and validates JSON into `FlowProposal`. |

The translation LLM proposes a human-readable flow concept and stream
assignments. Its geometry is then overwritten by the top deterministic
design-space candidate in `flora_translate/main.py`.

Important current detail:

- If the LLM emits inconsistent `residence_time_min`, `flow_rate_mL_min`, and
  `reactor_volume_mL`, `translation_llm.py` can correct the stated tau from
  `V/Q`. This is only pre-council; main then overwrites geometry from deterministic design space.

---

## 9. ENGINE Council v4

Council orchestration lives in:

- `flora_translate/engine/council_v4/chief.py::CouncilV4`

Council modules:

| Module | File | Role |
|---|---|---|
| Problem Framing | `chief.py`, `designer.py` | Classifies reaction flags and flow justification. |
| Designer | `designer.py` | Builds candidate pool from deterministic sampling. |
| Dr. Chemistry | `scoring.py` | Checks chemistry, mechanism, photonics, stream compatibility. |
| Dr. Kinetics | `scoring.py` | Scores residence time, conversion, IF, kinetic risk. |
| Dr. Fluidics | `scoring.py` | Scores Re, pressure, length, mixing, hardware practicality. |
| Dr. Safety | `scoring.py` | Scores BPR, thermal risk, material, hazard handling. |
| Skeptic | `skeptic.py` | Arithmetic, feasibility, scope, and flow-value audit. |
| Revision Board | `scoring.py`, `chief.py` | Applies bounded edits and recomputes candidates. |
| Chief | `chief.py` | Selects winner, applies final bounded changes, runs DFMEA. |

### 9.1 Pre-Council Feasibility Check

Implemented in:

- `flora_translate/engine/council_v4/chief.py::_intensification_feasibility_precheck`

Current behavior:

- It checks whether the kinetic anchor conflicts with the intensification mandate.
- It no longer hard-blocks near-boundary conflicts within 10%.
- It is now candidate-aware: if the current design-space candidate is under the tau ceiling and has projected conversion above the minimum hard gate, the council proceeds.
- In uncertain cases, final design is marked as `SCREEN_REQUIRED` instead of being treated as fully validated.

This fixes the previous issue where a conservative kinetic anchor such as
`187.5 min` could skip council even though the design-space top candidate was
`93.8 min`.

### 9.2 Scoring and Selection

The Chief combines:

- Domain scores.
- Geometry practicality.
- Process Value Score (PVS).
- Objective modifiers.
- Skeptic disqualifications.
- DFMEA risk.

The Chief must justify the final selection with a concrete flow advantage, not
only “highest score.”

### 9.3 Council Source of Truth

Council agents can recommend changes, but after selection:

```text
winner -> DesignCalculator().run(... target tau/Q/ID ...)
```

The deterministic calculator rerun is the final engineering source of truth.

---

## 10. Revision Mechanism

Revision is bounded, not free-form.

Responsible files:

| File | Responsibility |
|---|---|
| `flora_translate/engine/council_v4/scoring.py` | `run_revision_stage`, domain-specific revision suggestions. |
| `flora_translate/engine/council_v4/chief.py` | Applies winner patches and validates/recomputes. |
| `flora_translate/engine/sampling.py` | Recomputes metrics after candidate edits. |

Allowed revision examples:

- Kinetics can adjust `tau_min`.
- Fluidics can adjust `d_mm`.
- Safety can adjust `BPR_bar` or material.
- Chemistry can suggest concentration changes or stream compatibility concerns.

Revision workflow:

```text
agent proposes bounded patch
    -> code applies patch to candidate
    -> metrics recomputed deterministically
    -> hard filters run again
    -> patch accepted only if physically valid
```

If a patch creates an invalid design, it is rejected. Example: a revision that
increases reactor length beyond the bench limit is rejected.

---

## 11. Skeptic Audit

Responsible file:

- `flora_translate/engine/council_v4/skeptic.py`

The Skeptic checks:

- Arithmetic consistency.
- Beer-Lambert consistency.
- Geometry consistency.
- Conversion and hard-gate logic.
- Scope violations.
- Disqualification recommendations.
- Weak-pool/PVS logic.

Important current gas-liquid fix:

For liquid-only candidates:

```text
V_R = tau * Q
```

For gas-liquid candidates:

```text
V_liquid = tau * Q_liquid
V_total = V_liquid / (1 - gas_holdup)
```

The Skeptic now audits gas-liquid candidates against liquid holdup, not total
tube volume.

---

## 12. Final Synchronization Before GUI

Responsible file:

- `flora_translate/main.py`

Important functions:

- `_reconcile_final_bpr`
- `_sync_final_stream_flowrates`
- `_build_singlestep_topology`
- `_build_multistep_topology`

Current final sync rules:

- Final BPR is not silently overwritten by stale pre-council calculator values.
- If calculator BPR exceeds proposal BPR, it is treated as a validation warning unless it is the actual final matched design.
- Liquid stream pump rates are synchronized so their sum equals final `Q_liquid`.
- Gas streams keep MFC setpoint and actual gas volume flow.
- Synchronized streams are mirrored back to the Pydantic `FlowProposal` used by topology generation.
- This prevents Summary, Stream Assignments, and Topology from showing different flow rates.

For gas-liquid:

```text
Summary flow rate = liquid flow rate through reactor
MFC = gas setpoint at STP
Gas actual = gas volume flow at reactor T/P
Topology reactor volume = total physical tube volume
Residence time = liquid holdup / liquid flow rate
```

---

## 13. Topology and Diagram Generation

Topology is built in `flora_translate/main.py`.

Responsible functions:

| Function | Role |
|---|---|
| `_build_translate_topology` | Dispatches single-step vs multistep topology. |
| `_build_singlestep_topology` | Builds pump/MFC, mixer, reactor, BPR, quench, collector for single-stage designs. |
| `_build_multistep_topology` | Builds staged reactor graph with stage-specific feeds, gas injection, quench/workup skipping, and stage volumes. |
| `_enforce_gas_delivery_hardware` | Converts genuine gas feeds to MFC nodes. |

Diagram rendering:

| File | Role |
|---|---|
| `flora_design/visualizer/flowsheet_builder.py` | Builds SVG/PNG process diagram with icons and labels. |
| `flora_translate/topology_polisher.py` | Removes/polishes redundant topology nodes. |

Current topology behavior:

- Gas streams become MFCs.
- Liquid streams become pumps.
- Inert blanket/deoxygenation is not treated as gas reagent injection.
- Inline quench is a mixer/safety operation, not an extra reaction stage.
- Packed-bed reactors are rendered as reactors, not filters.
- Reactor nodes show residence time and reactor volume.
- BPR is retained for gas-liquid designs.

---

## 14. GUI Rendering

Primary GUI page:

- `pages/flora_design_unified.py`

Legacy/secondary translate components:

- `pages/translate.py`

Important GUI components:

| File | Responsibility |
|---|---|
| `pages/flora_design_unified.py` | Main unified Streamlit page, summary, tabs, council display, recipe. |
| `pages/translate.py` | Shared render helpers for chemistry plan and stream assignments. |
| `components/design_steps.py` | Engineering Design tab, 9-step calculation cards, gas-liquid/heat-transfer metrics. |
| `components/design_space_viz.py` | Design-space visualization. |
| `components/council_report.py` | Council report display. |

Key rendered tabs:

- Summary.
- Engineering Design.
- Process Diagram.
- Chemistry Plan and Recipe.
- Stream Assignments.
- Council Deliberation.
- Council Report.
- Raw JSON.

Current GUI consistency rule:

```text
Summary, Engineering Design, Stream Assignments, and Topology must all use the
post-council synchronized proposal/calculator values.
```

---

## 15. Output Formatting

Responsible file:

- `flora_translate/output_formatter.py`

The formatter creates the human-readable explanation from the final
`DesignCandidate` and selected analogies. It should not be treated as the
engineering source of truth; it reports values already selected/recomputed by
the deterministic pipeline.

---

## 16. Configuration and Model Routing

Responsible file:

- `flora_translate/config.py`

Important settings:

| Setting | Meaning |
|---|---|
| `MODEL_INPUT_PARSER` | Parser model. |
| `MODEL_CHEMISTRY_AGENT` | Chemistry planning model. |
| `MODEL_TRANSLATION` | Translation proposal model. |
| `ENGINE_PROVIDER` | Council backend: Anthropic, OpenAI, or local/Ollama. |
| `ENGINE_MODEL_OPENAI` | OpenAI council model, currently used for GPT-4o style runs. |
| `ENGINE_MODEL_ANTHROPIC` | Claude council model. |
| `ENGINE_MODEL_OLLAMA` | Local council model alias. |
| `LIGHTWEIGHT_UPSTREAM_MODE` | Full vs lightweight upstream behavior. |
| `FLOW_TRANSLATION_POLICY` | Current policy is intensification-first. |
| `BATCH_PROXIMITY_THRESHOLD` | Batch-like tau threshold for flow-value checks. |
| `PVS_THRESHOLD` | Weak-pool threshold. |

Lab hardware inventory:

- `flora_translate/data/lab_inventory.json`

The inventory defines pumps, tubing, BPR availability, light sources, and reactor
hardware.

---

## 17. Benchmarking and Validation Tooling

Benchmark scripts live in `benchmark/`.

Important scripts:

| File | Role |
|---|---|
| `benchmark/run_protocol_budget_benchmark.py` | Budget study runner. |
| `benchmark/run_pair_repeat_benchmark.py` | Pair/repeat benchmark runner. |
| `benchmark/run_protocol_openai_council.py` | OpenAI council benchmark path. |
| `benchmark/run_rescued_weak_upstream_benchmark.py` | Weak-upstream rescue benchmark. |
| `benchmark/run_lightweight_v2_smoke.py` | Lightweight upstream smoke tests. |
| `benchmark/run_multiphase_upgrade_validation.py` | Multiphase upgrade validation protocols. |
| `benchmark/make_benchmark_visualizations.py` | Model-study visualization. |
| `benchmark/make_pair_repeat_radar.py` | Pre/post radar metrics for repeated pair tests. |
| `benchmark/export_local_council_logs.py` | Exports raw council logs. |

Current focused regression tests:

- `flora_translate/tests/test_multiphase_design.py`
- `flora_translate/tests/test_multistage_topology.py`

Recent verified test status after multiphase/council fixes:

```text
18 passed
```

---

## 18. Current Important Fixes

Recent fixes now present in the code:

- Gas stream `solvent: null` from LLM JSON no longer crashes `FlowProposal`.
- Gas-liquid candidate generation now accounts for gas holdup and total tube volume.
- Gas-liquid candidates requiring routine BPR above 10 bar are rejected/flagged.
- Final BPR is no longer silently overwritten by stale pre-council pressure calculations.
- Stream Assignment tab and topology are synchronized to final liquid flow rate.
- Skeptic audits gas-liquid volume using liquid holdup, not total tube volume.
- Pre-council intensification feasibility is candidate-aware and does not skip council when a practical intensified candidate already exists.
- Near-boundary kinetic/intensification conflicts become `SCREEN_REQUIRED`, not hard failures.
- Quench/workup is not translated as an extra reaction reactor.
- Genuine gas feeds use MFC hardware; liquid streams containing words like "oxygen-sensitive" or "degassed" do not.

---

## 19. Known Scientific Limitations

These are still limitations, not solved truths:

- Kinetic conversion estimates are usually analogy/class estimates, not fitted reaction kinetics.
- Gas-liquid mass transfer uses simplified holdup/kLa/O2 supply approximations.
- Heat-transfer calculations use approximate U and estimated reaction enthalpy.
- LLM domain-agent reasoning can still be inconsistent; deterministic recomputation is required after any accepted edit.
- A `SCREEN_REQUIRED` final design is a screening hypothesis, not a fully validated operating point.
- Literature retrieval can be weak when no close analogies exist.

---

## 20. File Responsibility Map

| Area | Primary file(s) |
|---|---|
| Main pipeline orchestration | `flora_translate/main.py` |
| Data schemas | `flora_translate/schemas.py` |
| Config/model routing | `flora_translate/config.py` |
| Batch parsing | `flora_translate/input_parser.py`, `flora_translate/lightweight_upstream.py` |
| Deterministic input evidence | `flora_translate/batch_normalization.py` |
| Chemistry planning | `flora_translate/chemistry_agent.py`, `flora_translate/lightweight_upstream.py` |
| Intensification mandate | `flora_translate/intensification.py` |
| Retrieval | `flora_translate/vector_store.py`, `flora_translate/retriever.py`, `flora_translate/analogy_selector.py` |
| Translation prompt/LLM | `flora_translate/prompt_builder.py`, `flora_translate/translation_llm.py` |
| Engineering calculator | `flora_translate/design_calculator.py` |
| Initial design-space search | `flora_translate/engine/design_space.py` |
| Council candidate sampling | `flora_translate/engine/sampling.py` |
| Flow-value/PVS | `flora_translate/engine/flow_value.py` |
| Council orchestrator/Chief | `flora_translate/engine/council_v4/chief.py` |
| Council Designer | `flora_translate/engine/council_v4/designer.py` |
| Domain scoring/revision | `flora_translate/engine/council_v4/scoring.py` |
| Skeptic audit | `flora_translate/engine/council_v4/skeptic.py` |
| Tool schemas/execution | `flora_translate/engine/tool_definitions.py`, `flora_translate/engine/tools.py` |
| Output explanation | `flora_translate/output_formatter.py` |
| Topology polishing | `flora_translate/topology_polisher.py` |
| Diagram rendering | `flora_design/visualizer/flowsheet_builder.py` |
| Main GUI | `pages/flora_design_unified.py` |
| Stream/chemistry render helpers | `pages/translate.py` |
| Engineering design GUI cards | `components/design_steps.py` |
| Benchmark runners/visualizers | `benchmark/*.py` |
| Focused tests | `flora_translate/tests/test_multiphase_design.py`, `flora_translate/tests/test_multistage_topology.py` |

---

## 21. Practical Debugging Guide

When outputs disagree across tabs, check in this order:

1. `result["proposal"]`
2. `result["design_calculations"]`
3. `result["proposal"]["streams"]`
4. `result["process_topology"]`
5. `result["deliberation_log"]`

Responsible sync code:

- `flora_translate/main.py::_reconcile_final_bpr`
- `flora_translate/main.py::_sync_final_stream_flowrates`
- `flora_translate/main.py::_build_translate_topology`

When the council skips:

- Check `CouncilV4._intensification_feasibility_precheck` in
  `flora_translate/engine/council_v4/chief.py`.
- Current expected behavior: near-boundary or candidate-supported conflicts
  proceed as `SCREEN_REQUIRED`; only strong infeasibility blocks.

When gas-liquid numbers look inconsistent:

- Remember that reactor volume is total physical tube volume.
- Residence time is liquid holdup divided by liquid flow:

```text
V_liquid = V_total * (1 - gas_holdup)
tau = V_liquid / Q_liquid
```

When BPR looks too high:

- Check whether the final BPR came from the final matched calculator or stale
  pre-council calculation.
- Current sync prevents stale BPR promotion, but `bpr_reconciliation_note` will
  warn if a mismatch is detected.

