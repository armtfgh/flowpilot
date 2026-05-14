# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing

**Version:** 5.1  
**Updated:** April 2026 (last code sync: 2026-04-30)  
**Status:** Active development and benchmarking  
**Primary active path:** `flora_translate` batch-to-flow pipeline with ENGINE Council v4  

This document describes the current project state. It intentionally removes older
descriptions of obsolete council behavior and replaces them with the current
intensification-first pipeline, weak-model upstream modes, benchmark tooling, and
known failure modes observed in the model studies.

---

## 1. Project Purpose

FLORA translates a batch chemistry protocol into a continuous-flow process
proposal. The system combines LLM-based chemistry interpretation with
deterministic engineering calculations, literature retrieval, candidate
generation, multi-agent downstream review, process-diagram generation, and
benchmark analysis.

The project is currently focused on one central question:

**Can an LLM-assisted flow-chemistry agent produce a scientifically meaningful
flow translation that is not merely feasible, but actually improved versus the
batch protocol?**

That question now drives the pipeline design. A proposed flow process must show
process value: shorter residence time, reduced hazardous holdup, improved heat
or mass transfer, improved selectivity, higher productivity, or a defensible
screening plan when the evidence is uncertain.

---

## 2. High-Level Pipeline

Current FLORA-Translate execution is orchestrated in `flora_translate/main.py`.

```text
Batch protocol text or JSON
        |
        v
Input parsing
        |
        v
Chemistry planning
        |
        v
Literature retrieval and analogy selection
        |
        v
9-step deterministic design calculator
        |
        v
Design-space grid search
        |
        v
Translation LLM creates initial FlowProposal
        |
        v
ENGINE Council v4
        |
        v
Final FlowProposal + calculations + council log
        |
        v
Process topology and diagram
        |
        v
Output formatter and Streamlit UI
```

The most important architectural rule is:

**LLMs interpret chemistry and make judgments. Deterministic code computes the
engineering numbers.**

The council may select or reject candidates, but it should not invent
unrecomputed values for residence time, flow rate, pressure drop, Reynolds
number, reactor volume, productivity, BPR requirement, or mass-transfer metrics.

---

## 3. Core Data Models

The main Pydantic models live in `flora_translate/schemas.py`.

| Model | Role |
|---|---|
| `BatchRecord` | Structured representation of the input batch protocol. |
| `ChemistryPlan` | Upstream chemistry interpretation: mechanism, stream logic, sensitivities, quench, retrieval keywords, and intensification mandate. |
| `IntensificationMandate` | Explicit statement of why flow should improve the batch reaction. |
| `FlowProposal` | Proposed flow process: tau, Q, T, concentration, BPR, tubing, reactor volume, streams, notes, status flags. |
| `StreamAssignment` | Pump/stream-level contents, concentration, molar equivalent, flow rate, and reasoning. |
| `DesignCandidate` | Final packaged result: proposal, chemistry plan, council messages, safety report, deliberation log, explanation. |
| `ProcessTopology` | Ordered unit-operation graph used for the process diagram. |
| `DeliberationLog` | Backward-compatible log object used by UI/reporting. |

Important status fields on `FlowProposal`:

| Field | Meaning |
|---|---|
| `engine_validated` | `true` only when the council selected a defensible final design under the current deterministic constraints. |
| `confidence` | `HIGH`, `MEDIUM`, or `LOW`; local or repaired paths often stay `LOW`. |
| `safety_flags` | Includes fallback and screen-required messages such as `SCREEN_REQUIRED`. |

---

## 4. Upstream: Parsing and Chemistry Planning

The upstream stage has two modes.

### 4.1 Full Upstream Path

Used for stronger cloud models unless overridden.

```text
InputParser
        |
        v
ChemistryReasoningAgent
        |
        v
ChemistryPlan
```

Files:

| File | Role |
|---|---|
| `flora_translate/input_parser.py` | Free-text protocol to `BatchRecord`. |
| `flora_translate/chemistry_agent.py` | Full chemistry planning prompt for strong models. |
| `flora_translate/intensification.py` | Code-owned fallback for the intensification mandate. |

The chemistry planning agent identifies:

- Reaction class and mechanism.
- Key intermediates and sensitivities.
- Stream logic and quench logic.
- Retrieval keywords and similar reaction classes.
- `intensification_mandate`.

### 4.2 Lightweight Upstream Path

Implemented in `flora_translate/lightweight_upstream.py`.

This path exists because weak/local models struggled with the full
`ChemistryPlan` schema. It is controlled by:

```python
LIGHTWEIGHT_UPSTREAM_MODE = "auto" | "always" | "v2" | "fair_v2" | "never"
```

Current behavior:

| Mode | Behavior |
|---|---|
| `auto` | Use lightweight path for models matching weak/local markers such as `mini`, `gemma`, `llama`, `mistral`, `qwen`, `local`. |
| `always` | Force lightweight path. |
| `v2` / `fair_v2` | Use evidence-backed lightweight parsing and compact chemistry planning without overwriting the kinetic/design anchor. |
| `never` | Force full upstream path. |

The important distinction from earlier rescue experiments:

**Lightweight v2 is intended to be fair. It does not secretly replace the design
answer with a strong-model kinetic anchor. It uses compact JSON and
deterministic evidence extraction to reduce parsing failures.**

### 4.3 Evidence-Backed Parsing

`EvidenceBackedInputParser` performs deterministic protocol-text checks after
LLM parsing. This was added after local/weak models missed important arithmetic
such as:

```text
0.5 mmol in 1 mL  =>  0.5 M
```

This matters because concentration errors scale all downstream flow-rate,
volume, and productivity calculations.

---

## 5. Intensification Mandate

The intensification mandate is now a first-class object:

```python
IntensificationMandate(
    tau_reduction_target: float,
    minimum_flow_advantage: str,
    required_mixing_regime: str,
    flow_justification_basis: str,
)
```

It is generated by the chemistry agent when possible and guaranteed by
`ensure_intensification_mandate()`.

Default deterministic logic:

| Reaction feature | Typical mandate |
|---|---|
| Photochemistry, hazardous intermediate, or exotherm | About `6x` tau reduction target. |
| Moderate thermal chemistry | About `3x`. |
| Default / weakly classified chemistry | About `2.5x`. |

The mandate is propagated downstream to:

- Designer.
- Dr. Chemistry.
- Dr. Kinetics.
- Dr. Fluidics.
- Dr. Safety.
- Skeptic.
- Chief.

The current benchmark philosophy is intensification-first:

```python
FLOW_TRANSLATION_POLICY = "intensify"
FLOW_MAX_TAU_TO_BATCH_RATIO = 1.0
BATCH_PROXIMITY_THRESHOLD = 0.85
```

So a flow design that simply matches or exceeds batch residence time is not
treated as a successful flow translation.

---

## 6. Retrieval and Analogy Selection

Retrieval is plan-aware.

```text
ChemistryPlan + BatchRecord
        |
        v
VectorRetriever
        |
        v
AnalogySelector
        |
        v
Top flow analogies
```

Files:

| File | Role |
|---|---|
| `flora_translate/vector_store.py` | ChromaDB-backed vector storage. |
| `flora_translate/retriever.py` | Retrieves records using batch and chemistry-plan context. |
| `flora_translate/analogy_selector.py` | Selects final analogy set from retrieved records. |

Collections:

| Collection | Purpose |
|---|---|
| `flora_records` | Single-paper records. |
| `flora_pairs` | Batch-flow paired records when available. |

Current known issue:

Some extracted analogy intensification factors were directionally wrong or
ambiguous. The design calculator now records both raw and corrected IFs and
floors sub-unity IFs to `1.0` so a bad literature record cannot force a
de-intensified flow design.

---

## 7. 9-Step Deterministic Design Calculator

The design calculator is in `flora_translate/design_calculator.py`.

It is the authoritative source for initial engineering calculations.

Main outputs:

| Step | Output |
|---|---|
| 1 | Batch conditions: temperature, concentration, time, conversion. |
| 2 | Kinetics and residence time: batch-derived k, class IF, analogy IF, tau range. |
| 3 | Reactor sizing: volume, ID, length, flow rate, Peclet estimate. |
| 4 | Fluid dynamics: velocity, Reynolds number, regime. |
| 5 | Pressure drop: Hagen-Poiseuille estimate and pump margin. |
| 6 | Mass transfer: mixing time and Damkohler-style mass-transfer flag. |
| 7 | Heat transfer: heat generation/removal, surface-to-volume ratio. |
| 8 | BPR requirement: vapor/gas-liquid logic and minimum BPR. |
| 9 | Process metrics: STY, productivity, startup waste, intensification factor. |

Recent important behavior:

- Raw analogy IFs are retained as `raw_analogy_IFs`.
- Sub-unity analogy IFs are floored to `1.0`.
- `IF_floor_applied` is recorded.
- If analogy IF and class IF strongly disagree, Council v4 marks the kinetic
  anchor as uncertain rather than silently trusting it.

Example from the isoxazole benchmark after the IF-floor fix:

```text
raw_analogy_IFs = [0.089, 0.25, 24.0]
corrected analogy_IFs = [1.0, 1.0, 24.0]
IF_analogy = 1.0
IF_class = 10.0
tau_analogy = 15.0 min
tau_class = 1.5 min
status = kinetic anchor uncertain
```

---

## 8. Design-Space Search and Initial Proposal

After the calculator, FLORA runs a deterministic design-space search before
the council.

Files:

| File | Role |
|---|---|
| `flora_translate/engine/design_space.py` | Grid search over flow design candidates. |
| `flora_translate/engine/sampling.py` | Candidate sampling, metric computation, hard filtering, Pareto tagging. |
| `flora_translate/translation_llm.py` | LLM converts chemistry plan and calculator center into a structured `FlowProposal`. |
| `flora_translate/prompt_builder.py` | Builds the translation prompt. |

The design-space candidate can override the Translation LLM's geometry before
the council:

```python
proposal.residence_time_min = top_candidate.tau_min
proposal.flow_rate_mL_min = top_candidate.Q_mL_min
proposal.tubing_ID_mm = top_candidate.d_mm
proposal.reactor_volume_mL = top_candidate.V_R_mL
```

The Translation LLM is still useful for stream assignment, notes, and process
structure, but deterministic candidates drive the core engineering geometry.

---

## 9. ENGINE Council v4

ENGINE Council v4 is the active downstream multi-agent system.

Files:

| File | Role |
|---|---|
| `flora_translate/engine/council_v4/designer.py` | Problem framing and candidate generation/filtering. |
| `flora_translate/engine/council_v4/scoring.py` | Four domain scoring agents. |
| `flora_translate/engine/council_v4/skeptic.py` | Arithmetic, flow-sense, and weak-pool audit. |
| `flora_translate/engine/council_v4/chief.py` | Orchestrator, selection, refinement, fallback, final patching. |
| `flora_translate/engine/flow_value.py` | Deterministic flow-value and PVS calculations. |

### 9.1 Council Agents

| Agent | Responsibility |
|---|---|
| Designer | Generate candidate pool from deterministic sampling, apply flow-value self-filter, provide pool metadata. |
| Dr. Chemistry | Judge chemistry compatibility, selectivity, stream logic, concentration-related concerns. |
| Dr. Kinetics | Judge tau, conversion, kinetic plausibility, tau-reduction value. |
| Dr. Fluidics | Judge Re, pressure drop, mixing, tube ID, heat-transfer proxy. |
| Dr. Safety | Judge BPR, materials, thermal/hazard risk, operating safety. |
| Skeptic | Audit arithmetic, scope violations, process value, weak-pool conditions, disqualification recommendations. |
| Chief | Select winner using weighted score, PVS, geometry, objective, and audit results. |

### 9.2 Stage Flow

```text
Stage 0: Problem framing and intensification feasibility precheck
        |
        v
Stage 1: Designer candidate pool
        |
        v
Stage 2: Domain scoring by four Dr-agents
        |
        v
Stage 3: Skeptic audit
        |
        v
Stage 3.5: Bounded candidate refinement, recompute, rescore, reaudit
        |
        v
Stage 4: Chief selection
        |
        v
Stage 5: Apply winner to FlowProposal
        |
        v
Stage 6: DFMEA and validation recommendations
```

### 9.3 Stage 0: Feasibility Precheck

The Chief performs an intensification feasibility precheck before normal
candidate generation.

If the current kinetic anchor requires a tau longer than the intensification
ceiling:

```text
tau_ceiling = batch_time / tau_reduction_target
```

the council takes one of two paths:

| Condition | Behavior |
|---|---|
| Kinetic anchor is constraining and credible | Hard fallback to `screen_required`; Designer is skipped. |
| Kinetic anchor is uncertain | Proceed to Designer, but final output must remain `engine_validated=false` and `SCREEN_REQUIRED`. |

Uncertain anchors are detected when, for example:

- Sub-unity analogy IFs were floored.
- Analogy IF is batch-equivalent while class IF predicts strong intensification.
- Raw and corrected IF lists show major disagreement.

### 9.4 Stage 1: Designer Candidate Pool

Designer uses deterministic sampling and computes:

- Tau.
- Tube ID.
- Flow rate.
- Reactor volume.
- Length.
- Reynolds number.
- Pressure drop.
- Mixing ratio.
- `Da_mass`.
- Estimated conversion.
- Productivity.
- Flow-sense report.

The Designer also applies a self-challenge filter:

| Filter | Purpose |
|---|---|
| Batch proximity | Drop candidates too close to batch residence time. |
| Zero flow advantage | Drop candidates with no process-value signal. |
| Batch equivalence | Drop designs that do not need continuous flow. |

Pool metadata includes:

```python
pool_metadata = {
    "candidates_generated": int,
    "candidates_dropped": int,
    "drop_reasons": list[str],
    "pool_quality": "NORMAL" | "DEGRADED" | "INFEASIBLE",
    "regeneration_triggered": bool,
}
```

### 9.5 Stage 2: Domain Scoring

Every surviving candidate is scored by each domain agent.

Each domain returns:

- Numeric domain score.
- Verdict.
- Reasoning.
- Concerns.
- Bounded proposed changes.
- Flow-value contribution.

Domain-owned edits:

| Domain | May edit |
|---|---|
| Chemistry | Concentration. |
| Kinetics | Residence time. |
| Fluidics | Tubing ID. |
| Safety | BPR and tubing material. |

The four flow-value sub-scores are:

| Sub-score | Source |
|---|---|
| `selectivity_flow_value` | Dr. Chemistry. |
| `tau_reduction_flow_value` | Dr. Kinetics. |
| `heat_transfer_flow_value` | Dr. Fluidics. |
| `safety_delta_flow_value` | Dr. Safety. |

`flora_translate/engine/flow_value.py` currently overwrites or fills these
with deterministic code-owned defaults to reduce weak-model overstatement.

**Claude compact scoring mode** (added April 2026):

When `ENGINE_PROVIDER = "anthropic"` and the council model starts with `claude`,
scoring uses a compact prompt bundle (`_claude_compact_prompt_bundle`) that:

- Avoids tool calls entirely.
- Uses `max_tokens = 1400` per domain call.
- Returns JSON only with shorter reasoning strings.
- One-liner prompt per domain with strict field ownership.

This path was introduced to reduce benchmark cost and latency for Anthropic-backend council
runs. It is selected automatically by `_is_anthropic_council_mode()`.

### 9.6 Process Value Score

The Process Value Score, or PVS, is computed as:

```text
PVS =
  0.20 * selectivity_flow_value
+ 0.35 * tau_reduction_flow_value
+ 0.20 * heat_transfer_flow_value
+ 0.25 * safety_delta_flow_value
```

The Chief's final score uses:

```text
final_score =
  0.45 * domain_score_component
+ 0.40 * PVS
+ 0.15 * geometry_score
```

This makes "better than batch" nearly as important as the average domain
adequacy score.

### 9.7 Stage 3: Skeptic

The Skeptic audits:

- Unit consistency.
- `V = tau * Q`.
- Tube length formula.
- Recalculation consistency after proposed edits.
- BPR floor logic.
- Agent scope violations.
- Flow-sense errors.
- Batch-proximate candidates.
- Weak candidate pools.

Skeptic verdicts:

| Verdict | Meaning |
|---|---|
| `PASS` | Continue to Chief. |
| `EDIT` | Continue after bounded corrections. |
| `BLOCK` | Hard stop. |
| `WEAK_POOL` | Reject the whole candidate pool and send regeneration instructions to Designer. |

`MAX_WEAK_POOL_CYCLES = 2`.

### 9.8 Stage 3.5: Bounded Refinement

If domain agents propose bounded edits, the council:

1. Merges allowed edits into revision variants via `_candidate_revision_variants()`.
2. Generates descendant candidates when branching revision mode is active.
3. Recomputes all metrics deterministically via `_materialize_revised_candidate()`.
4. Reapplies hard gates.
5. Rescores the revised pool.
6. Reaudits before Chief selection.

**Revision variant types** (added April 2026):

`_candidate_revision_variants()` generates up to two kinds of variant per candidate:

| Variant type | Mode | Description |
|---|---|---|
| Merged | Always | All domain patches combined into one variant. |
| Domain-focused | `strong_revision_mode` only | One variant per domain, ranked by priority (kinetics → fluidics → safety → chemistry). |

Domain priority is controlled by `_DOMAIN_PRIORITY`.

`strong_revision_mode` is activated when the Skeptic identifies severe issues and
enables branching exploration across domain-specific revisions.

The benchmark mode can limit revision expansion:

```python
benchmark_max_descendants_per_candidate
benchmark_max_total_revised_candidates
```

This prevents runaway refinement loops.

### 9.9 Stage 4: Chief

The Chief receives:

- Candidate table.
- Domain scores.
- PVS rows.
- Skeptic audit.
- Pool metadata.
- Intensification mandate.
- Chemistry brief.
- User objective.

It must produce a concrete selection justification. If it cannot identify a
real flow advantage, it adds:

```text
selection_flag = "REQUIRES_HUMAN_REVIEW"
```

### 9.10 Screen-Required Output

When no fully validated design is defensible, the council does not pretend
success. It returns:

```python
engine_validated = False
confidence = "LOW"
safety_flags += ["SCREEN_REQUIRED: ..."]
```

The screen-required payload can include screen candidates with tau, ID, length,
flow rate, temperature, BPR, and acceptance criteria.

Current caveat:

Some fallback paths still preserve the pre-council proposal as the final
proposal while storing useful screen candidates in the safety report. This is a
known consistency issue to clean up.

---

## 10. Known Council Issue From Gemma-Gemma Deep Dive

The Gemma-Gemma benchmark exposed an important failure mode in local-model
council scoring.

Gemma sometimes returns contradictory score rows such as:

```json
{
  "verdict": "FAIL",
  "kinetics_score": 1.0,
  "reasoning": "Tau reduction target met, but conversion is critically low."
}
```

This is logically inconsistent. The text says failure, but the numeric score
allows the candidate to rank highly.

Observed consequence:

- Candidate tau around `2.0 min` satisfied the intensification target.
- Predicted conversion was only about `12-15%`.
- Mixing was poor.
- Domain agents wrote failure reasoning.
- Numeric scores were still high.
- The Chief selected the candidate, but final status was correctly downgraded
  to `engine_validated=false`, `confidence=LOW`, `SCREEN_REQUIRED`.

Required fix:

- Add deterministic verdict-score consistency checks.
- Treat `FAIL`, `REJECT`, and similar local-model verdicts as disqualifying or
  force score downgrades.
- Let Skeptic flag score/verdict contradictions as high severity.
- Prefer screen-matrix output over a single selected "design" when all
  candidates have low predicted conversion.

Until this is fixed, Gemma council outputs should be interpreted as screening
hypotheses, not validated designs.

---

## 11. Deterministic Tools

Tool schemas and dispatch live in `flora_translate/engine/tool_definitions.py`.
Numerical implementations live in `flora_translate/engine/tools.py`,
`flora_translate/engine/sampling.py`, and `flora_translate/design_calculator.py`.

| Tool / Function | Purpose |
|---|---|
| `calculate_reynolds` | Reynolds number and flow regime. |
| `calculate_pressure_drop` | Hagen-Poiseuille pressure drop. |
| `calculate_mixing_ratio` | Mixing time relative to residence time. |
| `calculate_bpr_required` | BPR requirement and gas-liquid handling. |
| `beer_lambert` | Photochemical absorbance. |
| `check_material_compatibility` | Tubing/material suitability. |
| `estimate_residence_time` | Kinetic tau estimate. |
| `compute_design_envelope` | Feasible operating envelope. |
| `sample_design_space` | Candidate grid generation. |
| `compute_metrics` | Candidate metrics. |
| `hard_filter` | Bench-physics filter. |
| `generate_candidates` | End-to-end candidate sampling and filtering. |

Anthropic and OpenAI council calls can use structured tools. Ollama/local
models use prompt-only compact scoring paths.

---

## 12. Process Topology and Diagram

Topology generation lives in `flora_translate/main.py`.

Diagram rendering lives in:

```text
flora_design/visualizer/flowsheet_builder.py
```

Current process-topology rules:

- Liquid feed streams become pump nodes.
- Gas streams become MFC nodes.
- Reactor feeds go to the main mixer.
- Quench/workup streams are injected after the reactor.
- A stream cannot be both reactor feed and quench.
- For multi-step processes, each stage gets its own reactor zone.
- Stage 1 tau selected by the Chief is propagated into `stage_parameters`.

Flow-rate source of truth:

```text
proposal.streams[*].flow_rate_mL_min
```

The topology builder should not recompute pump rates if the Chief has already
renormalized them.

Q conservation:

```text
Q_reactor_inlet = sum(reactor feed streams)
Q_outlet_after_quench = Q_reactor_inlet + sum(quench streams)
```

---

## 13. Streamlit UI

Main pages:

| Page | Purpose |
|---|---|
| `pages/translate.py` | Batch-to-flow translation UI. |
| `pages/flora_design_unified.py` | Unified design UI. |
| `pages/diagnose.py` | Protocol diagnostics. |
| `pages/optimize.py` | Optimization. |
| `pages/fundamentals.py` | Handbook knowledge. |
| `pages/corpus.py` | Corpus browsing. |
| `pages/prism.py` | Literature extraction/mining. |
| `pages/scout.py` | Search/scouting. |

Typical output tabs include:

- Summary.
- Engineering design.
- Process diagram.
- Chemistry plan and recipe.
- Stream assignments.
- Council deliberation.
- Council report.
- Raw JSON.

---

## 14. LLM Provider Abstraction

Provider/model selection lives in `flora_translate/config.py`.

Current model fields:

```python
MODEL_INPUT_PARSER
MODEL_CHEMISTRY_AGENT
MODEL_TRANSLATION
MODEL_OUTPUT_FORMATTER
MODEL_REVISION_AGENT
MODEL_CONVERSATION_AGENT
MODEL_EMBEDDING_SUMMARY
MODEL_TOPOLOGY_POLISHER
```

Council provider:

```python
ENGINE_PROVIDER = "anthropic" | "openai" | "ollama"
ENGINE_MODEL_ANTHROPIC
ENGINE_MODEL_OPENAI
ENGINE_MODEL_OLLAMA
OLLAMA_BASE_URL
```

**Current default** (as of April 2026): `ENGINE_PROVIDER = "openai"`. The config was
temporarily set to `"openai"` during model matrix benchmarking. Change back to
`"anthropic"` for Claude-council runs.

**New provider-agnostic helpers** (added April 2026):

| Function | Purpose |
|---|---|
| `infer_provider_for_model(model, provider)` | Auto-detects provider from model name prefix (`claude` → anthropic, `gpt-/o1/o3/o4` → openai, else ollama). |
| `call_model_messages(model, system, messages, max_tokens, provider)` | Provider-agnostic text generation for upstream modules. Returns `TextGenerationResult`. |
| `TextGenerationResult` | Dataclass: `text`, `provider`, `model`, `usage`, `stop_reason`, `finish_reason`. |

These allow upstream modules (`input_parser`, `chemistry_agent`, etc.) to call any provider
without hard-coding Anthropic/OpenAI client logic at the call site.

Current benchmark providers:

| Label | Meaning |
|---|---|
| `claude` | Claude Sonnet/Opus configuration depending on component. |
| `gpt4o` | OpenAI GPT-4o. |
| `gpt4omini` | OpenAI GPT-4o-mini. |
| `gemma` | Local/vLLM/Ollama Gemma 31B-style endpoint, depending on benchmark script. |
| `gpt4omini_rescued` | Weak upstream with rescue/lightweight procedure. |
| `gemma_rescued` | Gemma upstream with rescue/lightweight procedure. |

Important distinction:

**Raw weak upstream** and **rescued/lightweight upstream** are not equivalent
benchmark conditions. They must be labeled separately.

---

## 15. Benchmark Infrastructure

Benchmark code lives in `benchmark/`.

| File | Purpose |
|---|---|
| `benchmark/cases.py` | Shared benchmark protocols. |
| `benchmark/pipeline.py` | Common benchmark execution helpers. |
| `benchmark/recorder.py` | Stage events, snapshots, manifests, LLM usage logs. |
| `benchmark/run_candidate_budget_benchmark.py` | Candidate-budget benchmark. |
| `benchmark/run_protocol_budget_benchmark.py` | Protocol-level budget benchmark. |
| `benchmark/run_model_matrix_benchmark.py` | 3×3 upstream × council model matrix benchmark. |
| `benchmark/run_protocol_openai_council.py` | Protocol runs with OpenAI council. |
| `benchmark/run_local_model_benchmark.py` | Local Gemma/Claude benchmark. |
| `benchmark/run_upstream_mode_comparison.py` | Comparison across lightweight/full/rescued upstream modes. |
| `benchmark/run_rescued_weak_upstream_benchmark.py` | Weak-upstream rescue benchmark. |
| `benchmark/run_lightweight_v2_smoke.py` | Lightweight v2 smoke test. |
| `benchmark/run_protocol_gemma_council.py` | Protocol runs with Gemma council. |
| `benchmark/make_benchmark_visualizations.py` | Model benchmark figures and CSVs. |
| `benchmark/make_local_model_visualizations.py` | Local model benchmark figures. |
| `benchmark/export_local_council_logs.py` | Full council logs and reasoning export. |
| `ablation_summary.py` | Exports no-council / 1-candidate / 12-candidate ablation comparison. |

Benchmark outputs are stored under:

```text
benchmark/data/
```

Typical benchmark output structure:

```text
benchmark/data/<benchmark_name>/
    matrix_manifest.csv or budget_manifest.csv
    benchmark_config.json
    U_<upstream>/contexts/<case>/
        prepared_context.json
        snapshots/
        llm_events.jsonl
        stage_events.jsonl
    U_<upstream>/C_<council>/runs/<case>/budget_<n>/repeat_01/
        result.json
        run_summary.json
        metadata.json
        snapshots/
        llm_events.jsonl
        stage_events.jsonl
```

---

## 16. Current Benchmark Findings

### 16.1 Budget Benchmark

The original candidate-budget benchmark used Claude upstream and GPT-4o council.
It generated useful raw data, but trend interpretation was weak because
increasing candidate budget did not create a simple monotonic improvement.

Recent Gemma-Gemma budget benchmark:

```text
benchmark/data/gemma_gemma_budget_benchmark_20260507_205316
```

Budgets tested:

```text
1, 6, 12, 24
```

All budgets converged to the same screening point:

```text
tau = 2.0 min
Q = 4.197 mL/min
ID = 0.75 mm
V = 8.394 mL
engine_validated = false
confidence = LOW
SCREEN_REQUIRED
```

Interpretation:

- Increasing budget increased runtime and tokens.
- It did not improve Gemma-Gemma design quality under current guardrails.
- The result is a screening hypothesis, not a validated design.

### 16.2 Model Matrix Benchmark

The model study compares upstream model choice and council model choice.

Important observation:

- Strong upstreams such as Claude and GPT-4o usually produce meaningful
  candidate pools.
- Raw GPT-4o-mini upstream frequently falls back to screen-required output.
- Raw Gemma upstream is highly sensitive to JSON/kinetic-anchor problems.
- Rescued/lightweight weak upstreams can produce evaluated final results, but
  these must be labeled as rescued conditions.

### 16.3 GPT-4o-mini Upstream

Raw `gpt4omini` upstream did not produce validated final designs in the latest
full matrix. It returned the same fallback-like output across downstream models:

```text
tau = 7.5 min
Q = 2.68083 mL/min
ID = 1.6 mm
engine_validated = false
SCREEN_REQUIRED: intensification infeasible with current kinetic anchor
```

Rescued `gpt4omini_rescued` upstream produced evaluated final results, but it is
a different experimental condition and should not be mixed with raw upstream.

### 16.4 Gemma-Gemma

Gemma-Gemma currently demonstrates the strongest local-model stress test.

Observed issues:

- Local model JSON and schema adherence are fragile.
- Kinetic-anchor uncertainty is common.
- Domain text can identify failure while numeric score remains high.
- Final outputs must be treated as screen-required unless deterministic
  guardrails are strengthened.

---

## 17. Scientific Interpretation Policy

FLORA outputs should be interpreted according to status.

| Status | Interpretation |
|---|---|
| `engine_validated=true` | The system selected a defensible flow design within its deterministic model and current assumptions. |
| `engine_validated=false` + `SCREEN_REQUIRED` | The output is a proposed experimental screen or hypothesis, not a final design. |
| `confidence=LOW` | Literature/kinetic/model support is weak; lab validation is required. |
| `REQUIRES_HUMAN_REVIEW` | Chief could not articulate a concrete flow advantage or major uncertainty remains. |

A successful FLORA design is not just one that runs in a tube. It should answer:

- Is flow actually better than batch?
- What is the concrete flow advantage?
- Is residence time meaningfully shorter?
- Is the flow rate practical?
- Is mixing/dispersion acceptable?
- Is reactor volume/length practical?
- Is safety improved rather than merely unchanged?
- Is the result validated by evidence, or only a screen hypothesis?

---

## 18. Current Limitations

| Limitation | Current status |
|---|---|
| Kinetic uncertainty | Tau estimates rely on batch-derived kinetics and retrieved analogy IFs; uncertain anchors trigger screen-required behavior. |
| Analogy quality | Extracted IFs can be wrong or directionally ambiguous; sub-unity values are now floored but still mark uncertainty. |
| Weak-model upstream | Raw GPT-4o-mini and Gemma upstream are not reliably validated without lightweight/rescue scaffolding. |
| Local-model council scoring | Gemma can return contradictory verdicts and numeric scores. |
| Fallback consistency | Some screen-required paths preserve the pre-council proposal while screen candidates live in the safety report. |
| True experimental validation | No closed-loop lab feedback yet; `engine_validated` means validated by the engine model, not by physical experiment. |

---

## 19. Near-Term Engineering Priorities

1. Add deterministic verdict-score consistency checks.
2. Treat local-model `FAIL` / `REJECT` verdicts as score downgrades or
   disqualification unless explicitly mapped otherwise.
3. Make fallback behavior consistent: promote best screen candidate only as
   `SCREEN_ONLY_NOT_ENGINE_VALIDATED`, or keep final proposal unchanged but make
   screen matrix the primary output.
4. Give downstream council controlled authority to override upstream kinetic
   judgments when upstream evidence is inconsistent.
5. Require override provenance: bad analogy, IF direction ambiguity,
   batch-equivalent tau, contradiction with intensification mandate, or
   unphysical operating point.
6. Keep overridden results low-confidence unless supported by literature or
   experiment.
7. Improve weak/local model JSON robustness with stricter schemas, retries, and
   deterministic post-processing.
8. Add trend/statistical analysis for budget studies beyond raw visualizations.

---

## 20. Repository Map

```text
flora_translate/
    config.py                         Provider/model selection and policy constants
    schemas.py                        Pydantic models
    main.py                           Main translate pipeline and topology builders
    input_parser.py                   Full input parser
    lightweight_upstream.py           Local/weak-model parser and chemistry path
    chemistry_agent.py                Full chemistry planning agent
    intensification.py                Code-owned intensification mandate helpers
    retriever.py                      Plan-aware retrieval
    analogy_selector.py               Top analogy selection
    vector_store.py                   ChromaDB access
    design_calculator.py              9-step deterministic calculator
    translation_llm.py                Initial FlowProposal generation
    output_formatter.py               Human-readable output
    revision_agent.py                 Targeted proposal revision path
    batch_normalization.py            Evidence-backed batch normalization
    engine/
        llm_agents.py                 Provider abstraction, LLM calls, call_model_messages()
        tool_definitions.py           Tool schemas and dispatcher
        tools.py                      Deterministic engineering tools
        sampling.py                   Candidate metrics and filtering
        design_space.py               Deterministic design-space search
        flow_value.py                 Flow-sense and PVS calculations
        council_v4/
            designer.py               Candidate pool generation and filtering
            scoring.py                Domain scoring; Claude compact mode
            skeptic.py                Audit and WEAK_POOL logic
            chief.py                  Orchestration, strong_revision_mode, selection, fallback
        council_v3/                   Legacy, not active in current pipeline

flora_design/
    visualizer/flowsheet_builder.py   Graphviz process diagram renderer

flora_fundamentals/
    data/rules.json                   2537 handbook-derived rules
    knowledge_store.py                Rule access
    handbook_reader.py                Handbook ingestion helpers

benchmark/
    cases.py                              Shared protocols
    pipeline.py                           Benchmark execution helpers
    recorder.py                           Logs, snapshots, manifests
    run_model_matrix_benchmark.py         3×3 upstream × council model matrix
    run_protocol_openai_council.py        Protocol runs with OpenAI council
    run_protocol_gemma_council.py         Protocol runs with Gemma council
    run_upstream_mode_comparison.py       Upstream mode comparison
    run_rescued_weak_upstream_benchmark.py
    run_local_model_benchmark.py
    run_candidate_budget_benchmark.py
    run_protocol_budget_benchmark.py
    run_lightweight_v2_smoke.py
    make_benchmark_visualizations.py
    make_local_model_visualizations.py
    export_local_council_logs.py
ablation_summary.py                       No-council/1-cand/12-cand ablation export

pages/
    translate.py                      Batch-to-flow UI
    flora_design_unified.py           Unified design UI
    diagnose.py                       Diagnostics UI
    optimize.py                       Bayesian optimization UI
    fundamentals.py                   Handbook rules UI
    corpus.py                         Corpus UI
    prism.py                          Literature mining UI
    scout.py                          Search/scouting UI
```

---

## 21. How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Provider selection is controlled in `flora_translate/config.py` and environment
variables:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

For local Gemma-style runs, configure:

```python
ENGINE_PROVIDER = "ollama"
ENGINE_MODEL_OLLAMA = "gemma4-flora"
OLLAMA_BASE_URL = "http://<host>:<port>/v1"
```

For vLLM OpenAI-compatible local endpoints, benchmark scripts can override base
URL and model name directly.

---

## 22. Bottom Line

The current FLORA pipeline is no longer only a feasibility checker. It is being
converted into an intensification-aware flow-design system.

The current strongest parts are:

- Deterministic calculator.
- Explicit intensification mandate.
- Candidate-space enumeration.
- Council v4 stage logs and benchmark artifacts.
- Strong-model model matrix results.
- Visualization and CSV export tooling.

The current weakest parts are:

- Weak/local model upstream reliability.
- Local-model council score/verdict consistency.
- Kinetic-anchor uncertainty.
- Consistent handling of screen-required outputs.

These weaknesses are now explicit in the pipeline instead of hidden: uncertain
designs should become low-confidence screen hypotheses, not falsely validated
flow processes.
