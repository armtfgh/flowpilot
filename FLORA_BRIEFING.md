# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing — current state

**Version:** 4.2 | April 2026
**Status:** Active Development
**Council:** v4 (stage-gated scoring, active) with a pre-selection expert
refinement loop. v3 (advocacy) kept in tree for reference but no longer invoked.

**Recent work (April 2026):**
- ENGINE Council upgraded from one-pass scoring to a **score → audit →
  bounded expert refinement → deterministic recompute → rescore** loop.
  Domain agents can now propose limited candidate edits before Chief
  selection; the revised candidate matrix is rescored and re-audited.
- Candidate tables now expose `BPR`, tubing material, and concentration so
  the second scoring pass sees the actual revised state instead of the
  original center-point assumptions.
- Multi-step pipeline rebuilt around a single-source-of-truth flow-rate
  contract — `proposal.streams` is authoritative, never recomputed. Stream
  role classification (reactor feed vs quench) now drives topology layout.
- Chief renormalisation rescales reactor-feed pumps to `Σ = Q_winner` and
  updates each stream's derivation text so UI numbers agree with arithmetic.
- Single-step builder's legacy quench block rewritten: no duplicate pumps,
  real Q conservation at the quench mixer, actual quench rate (not ×2).
- τ display precision fixed (`:.1f` instead of `:.0f`) — the diagram no
  longer shows 2.7 min as "3 min".
- Council-selected per-stage τ propagated into `stage_parameters`, honoured
  by the multi-step topology builder.

> This briefing reflects the *current* design of the system. Use `git log`
> for prior architectures.

---

## 1. What FLORA is

FLORA is an AI-powered platform that automates the translation of batch
chemistry protocols into validated continuous-flow process designs. It
combines:

1. **Literature-grounded RAG** over a curated corpus of flow chemistry papers
   (ChromaDB, ~1500 records).
2. **Deterministic first-principles physics** — a 9-step design calculator
   (`flora_translate/design_calculator.py`) that is authoritative for Re,
   ΔP, Da, residence time, pressure drop, Beer-Lambert absorption, and BPR
   sizing. No LLM re-derives these.
3. **ENGINE Council v4** — a stage-gated multi-agent scoring and refinement
   system (Designer, Dr. Chemistry, Dr. Kinetics, Dr. Fluidics, Dr. Safety,
   Skeptic, Chief) that selects a winning design from the deterministically
   generated candidate shortlist after a bounded expert revision loop. See
   Section 4.
4. **Handbook-grounded fundamentals** — 2537 rules extracted from 6 flow
   chemistry textbooks, injected into agent prompts by domain.
5. **Multi-provider LLM backends** — Anthropic, OpenAI, or local Ollama
   (`ENGINE_PROVIDER` in `flora_translate/config.py`).

### What FLORA produces

Given a batch protocol (e.g., *"Ir(ppy)₃ photocatalyzed decarboxylative
radical addition in DMF, 0.1 M, 450 nm LED, N₂, 24 h, 72 %"*), FLORA returns:

- A validated **`FlowProposal`** with residence time, flow rate, tubing ID,
  BPR, temperature, wavelength, and fully-specified stream assignments.
- A **process flow diagram** (SVG/PNG) with syringe-pump icons for liquid
  streams and **MFC icons for gas streams** (O₂, H₂, CO₂, N₂, Ar, air).
- A **full deliberation log** — Designer candidate matrix, domain-scored
  evaluations, Skeptic's arithmetic audit, pre-selection candidate
  refinement, DFMEA, and Chief's weighted selection with reasoning.
- A **before/after proposal comparison table** showing what the council
  changed versus the initial calculator proposal.
- A **design envelope** — ±30 % feasible operating window feeding the
  Bayesian optimizer.
- **Literature references** supporting the analogy choices.
- A **confidence score** (HIGH / MEDIUM / LOW) from analogy quality.

---

## 2. System architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                              │
│  Batch-to-Flow  │  Process Design  │  Condition Optimization (BO)   │
└────┬───────────────┬──────────────────────────┬───────────────────┘
     │               │                          │
     ▼               ▼                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                     SHARED BACKEND                                  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  9-Step Design   │  │   ENGINE     │  │  ChromaDB           │  │
│  │  Calculator      │  │  Council v4  │  │  Vector Store       │  │
│  │  (deterministic) │◄►│  (6 agents)  │◄►│  (RAG + analogies)  │  │
│  └──────────────────┘  └──────┬───────┘  └─────────────────────┘  │
│                                │                                    │
│                     ┌──────────┴───────────┐                       │
│                     │   2537 handbook      │                       │
│                     │   rules (injected)   │                       │
│                     └──────────────────────┘                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Lab Inventory  │  LLM Providers (Anthropic/OpenAI/Ollama)  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modules

### 3.1 FLORA-Translate — batch → flow

Given an existing batch protocol, produce a flow design.

Pipeline:
1. **Input Parser** — free text or structured JSON → `BatchRecord` (LLM).
2. **Chemistry Reasoning Agent** — mechanism + stream logic +
   sensitivities → `ChemistryPlan` (LLM + handbook rules).
3. **Plan-aware retrieval** — ChromaDB search for similar flow papers
   (k=20, reranked to k=3 analogies).
4. **9-step Design Calculator** — deterministic; computes τ_center, k, Re,
   ΔP, Da, Beer-Lambert, BPR, τ_analogy_min, τ_class_min. Authoritative.
5. **ENGINE Council v4** — stage-gated scoring deliberation
   (see Section 4). Produces a winning design.
6. **Process topology + diagram builder** — converts the council-validated
   `FlowProposal` into a `ProcessTopology` and renders an SVG/PNG.
7. **Output formatter** — human-readable explanation + Streamlit result tabs.

### 3.2 FLORA-Design — goal → flow (no batch protocol)

Given a free-text chemistry goal (no existing batch data), produce a flow
process from scratch. Uses the same shared backend (Calculator + Council v4)
after a feature-extraction + topology selection phase that picks unit
operations (pumps, mixers, reactors, BPR, degasser, quench, MFCs, …).

### 3.3 FLORA-Fundamentals — handbook knowledge base

Extracts engineering rules from handbook PDFs in a two-pass pipeline
(Haiku relevance scan → Sonnet detailed extraction). Typical skip rate
60–70 %. Rules are stored in `flora_fundamentals/data/rules.json` and
auto-injected into the Chemistry Agent and every Council specialist's
prompt, filtered by domain (kinetics, fluidics, safety, photochemistry,
materials, scale-up).

### 3.4 Condition Optimization — Bayesian optimizer

Self-contained Gaussian-Process surrogate (Matern kernel) + Expected
Improvement acquisition over a 5000-point candidate grid. No LLM
dependency. Consumes FLORA's ±30 % design envelope as bounds.

### 3.5 Protocol Diagnostics — standalone ENGINE

Validates any flow protocol against engineering constraints without
running the full translation pipeline. Reuses the Skeptic's code-layer
safety rules.

### 3.6 Knowledge pipeline — literature mining

Corpus construction: DOI → PDF → chunk → embed → ChromaDB. Two
collections: `flora_records` (single-paper records) and `flora_pairs`
(batch↔flow pairs for direct analogy retrieval).

---

## 4. ENGINE Council v4 — stage-gated scoring

> **Core principle:** LLMs do judgment, tools do arithmetic, hard rules do
> safety. No agent invents numbers; no agent overrides physics.

### 4.1 Why v4 replaced v3

The v3 advocacy model had four specialists each pick a single candidate
and argue for it. This produced lively deliberation but three structural
problems:

1. **Collapse to unanimity** — specialists often gravitated to the same
   obvious pick, wasting deliberation.
2. **No comparability across candidates** — each specialist only evaluated
   their favourite; candidates outside the picks had no domain-specific
   assessment attached.
3. **Hidden trade-offs** — with per-specialist picks, the Chief had to
   reason from fragments rather than a complete scored matrix.

v4 keeps the domain separation but replaces advocacy with **scoring every
surviving candidate on every domain** (0–1), then uses a deterministic
weighted formula as the tiebreak input. The Chief still resolves using
user objectives and natural-language reasoning, but the underlying numbers
are auditable.

### 4.2 Division of responsibility

| Responsibility | Owner | Why |
|---|---|---|
| Numerical computation (τ, d, Q, Re, ΔP, X, productivity) | Deterministic tools | Reproducible, no hallucination surface |
| Hard gates (L≤25 m, V_R≤50 mL, X_min=0.50, Re<2300, BPR floor for gas-liquid) | Designer + Skeptic code layers | Not debatable — they're physics |
| Domain scoring (chemistry/kinetics/fluidics/safety per candidate) | Dr-agents with tools | Where specialists add value |
| Bounded candidate edits before selection | Dr-agents + deterministic merge/recompute | Lets experts refine candidates without letting prompts bypass physics |
| Arithmetic audit (units, V_R=τQ, L formula, recalc chains) | Skeptic LLM + code | Cross-checks every claim from every agent |
| Final selection | Chief (weighted formula + LLM reasoning) | Closes the loop against user objectives |

### 4.3 Pipeline — stage by stage

```
┌───────────────────────────────────────────────────────────────┐
│  Upstream: BatchRecord, ChemistryPlan, analogies, inventory,  │
│  FlowProposal → 9-step Design Calculator                      │
│     (τ_center, τ_lit, k, IF, pump_max, C_reactor, ṅ_limiting) │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 0 — DESIGNER (framing)                              │
 │  LLM chooses sampling strategy (τ range, d set, L fraction│
 │  preferences, log vs linear spacing, number of samples).   │
 └───────────────────────────┬───────────────────────────────┘
                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 1 — CANDIDATE MATRIX (deterministic)                │
 │  sampling.generate_candidates(...) →                       │
 │    • sample_design_space (τ × d × L_fraction triplets)     │
 │    • compute_metrics (Re, ΔP, L, V_R, r_mix, Da_mass, X,   │
 │                       STY, productivity, absorbance)       │
 │    • hard_filter (bench-physics rejects)                   │
 │    • Pareto front on {prod↑, L↓, r_mix↓}                   │
 │    • Diversity-preserving reduction → ~12 candidates       │
 │    • v4 hard gates: L≤25 m, V_R≤50 mL, X≥0.50              │
 └───────────────────────────┬───────────────────────────────┘
                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 2 — SCORING (four domains in parallel)              │
 │  Every surviving candidate scored 0–1 on every domain:     │
 │                                                             │
 │  Dr. Chemistry  — mechanism fit, photon economy,           │
 │                   Beer-Lambert (ε default 20 for Ir),      │
 │                   solvent/reagent compatibility            │
 │  Dr. Kinetics   — τ vs τ_kinetics margin, enhancement      │
 │                   factor plausibility, conversion estimate │
 │  Dr. Fluidics   — Re, ΔP headroom, r_mix, dual-criterion   │
 │                   mixing check, d-change recommendations    │
 │  Dr. Safety     — BPR adequacy, Da_thermal, material       │
 │                   compatibility, hazard flags, DFMEA prep  │
 │                                                             │
 │  Each agent outputs:                                        │
 │    • overall commentary                                     │
 │    • per-candidate {score, reasoning, concerns, verdict}    │
 │    • optional bounded `proposed_changes` owned by that      │
 │      domain only (chemistry: C; kinetics: τ; fluidics: d;  │
 │      safety: BPR/material)                                  │
 │    • tool calls (for audit trail)                           │
 └───────────────────────────┬───────────────────────────────┘
                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 3 — SKEPTIC (arithmetic audit)                      │
 │  Full auditor with tools:                                   │
 │    • unit checks (Beer-Lambert, ε, ℓ, C consistency)       │
 │    • formula checks (V_R = τ·Q, L = 4V/(πd²), Re = ρvd/μ)  │
 │    • recalculation chains — "if d changed to X, Re becomes │
 │      Y, ΔP becomes Z"; flags any inconsistent knock-on    │
 │    • scope violations — flags when an agent made claims    │
 │      outside its domain                                     │
 │    • mixing-direction sanity checks                         │
 │    • disqualify recommendations for candidates with        │
 │      CRITICAL errors                                        │
 │  Output: audit summary, per-agent issues, disqualify list. │
 └───────────────────────────┬───────────────────────────────┘
                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 3.5 — PRE-SELECTION EXPERT REFINEMENT               │
 │  • Merge bounded domain edits on a per-candidate basis.    │
 │  • Ignore proposals from domains whose first-pass output   │
 │    failed the Skeptic audit at HIGH/CRITICAL severity.     │
 │  • Recompute every revised candidate with deterministic    │
 │    physics (`compute_metrics`, hard filters, hard gates).  │
 │  • Rescore the revised candidate matrix across all four    │
 │    domains and re-run the Skeptic audit.                   │
 │  • Chief sees the rescored matrix, not the stale one-pass  │
 │    scores.                                                 │
 └───────────────────────────┬───────────────────────────────┘
                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 4 — CHIEF (weighted selection)                      │
 │  1. Compute combined score per candidate:                  │
 │       combined = 0.25·chemistry + 0.20·kinetics            │
 │                + 0.20·fluidics + 0.20·safety               │
 │                + 0.15·geometry + objective_modifier        │
 │  2. Disqualify candidates per Skeptic + hard gates.        │
 │  3. LLM selects winner with tools (compute_design_envelope)│
 │     using the full weighted score table + chemistry brief  │
 │     + Skeptic summary + user objectives.                   │
 │  4. Tie-break: safety → envelope width → τ vs τ_lit.       │
 │  5. If all survivors within 10%: return TOP-3 envelope.    │
 │  6. If flow not justified: return "batch is better".       │
 │  7. Chief derives per-pump flowrates from ṅ_limiting + eq  │
 │     and feed concentrations; Chief output carries both     │
 │     winner params and `pump_flowrates[]`.                  │
 │  8. DFMEA on the winner (failure modes, SPOFs, validation  │
 │     experiments).                                          │
 └───────────────────────────┬───────────────────────────────┘
                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │  STAGE 5 — APPLY & RE-VALIDATE                             │
 │  • Patch FlowProposal with winner (τ, d, Q, V_R) + Chief   │
 │    overrides (BPR, material, deox method, BPR_bar).        │
 │  • Apply Chief's pump_flowrates to proposal.streams.       │
 │  • Renormalise REACTOR-FEED streams (excluding quench)     │
 │    so Σ = Q_winner; update each stream's derivation text.  │
 │  • Propagate winner τ into stage_parameters[stage 1] so    │
 │    the multi-step topology builder honours it.             │
 │  • Re-run 9-step calculator on the patched proposal.       │
 │  • Build DeliberationLog (scoring matrix, DFMEA, audit).   │
 └───────────────────────────────────────────────────────────┘
```

### 4.4 Objective elicitation

Chief accepts a user objective string and applies a modifier to domain
weights:

| Objective | Weight shift |
|---|---|
| `de-risk first-run` | safety +0.10, chemistry +0.05, throughput −0.15 |
| `yield-oriented` | kinetics +0.05, chemistry +0.05, throughput −0.10 |
| `throughput-oriented` | fluidics +0.05, geometry +0.05, safety −0.10 |
| `balanced` (default) | no shift |

### 4.5 What the council *never* does

- Invent numeric values for τ, d, Q, Re, ΔP, productivity, Beer-Lambert A.
  These come from `compute_metrics` and deterministic tools.
- Skip deterministic recomputation. Candidate refinement re-invokes
  `compute_metrics` / hard filters / hard gates, and the final applied
  winner still re-runs `DesignCalculator`.
- Override hard safety rules. The Skeptic's code layer is final.
- Score with arbitrary weighted sums that can't be defended. The weights
  are fixed (25/20/20/20/15) and the objective modifier is documented.

---

## 5. Deterministic tools

### 5.1 Physics tools — `engine/tools.py` + `engine/sampling.py`

Every number FLORA shows comes from one of these functions.

| Tool | Purpose |
|---|---|
| `beer_lambert(C, ε, ℓ)` | A = ε·C·(d/10); inner-filter risk classification |
| `calculate_reynolds(Q, d, solvent, T)` | Re + flow regime |
| `calculate_pressure_drop(Q, d, L, solvent)` | Hagen-Poiseuille ΔP |
| `calculate_mixing_ratio(d, τ)` | t_mix / τ |
| `calculate_bpr_required(T, solvent, ΔP, is_gl)` | BPR sizing + 7 bar floor for gas-liquid |
| `check_material_compatibility(mat, solv, T)` | FEP/PFA/PTFE/SS/PEEK rules |
| `check_redox_feasibility(E*, E_ox)` | SET thermodynamic check |
| `estimate_residence_time(k, target_X)` | Independent τ_kinetics |
| `compute_design_envelope(center, ...)` | ±30 % feasible window for BO |
| `sampling.sample_design_space(...)` | Deterministic (τ, d, Q, L_frac) triplets |
| `sampling.compute_metrics(...)` | Full metric set for one candidate |
| `sampling.hard_filter(...)` | Bench-physics reject rules |
| `sampling.generate_candidates(...)` | End-to-end sample → metric → filter → Pareto → diversity-reduce |

### 5.2 Tool-use infrastructure — `engine/tool_definitions.py`

Anthropic-format schemas for all tools, `execute_tool()` dispatcher,
named tool sets per agent, and `to_openai_tools()` converter.
`call_llm_with_tools()` in `llm_agents.py` handles the multi-turn
tool-use loop for both Anthropic and OpenAI providers. Ollama falls
back gracefully to prompt-only mode.

---

## 6. Multi-step & quench workflow — single source of truth

A recurring historical bug: the same flow rate appearing as three
different numbers in three different tabs. The v4.2 pipeline enforces
**one source of truth**: `proposal.streams[label].flow_rate_mL_min`,
written by the Chief and never recomputed downstream.

### 6.1 Stream classification (at topology-build time)

Every entry in `proposal.streams` is classified exactly once:

- **reactor_feed** — its `pump_role` contains none of
  {`quench`, `neutraliz`, `workup`, `post-reactor`} and its contents
  don't match `chemistry_plan.quench_reagent`.
- **quench_stream** — matches any of the above keywords.
- **gas** — its role matches `_GAS_PURE_ROLES` (N₂, O₂, H₂, CO₂, Ar, air,
  MFC, etc.) — always gets an MFC icon, never a syringe pump.

Classification lives in `_is_quench_stream()` and `_stream_is_gas()` in
`flora_translate/main.py`. Both functions are role-based, not
content-string-matching, to avoid false positives ("N₂-degassed reagent"
is not a gas stream).

### 6.2 Q conservation contract

For a single-step process with quench:

```
Q_reactor_inlet = Σ Q_i    for i in reactor_feeds
Q_outlet        = Q_reactor_inlet + Σ Q_j    for j in quench_streams
```

The main reactor T-mixer connects reactor_feed pumps ONLY. Quench streams
are injected at a separate post-reactor Quench T-mixer. Pumps cannot
appear twice in the topology.

For multi-step processes:

```
Q_inlet_stage_i = Q_outlet_stage_{i-1} + Σ Q_new_feeds_stage_i
V_R_stage_i     = τ_i × Q_inlet_stage_i
```

A feed-stream label that is declared in multiple stages is only created
as a pump at the LAST stage that declares it (`_label_last_stage` filter
in `_build_multistep_topology`). This prevents the chemistry LLM's
occasional mis-placement of a quench stream in Stage 1's feed_streams
from producing a duplicate pump node.

### 6.3 Chief renormalisation

The Chief derives `Q_i = ṅ_limiting × eq_i / C_feed_i` for each reactor
feed. That derivation is batch-throughput arithmetic and can differ from
the engineering candidate's Q (which comes from τ × d geometry). To
reconcile:

1. The Chief writes derived Q_i into `proposal.streams[i].flow_rate_mL_min`.
2. `_apply_winner` computes `Σ Q_i over reactor_feeds`.
3. If `Σ ≠ Q_winner` (>2% drift), every reactor_feed Q_i is scaled by
   `Q_winner / Σ`. Quench stream rates are left untouched.
4. Each scaled stream's `reasoning` text is appended with the scaling
   note so the UI derivation formula stays consistent with the displayed
   rate. Example:

   ```
   ṅ_lim=0.333 mmol/min × eq=1.0 / C_feed=0.667 M = 0.500 mL/min
    → scaled ×2.618 to 1.3090 mL/min so Σ reactor-feed pumps
      = Q_reactor_inlet = 2.6180 mL/min
   ```

### 6.4 τ propagation for multi-step

Chief sets `stage_parameters[stage_number=1]["residence_time_min"] =
winner.tau_min` so `_build_multistep_topology` honours it for Stage 1
instead of recomputing from batch time / IF. Later stages still compute
independently (each has its own kinetics). τ display in the diagram uses
`:.1f` precision to avoid silently rounding 2.7 → 3.

---

## 7. Chemistry reasoning — the three-layer architecture

Distinct LLM roles with clear separation:

- **Layer 1 — Chemistry Agent (`chemistry_agent.py`):** mechanism, species,
  stream logic, sensitivities, quench requirement. No hardware decisions.
  Injects handbook rules filtered by mechanism / solvent / photocatalyst.
  - **L-L separator guard:** `post_stage_action = "liquid-liquid extraction"`
    is only emitted when TWO IMMISCIBLE LIQUID PHASES genuinely exist AND
    the batch protocol explicitly mentions aqueous work-up. Gas atmosphere
    changes (deoxygenation, gas injection) never produce a separator node.
- **Layer 2 — Translation Agent (`translation_llm.py`):** FlowProposal
  generation from ChemistryPlan + retrieved analogies + Calculator
  center-point.
- **Layer 3 — ENGINE Council v4:** Designer / four Dr-agents / Skeptic /
  Chief (see Section 4).

This layering keeps each LLM's system prompt short and narrow, which
matters most for local models (gemma4) where context length and
reasoning consistency are limiting.

---

## 8. Retrieval — hard metadata filters

ChromaDB semantic search is pre-filtered by hard metadata (reaction class,
photocatalyst class, solvent family, phase regime) before the 20-top
similarity reranking. Prevents cross-class leakage (e.g., thermal
Suzuki couplings retrieved for a photoredox query).

Two collections:
- `flora_records` — single-paper records (batch OR flow)
- `flora_pairs` — batch↔flow pairs for direct analogy retrieval

---

## 9. Process flow diagram

Rendered via Graphviz in `flora_design/visualizer/flowsheet_builder.py`
with real equipment icons:

- **Syringe pump** icon for liquid reagent streams (`op_type = "pump"`)
- **MFC (Mass Flow Controller)** red-bordered box for gas streams
  (`op_type = "mfc"`). The `op_type` field is set explicitly at topology
  build time in `main.py` using role-based gas detection
  (`_stream_is_gas()`).
- **Coil reactor** / **photoreactor** / **microchannel** / **BPR** /
  **vial** / **mixer (T / cross)** / **quench coil** / **L-L separator**
  icons as appropriate.
- Labels show τ (`:.1f` precision), temperature, wavelength,
  concentrations, flow rates.
- Each unit operation shows a **"Why:"** rationale from the ChemistryPlan
  in the UI; non-standard operations without rationale get a ⚠ warning.

---

## 10. Streamlit UI — result tabs

`pages/translate.py` and `pages/flora_design_unified.py` expose:

- **Summary tab:** key metrics grid (τ, Q, V_R, Re, ΔP, …),
  council winner reasoning, open risks. Reads directly from council
  output — no LLM in the render path.
- **Engineering Design tab:** before/after comparison table
  (pre-council vs post-council proposal) with ★ markers on changed
  fields. Per-reactor breakdown when multi-step or quench exists.
- **Process Diagram tab:** SVG/PNG with per-operation "Why" rationale.
- **Chemistry Plan & Recipe tab:** mechanism, stream logic, sensitivities,
  handbook rules injected for this reaction.
- **Stream Assignments tab:** per-pump details. Shows **two** Q metrics
  when quench exists:
  - `Q_reactor_inlet` — Σ reactor-feed pumps (matches `proposal.flow_rate_mL_min`)
  - `Q_outlet (after quench)` — reactor outlet + quench pumps
  Each pump expander shows the Chief's derivation plus the scaling note
  when renormalisation happened.
- **Council Deliberation tab:**
  - Input Design expander — the FlowProposal seen by the council.
  - Designer candidate shortlist with full metric table.
  - Per-domain score tables (chemistry/kinetics/fluidics/safety) with
    0–1 bars and reasoning.
  - Skeptic audit — per-agent issues, disqualify list, summary.
  - Chief weighted matrix + winner reasoning + DFMEA.
- **Council Report tab:** human-readable export of the full deliberation.
- **Raw JSON tab:** the complete `FlowProposal` + `DeliberationLog` for
  debugging.

---

## 11. LLM provider abstraction

`ENGINE_PROVIDER` in `flora_translate/config.py` switches backends:

- `"anthropic"` → Claude Sonnet / Opus (default)
- `"openai"` → GPT-4o or GPT-4o-mini
- `"ollama"` → local model (currently `gemma4-flora`) at
  `OLLAMA_BASE_URL`

Local models require:
- `think=False` API flag + `/no_think` user prefix to suppress Gemma
  thinking mode (otherwise all tokens spent on internal reasoning,
  content field empty).
- 3× max_tokens budget for complex specialist prompts.
- JSON output reminder appended at the end of the user message.
- Custom Modelfile with `num_ctx 8192` (the API-level `num_ctx` override
  is ignored by Ollama; must be baked into the model).

---

## 12. Technology stack

- **LLM orchestration:** Anthropic Python SDK, OpenAI SDK (for GPT + Ollama)
- **Vector DB:** ChromaDB
- **UI:** Streamlit
- **Diagramming:** Graphviz
- **Numerics:** scikit-learn (GP surrogate), scipy, numpy
- **Chemistry:** RDKit (optional, for SMILES parsing)
- **PDF extraction:** pymupdf, pdfplumber

---

## 13. How to run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API keys (pick one provider)
export ANTHROPIC_API_KEY=...      # or
export OPENAI_API_KEY=...         # or
# Ollama: ensure `ollama serve` is running on OLLAMA_BASE_URL

# 3. Select provider in flora_translate/config.py
ENGINE_PROVIDER = "anthropic"     # or "openai" or "ollama"

# 4. Launch dashboard
streamlit run app.py
```

---

## 14. Limitations & roadmap

**Current limitations:**

- **Point estimates, not distributions.** The calculator treats k, ε, ΔH
  as known. Real values are uncertain (± 2–3× for ε at LED wavelength,
  ± 30–50 % for k estimated from batch). Future: Monte Carlo propagation
  of parameter uncertainty through the metric computation.
- **Open-loop design.** No feedback from lab runs. Future: closed-loop
  calibration where experimental yield/pressure/issues update priors on
  k, ε, mixing constants for the reaction class.
- **Multi-stage kinetics coupling.** Each stage computes τ from its own
  batch time + IF factor; the council's selected τ is propagated only
  to Stage 1. For multi-stage chemistries where stage-2 kinetics matter
  (not just quench/workup), Stage 2+ τ is still computed from batch data.
- **Fixed objective ontology.** Chief accepts four predefined objective
  strings. Future: fine-grained user priority elicitation (Phase 0).

**In progress / near-term:**

- Information-gain-based experimental plan output ("run this τ first
  because it reduces k uncertainty the most").
- Feedback-log-driven calibration of handbook rules and class IF bounds.

---

## 15. Repository map

```
flora_translate/
├── config.py                    ENGINE_PROVIDER + per-component models
├── design_calculator.py         9-step deterministic calculator
├── input_parser.py              Free text → BatchRecord
├── chemistry_agent.py           Layer 1: mechanism / stream logic / quench
├── translation_llm.py           Layer 2: FlowProposal generation
├── revision_agent.py            Targeted patch to existing design
├── schemas.py                   All Pydantic models (incl. DeliberationLog)
├── main.py                      Translate pipeline orchestration +
│                                topology builders (_build_singlestep_topology,
│                                _build_multistep_topology, _is_quench_stream)
└── engine/
    ├── llm_agents.py            call_llm() + call_llm_with_tools()
    ├── tool_definitions.py      Anthropic/OpenAI tool schemas + execute_tool()
    ├── tools.py                 Deterministic domain tools
    ├── sampling.py              Design-space sampling + Pareto
    ├── design_space.py          Design envelope utilities
    ├── council_v3/              LEGACY — advocacy model (not invoked)
    └── council_v4/              ACTIVE — stage-gated scoring model
        ├── __init__.py          Exposes CouncilV4
        ├── designer.py          Stage 0 framing + Stage 1 candidate matrix
        ├── scoring.py           Stage 2: four Dr-agents scoring 0–1
        ├── skeptic.py           Stage 3: arithmetic audit + disqualify
        └── chief.py             Stage 4: weighted selection + DFMEA +
                                 _apply_winner (Q renormalisation +
                                 stage_parameters τ propagation)

flora_design/
└── visualizer/
    └── flowsheet_builder.py     Graphviz flowsheet; MFC via op_type field;
                                 τ displayed with :.1f precision

flora_fundamentals/
├── extractor.py                 Two-pass handbook rule extraction
└── data/rules.json              2537 extracted rules

pages/
├── flora_design_unified.py      Main Streamlit page (8 tabs)
├── translate.py                 Translate result tabs
│                                (_render_streams shows Q_reactor_inlet
│                                 and Q_outlet separately when quench exists)
└── diagnose.py                  Protocol diagnostics standalone
```
