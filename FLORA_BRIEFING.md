# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing — current state

**Version:** 3.0 | April 2026
**Status:** Active Development
**Last major change:** ENGINE Council v3 (advocacy-based multi-agent council
  with Designer → Expert ⇄ Skeptic ⇄ Expert → Chief Engineer; replaces the v2
  parallel-review + priority-ladder pipeline)

> This briefing reflects the *current* design of the system. Historical change
> logs have been removed; use `git log` if you need the prior architectures.

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
3. **An advocacy-based multi-agent ENGINE Council (v3)** — four LLM agents
   (Designer, Expert panel, Skeptic, Chief Engineer) that select a winning
   design from a shortlist of physically-feasible candidates and justify the
   pick against user-stated objectives.
4. **Handbook-grounded fundamentals** — 2537 rules extracted from 6 flow
   chemistry textbooks, injected into agent prompts by domain.
5. **Multi-provider LLM backends** — Anthropic, OpenAI, or local Ollama
   (`ENGINE_PROVIDER` in `flora_translate/config.py`).

### What FLORA produces

Given a batch protocol (e.g., *"Ir(ppy)₃ photocatalyzed decarboxylative
radical addition in DMF, 0.1 M, 450 nm LED, N₂, 24 h, 72 %"*), FLORA returns:

- A validated **`FlowProposal`** with residence time, flow rate, tubing ID,
  BPR, temperature, wavelength, stream assignments.
- A **process flow diagram** (SVG/PNG) with syringe-pump icons for liquid
  streams and **MFC icons for gas streams** (O₂, H₂, CO₂, N₂, Ar, air).
- A **full deliberation log** — Designer strategy, 4 specialist picks,
  Skeptic attacks + mitigations, Chief's winning choice with reasoning.
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
│  │  Calculator      │  │  Council v3  │  │  Vector Store       │  │
│  │  (deterministic) │◄►│  (4 agents)  │◄►│  (RAG + analogies)  │  │
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
5. **ENGINE Council v3** — advocacy-based multi-agent deliberation
   (see Section 4). Produces a winning design.
6. **Process diagram builder** — Graphviz flowsheet with equipment icons.
7. **Output formatter** — human-readable explanation + Streamlit result tabs.

### 3.2 FLORA-Design — goal → flow (no batch protocol)

Given a free-text chemistry goal (no existing batch data), produce a flow
process from scratch. Uses the same shared backend (Calculator + Council v3)
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
running the full translation pipeline. Reuses the Council's Skeptic
layer for code-enforced safety rules.

### 3.6 Knowledge pipeline — literature mining

Corpus construction: DOI → PDF → chunk → embed → ChromaDB. Two
collections: `flora_records` (single-paper records) and `flora_pairs`
(batch↔flow pairs for direct analogy retrieval).

---

## 4. ENGINE Council v3 — advocacy-based deliberation

> This section is the heart of FLORA and replaces the prior v2 "parallel
> review + priority ladder" pipeline. The v3 architecture is built on one
> principle: **LLMs do judgment, tools do arithmetic, hard rules do safety.**

### 4.1 Why v3 exists

The v2 council ran seven specialists in parallel review and resolved
conflicts via a fixed priority ladder (Safety > Chemistry > Fluidics >
Kinetics). This produced a consistent answer but had three structural
problems:

1. **Scoring hidden as physics** — the calculator chose one "winner" using
   hardcoded weights (productivity=0.5, L=0.15, mixing=0.25, …). The
   council saw only that single point. Weights are opinions masquerading
   as math; any reviewer can ask "why 0.5?" and there is no physics answer.
2. **Specialists forced to agree** — parallel review collapsed into
   unanimous rubber-stamping because every specialist saw the same
   pre-chosen candidate.
3. **LLMs asked to do arithmetic** — verifying Re or ΔP is a boolean
   comparison, not a judgment task; LLM time was wasted on it.

### 4.2 v3 core principle

Three non-negotiable rules govern the v3 pipeline:

| Responsibility | Owner | Why |
|---|---|---|
| Numerical computation (τ, d, Q, Re, ΔP, X, productivity) | Deterministic tools | Reproducible, no hallucination surface |
| Safety / bench rules (BPR ≥ 5 bar gas-liquid, PTFE opaque, L ≤ 20 m, …) | Code | Not debatable — they're physics |
| Design judgment (which candidate fits THIS chemistry) | LLM agents | Where specialists actually add value |

### 4.3 The four agents

**A. Designer** (`council_v3/designer.py`)
- *Role:* chooses a **sampling strategy** (τ range factors, which tubing
  IDs, log vs. linear spacing, number of τ samples), never specific numbers.
- *Expertise:* deep microfluidics — Hagen-Poiseuille, Dean vortex,
  Taylor dispersion, Beer-Lambert, slug-flow for gas-liquid, class-typical
  intensification factors, bench envelope.
- *Tool calls:* `sampling.generate_candidates(...)` — deterministic
  enumeration of (τ, d, Q) triplets, metric computation, hard filter,
  Pareto + diversity-preserving shortlist.
- *Output:* ~12 feasible candidates with full metrics (Re, ΔP, r_mix,
  Da_mass, X, productivity, absorbance), organized as a markdown table.

**B. Expert** (`council_v3/expert.py`)
- *Role:* four specialists — **Dr. Kinetics, Dr. Fluidics, Dr. Photonics,
  Dr. Chemistry** — each reads the shortlist and **advocates for ONE
  candidate** from their domain.
- *Anti-goal:* unanimity. If all four pick the same candidate, the council
  added no value. Disagreement surfaces real trade-offs.
- *Domain logic:*
  - **Kinetics** → best τ-anchor + X compliance (τ ≥ τ_lit/2, X ≥ 0.9,
    IF in class range).
  - **Fluidics** → healthiest ΔP / Re / r_mix / L margins (operator-friendly).
  - **Photonics** → best photon budget (low absorbance, small d,
    transparent material). Abstains for non-photochemical reactions.
  - **Chemistry** → mechanism fit (atmosphere integrity, stream
    compatibility, redox feasibility, byproduct suppression).
- *Output:* 2–4 rival picks with domain-specific reasoning and runner-ups.

**C. Skeptic** (`council_v3/skeptic.py`)
Two layers:
- *Code layer* (`sanity_code_check`) — non-debatable hard rules. Blocks
  a candidate if: ΔP > 95 % pump_max, Re ≥ 2300, L > 20 m, V_R > 25 mL,
  BPR < 5 bar for gas-liquid, opaque tubing in a photoreactor, d > 1.0 mm
  for photochem.
- *LLM layer* — attacks the **assumptions** behind each Expert pick. Not
  arithmetic. Targets things like:
  - k not constant across τ (photocatalyst bleaching, catalyst
    deactivation, chain termination).
  - ε assumed from spectrum peak but LED wavelength is off-peak (3× error
    propagates).
  - Plug-flow X = 1 − exp(−τ/τ_k) ignores laminar RTD broadening.
  - Single-liquid mixing model (Da = k·d²/D) fails in gas-liquid, packed
    bed, suspensions.
  - Atmosphere integrity across stages — back-diffusion of O₂ through
    a single tube is real.
  - Henry's law at operating pressure ≠ at 1 atm.
  - Hardware availability (low-Q candidates need syringe pumps).
- *Output:* list of attacks with severity (HIGH / MEDIUM / LOW) and a
  concrete mitigation for each.

**D. Chief Engineer** (`council_v3/chief.py`)
- *Role:* breaks ties. Weighs surviving picks against **user-stated
  objectives** (de-risk first-run / yield-oriented / throughput /
  balanced) and picks the winner.
- *Tie-break order when picks look equal:*
  1. Fewest HIGH-severity Skeptic attacks
  2. Pareto-front membership
  3. Kinetics advocate's pick (τ-anchor is the hardest physical constraint)
- *Fallback:* if the LLM fails to return a valid pick, a deterministic
  `_deterministic_resolve` runs — majority vote of specialists, tie-broken
  by attack severity.
- *Output:* winning candidate id + reasoning + open risks for the lab +
  `changes_to_apply` dict that patches the `FlowProposal`.

### 4.4 Pipeline — step by step

```
 ┌──────────────────────────────────────────┐
 │  Upstream: BatchRecord, ChemistryPlan,   │
 │  analogies, inventory, FlowProposal      │
 │  → 9-step Design Calculator              │
 │     (τ_center, τ_lit, k, IF, pump_max)   │
 └──────────────────┬───────────────────────┘
                    │
                    ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STEP 1 — DESIGNER                                          │
 │                                                             │
 │  LLM picks sampling strategy:                               │
 │     {tau_low_factor, tau_high_factor, n_tau,                │
 │      tau_log_spaced, d_exclude_above_mm,                    │
 │      include_long_L_fraction}                               │
 │                                                             │
 │  Tool generate_candidates(...) →                            │
 │     • sample_design_space (τ × d × L_fraction triplets)     │
 │     • compute_metrics (Re, ΔP, L, V_R, r_mix, Da_mass,      │
 │                        X, STY, productivity, absorbance)   │
 │     • hard_filter (bench-physics rules only)                │
 │     • Pareto front on {prod↑, L↓, r_mix↓}                   │
 │     • Diversity-preserving reduction to N_target ≈ 12       │
 │                                                             │
 │  Output: candidate shortlist + markdown table               │
 └──────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STEP 2 — EXPERT PANEL (round 1)                            │
 │                                                             │
 │  Four specialists read the SAME shortlist in parallel:      │
 │     Dr. Kinetics  → pick id_K with reason                  │
 │     Dr. Fluidics  → pick id_F with reason                  │
 │     Dr. Photonics → pick id_P with reason (skip if not     │
 │                     photochem)                              │
 │     Dr. Chemistry → pick id_C with reason                  │
 │                                                             │
 │  Each pick includes: domain, pick_id, runner_up_id,         │
 │  pick_reason, concerns_on_other_picks                       │
 └──────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STEP 3 — SKEPTIC (round 1)                                 │
 │                                                             │
 │  A) Code-layer sanity_code_check on each pick:              │
 │     Hard rules → blocked_picks set                          │
 │     Warnings  → passed to LLM for context                   │
 │                                                             │
 │  B) LLM attack on non-blocked picks:                        │
 │     For each surviving pick, produce up to 2 assumption     │
 │     attacks: {pick_id, specialist, assumption_at_risk,      │
 │     mitigation, severity ∈ {HIGH, MEDIUM, LOW}}             │
 │                                                             │
 │  Output: code_checks, llm_attacks, blocked_picks, summary   │
 └──────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
           ┌────────┴─────────┐
           │  Any HIGH attacks │
           │  or blocks? (y/n) │
           └─┬────────────┬────┘
       yes │              │ no (converged)
           ▼              │
 ┌────────────────────┐   │
 │  STEP 4 — EXPERT   │   │
 │  (round 2)         │   │
 │                    │   │
 │  Specialists see:  │   │
 │   • their prior    │   │
 │     picks          │   │
 │   • Skeptic's      │   │
 │     HIGH attacks   │   │
 │  They may defend   │   │
 │  or switch picks.  │   │
 └─────────┬──────────┘   │
           │              │
           ▼              │
 ┌────────────────────┐   │
 │  STEP 5 — SKEPTIC  │   │
 │  (round 2)         │   │
 │                    │   │
 │  Final attack pass │   │
 └─────────┬──────────┘   │
           │              │
           └──────┬───────┘
                  ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STEP 6 — CHIEF ENGINEER                                    │
 │                                                             │
 │  Inputs: surviving Expert picks, Skeptic attacks,           │
 │  user objectives ("de-risk" / "yield" / "throughput" /      │
 │  "balanced"), shortlist table                               │
 │                                                             │
 │  LLM picks winner + writes reasoning + open-risks list.     │
 │  Fallback: deterministic majority vote on non-blocked picks │
 │  with tie-break by attack severity and Pareto membership.   │
 │                                                             │
 │  Output: winning_pick_id, winner_reasoning,                 │
 │  open_risks[], changes_to_apply{}                           │
 └──────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STEP 7 — APPLY & RE-VALIDATE                               │
 │                                                             │
 │  • Patch FlowProposal with winner's (τ, d, Q, V_R)         │
 │    plus any Chief overrides (BPR, material, deox method).   │
 │  • Re-run 9-step Design Calculator on the patched proposal  │
 │    to refresh all downstream numbers.                       │
 │  • Build AgentDeliberation records (Designer, 4 specialists │
 │    + Skeptic + Chief) for UI compatibility with v2 shape.   │
 │                                                             │
 │  Output: DesignCandidate + updated DesignCalculations       │
 └─────────────────────────────────────────────────────────────┘
```

### 4.5 Convergence

The Expert ⇄ Skeptic loop is capped at **2 rounds**. Early convergence
when the Skeptic returns zero HIGH attacks and zero hard blocks. The cap
prevents the ping-pong failure mode common in self-refining LLM loops.

### 4.6 What the council *never* does

- Invent numeric values for τ, d, Q, Re, ΔP, productivity, Beer-Lambert A,
  etc. These come from `compute_metrics`.
- Skip the calculator. Every round of changes re-invokes `DesignCalculator.run`
  to refresh downstream quantities.
- Override hard safety rules. The Skeptic's code layer is final.
- Score candidates with a weighted sum. The Pareto front + specialist
  advocacy + user-stated objectives replace arbitrary weights.

### 4.7 Objective elicitation

The Chief accepts a string-level user objective and maps it to a
selection preference:

| Objective keyword | Chief bias |
|---|---|
| `de-risk first-run` | Highest safety margins (ΔP headroom, r_mix, X well above threshold); accept lower productivity. |
| `yield-oriented` | τ with comfortable margin over τ_kinetics; mechanism-fit advocate wins ties. |
| `throughput-oriented` | Higher productivity_mg_h with acceptable margins. |
| `balanced` (default) | Fewest HIGH-severity attacks on a Pareto-front pick wins. |

---

## 5. Deterministic tools (`engine/tools.py` + `engine/sampling.py`)

Used by the Designer and throughout the pipeline. Every number FLORA shows
comes from one of these functions.

| Tool | Purpose |
|---|---|
| `beer_lambert(C, ε, ℓ)` | A = ε·C·(d/10); inner-filter risk classification |
| `calculate_reynolds(Q, d, solvent, T)` | Re + flow regime |
| `calculate_pressure_drop(Q, d, L, solvent)` | Hagen-Poiseuille ΔP |
| `calculate_mixing_ratio(d, τ)` | t_mix / τ |
| `calculate_bpr_required(T, solvent, ΔP, is_gl)` | BPR sizing + 7 bar floor for gas-liquid |
| `check_material_compatibility(mat, solv, T)` | FEP/PFA/PTFE/SS/PEEK rules |
| `check_redox_feasibility(E*, E_ox)` | SET thermodynamic check |
| `compute_design_envelope(center, ...)` | ±30 % feasible window for BO |
| `sampling.sample_design_space(...)` | Deterministic (τ, d, Q, L_frac) triplets |
| `sampling.compute_metrics(...)` | Full metric set for one candidate |
| `sampling.hard_filter(...)` | Bench-physics reject rules |
| `sampling.generate_candidates(...)` | End-to-end sample → metric → filter → Pareto → diversity-reduce |

---

## 6. Chemistry reasoning — the three-layer architecture

Distinct LLM roles with clear separation:

- **Layer 1 — Chemistry Agent:** mechanism, species, stream logic,
  sensitivities. No hardware decisions. Injects handbook rules filtered
  by mechanism/solvent/photocatalyst.
- **Layer 2 — Translation Agent:** FlowProposal generation from
  ChemistryPlan + retrieved analogies + Calculator center-point.
- **Layer 3 — ENGINE Council (v3):** Designer / Expert / Skeptic / Chief
  (see Section 4).

This layering keeps each LLM's system prompt short and narrow, which
matters most for local models (gemma4) where context length and
reasoning consistency are limiting.

---

## 7. Retrieval — hard metadata filters

ChromaDB semantic search is pre-filtered by hard metadata (reaction class,
photocatalyst class, solvent family, phase regime) before the 20-top
similarity reranking. Prevents cross-class leakage (e.g., thermal
Suzuki couplings being retrieved for a photoredox query).

Two collections:
- `flora_records` — single-paper records (batch OR flow)
- `flora_pairs` — batch↔flow pairs for direct analogy retrieval

---

## 8. Process flow diagram

Rendered via Graphviz in `flora_design/visualizer/flowsheet_builder.py`
with real equipment icons:

- **Syringe pump** icon for liquid reagent streams
- **MFC (Mass Flow Controller)** red-bordered box for gas streams —
  automatically detected via `_is_gas_pump()` based on stream
  contents (O₂, H₂, CO₂, N₂, Ar, air, gas, mfc keywords)
- **Coil reactor** / **photoreactor** / **microchannel** / **BPR** /
  **vial** / **mixer (T / cross)** icons as appropriate
- Labels show τ, temperature, wavelength, concentrations, flow rates

---

## 9. LLM provider abstraction

`ENGINE_PROVIDER` in `flora_translate/config.py` switches backends:

- `"anthropic"` → Claude Sonnet / Opus (default)
- `"openai"` → GPT-4o or GPT-4o-mini
- `"ollama"` → local model (currently `gemma4-flora`) at
  `OLLAMA_BASE_URL`

Local models require:
- `think=False` API flag + `/no_think` user prefix to suppress Gemma
  thinking mode (otherwise all tokens spent on internal reasoning,
  content field empty)
- 3× max_tokens budget for complex specialist prompts
- JSON output reminder appended at the end of the user message
- Custom Modelfile with `num_ctx 8192` (the API-level `num_ctx` override
  is ignored by Ollama; must be baked into the model)

---

## 10. Technology stack

- **LLM orchestration:** Anthropic Python SDK, OpenAI SDK (for GPT + Ollama)
- **Vector DB:** ChromaDB
- **UI:** Streamlit
- **Diagramming:** Graphviz
- **Numerics:** scikit-learn (GP surrogate), scipy, numpy
- **Chemistry:** RDKit (optional, for SMILES parsing)
- **PDF extraction:** pymupdf, pdfplumber

---

## 11. How to run

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

## 12. Limitations & roadmap

**Current limitations:**

- **Point estimates, not distributions.** The calculator treats k, ε, ΔH
  as known. Real values are uncertain (± 2–3× for ε at LED wavelength,
  ± 30–50 % for k estimated from batch). Future: Monte Carlo propagation
  of parameter uncertainty through the metric computation.
- **Open-loop design.** No feedback from lab runs. Future: closed-loop
  calibration where experimental yield/pressure/issues update priors on
  k, ε, mixing constants for the reaction class.
- **Single-stage focused.** Multi-stage sequential chemistries (e.g., Giese +
  aerobic oxidation in one pot) are supported but the council reasoning
  does not yet debate stage-boundary hardware (gas segmentation, inline
  degassing).
- **Fixed objective ontology.** Chief accepts four predefined objective
  strings. Future: fine-grained user priority elicitation (Phase 0).

**In progress / near-term:**

- Information-gain-based experimental plan output ("run this τ first
  because it reduces k uncertainty the most").
- Feedback-log-driven calibration of handbook rules and class IF bounds.

---

## 13. Repository map

```
flora_translate/
├── config.py                    ENGINE_PROVIDER + per-component models
├── design_calculator.py         9-step deterministic calculator
├── input_parser.py              Free text → BatchRecord
├── chemistry_agent.py           Layer 1: mechanism / stream logic
├── translation_llm.py           Layer 2: FlowProposal generation
├── revision_agent.py            Targeted patch to existing design
├── schemas.py                   All Pydantic models
└── engine/
    ├── llm_agents.py            Provider-agnostic call_llm()
    ├── tools.py                 Deterministic domain tools
    ├── sampling.py              v3 design-space sampling + Pareto
    ├── triage.py                GREEN/YELLOW/RED classification
    ├── orchestrator.py          v2 council (parallel review) — LEGACY
    ├── agents_v2.py             v2 specialist prompts — LEGACY
    └── council_v3/
        ├── __init__.py          Exposes CouncilV3
        ├── designer.py          Sampling-strategy agent
        ├── expert.py            4-advocate panel
        ├── skeptic.py           Assumption attacker + code rules
        └── chief.py             CouncilV3 orchestrator class

flora_design/
└── visualizer/
    └── flowsheet_builder.py     Graphviz flowsheet; MFC for gas

flora_fundamentals/
├── extractor.py                 Two-pass handbook rule extraction
└── data/rules.json              2537 extracted rules

pages/
├── flora_design_unified.py      Main Streamlit page
└── ...                          Other tabs (BO, diagnostics, KB)
```
