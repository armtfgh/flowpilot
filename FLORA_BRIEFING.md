# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing Document

**Version:** 2.1 | April 2026
**Status:** Active Development
**Codebase:** ~19,500+ lines across 70+ files
**Last major update:** April 2026 — τ Source-of-Truth Fix (single authoritative residence time across proposal, design calculations, diagram, and explanation; DesignCalculator council-approved override path; OutputFormatter anchor injection)

---

## 1. Executive Summary

FLORA is an AI-powered platform that automates the translation of batch chemistry protocols into validated continuous flow process designs. It combines **literature-grounded retrieval-augmented generation (RAG)**, a **multi-agent deliberation council** where five LLM-powered specialist engineers debate and converge on designs, **deep chemistry reasoning**, and **handbook-level foundational knowledge** (2,537 rules from 6 textbooks) to produce proposals that a process chemist can take directly to the lab.

The system addresses a core bottleneck in flow chemistry adoption: designing a flow process from a batch protocol requires deep domain expertise in reaction engineering, fluid dynamics, photochemistry, and hardware compatibility — knowledge that is scattered across hundreds of journal articles and textbooks. FLORA consolidates this knowledge into a queryable, validated pipeline.

### What FLORA produces

Given a batch protocol like:

> *"Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative radical addition of N-Boc-proline to methyl vinyl ketone, K2HPO4, DMF, 0.1M, 25°C, 450nm LED, N2, 24h, 72% yield"*

FLORA returns:

- A **validated flow design** with named chemical streams, reactor specifications, and operating conditions
- A **process flow diagram** (SVG/PNG) showing actual chemicals in each stream
- A **multi-agent engineering deliberation report** with full chain-of-thought from 5 specialist agents (kinetics, fluidics, safety, chemistry, process integration)
- **Literature references** supporting every design decision
- **Handbook-grounded rules** injected from an extracted fundamentals knowledge base
- A **confidence score** (HIGH/MEDIUM/LOW) based on analogy quality

---

## 2. System Architecture

FLORA is structured as **six backend modules** sharing a common data layer, exposed through a unified **Streamlit web dashboard**.

```
┌──────────────────────────────────────────────────────────────────┐
│                        STREAMLIT DASHBOARD                        │
│                                                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│  │ Batch-to-   │  │ Process      │  │ Condition Optimization │   │
│  │ Flow        │  │ Design       │  │ (Bayesian BO)          │   │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘   │
│         │                │                        │               │
│  ┌──────┴────────────────┴────────────────────────┘              │
│  │                  SHARED BACKEND                                │
│  │                                                                │
│  │  ┌──────────────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │  │ ENGINE Council   │ │ ChromaDB │ │ Lab Inventory        │ │
│  │  │ (5 LLM agents + │ │ Vector   │ │ (hardware specs)     │ │
│  │  │  Orchestrator +  │ │ Store    │ └──────────────────────┘ │
│  │  │  2537 rules)     │ └──────────┘                          │
│  │  └──────────────────┘                                        │
│  │                                                                │
│  │  ┌──────────────────────────────────────────────────────┐    │
│  │  │ KNOWLEDGE PIPELINE                                    │    │
│  │  │                                                       │    │
│  │  │  Literature Mining → PDF Extraction → ChromaDB       │    │
│  │  │                                                       │    │
│  │  │  Handbook Extraction → Fundamentals Rules → Prompt   │    │
│  │  └──────────────────────────────────────────────────────┘    │
│  └──────────────────────────────────────────────────────────────│
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Modules

### 3.1 FLORA-Translate — Batch-to-Flow Translation

**Purpose:** Convert an existing batch chemistry protocol into a validated flow design.

**When to use:** A researcher has already run a reaction in batch and wants to move it to flow.

**Pipeline (7 steps):**

```
Batch Protocol (text or structured input)
    │
    ▼
[1] INPUT PARSER
    Normalises free text or JSON into a structured BatchRecord.
    Uses Claude for free-text parsing.
    │
    ▼
[2] CHEMISTRY REASONING AGENT (Layer 1)
    Pure chemistry analysis — NO hardware decisions.
    │
    │  Identifies every species: substrate, photocatalyst, oxidant,
    │  base, solvent, additives — by name, role, and equivalents.
    │
    │  Proposes the reaction mechanism step-by-step:
    │  photoexcitation → SET/EnT/HAT → radical addition → product
    │
    │  Determines stream separation logic:
    │  "Photocatalyst must be co-dissolved with substrate (short-lived
    │   excited state). Oxidant in separate stream to prevent dark
    │   reaction before the reactor."
    │
    │  Flags sensitivities: O₂-sensitive, moisture-sensitive,
    │  light-sensitive reagents.
    │
    │  Automatically injects relevant FUNDAMENTALS RULES from the
    │  handbook knowledge base into its prompt (e.g., "FEP tubing
    │  required for photochemistry, ID 0.5–2.0 mm").
    │
    │  Generates retrieval keywords for smarter database search.
    │
    │  Output: ChemistryPlan
    │
    ▼
[3] PLAN-AWARE RETRIEVAL (Layer 2)
    Searches ChromaDB for the most similar batch→flow translations
    from the literature corpus.
    │
    │  THREE-TIER RETRIEVAL with hard metadata filters:
    │
    │  Tier 1 (most specific): Hard filter by mechanism_type AND
    │          phase_regime + pairs-only collection.
    │          e.g., only SET radical papers for a SET radical query.
    │
    │  Tier 2 (fallback): If <3 results, relax hard filters.
    │          Full semantic + field reranking, pairs-only.
    │
    │  Tier 3 (last resort): All records (no translation-pair filter).
    │
    │  Field reranking weights within each tier:
    │    Photocatalyst class: 30%, Solvent: 20%, Wavelength: 20%,
    │    Temperature: 15%, Concentration: 15%
    │
    │  Final score = 0.6 × semantic + 0.4 × field similarity
    │  Returns top-3 analogies.
    │
    ▼
[4] TRANSLATION LLM
    Claude receives:
    - The ChemistryPlan (authoritative for chemistry decisions)
    - The 3 literature analogies (authoritative for hardware parameters)
    - The batch protocol
    │
    │  Generates a FlowProposal with:
    │  - Named stream assignments (actual chemicals, not "reagent 1")
    │  - Hardware parameters (residence time, flow rate, reactor, BPR)
    │  - Per-field reasoning citing analogies or chemistry rules
    │
    ▼
[5] ENGINE DELIBERATION COUNCIL (Layer 3)
    Five LLM-powered specialist agents deliberate the design:
    │
    │  Round 1 — Independent Analysis (5 parallel agents):
    │    Dr. Kinetics    — τ validation, PFR design eq., Da analysis
    │    Dr. Fluidics    — ΔP, Re, mixing quality, Dean vortices
    │    Dr. Safety      — thermal runaway, ΔT_ad, material compat
    │    Dr. Chemistry   — mechanism fidelity, Beer-Lambert, streams
    │    Dr. Process     — unit ops sequence, STY, scale-up path
    │
    │  Each agent: chain-of-thought + calculations + proposals
    │  Each agent receives 25 domain-specific handbook rules
    │
    │  Chief Engineer — cross-agent sanity check, conflict resolution
    │
    │  Round 2 — Cross-Agent Debate:
    │    All agents see each other's Round 1 findings.
    │    Agree, disagree, or refine with cross-domain reasoning.
    │
    │  Round 3 — only if unresolved conflicts remain (max 3)
    │
    ▼
[6] OUTPUT FORMATTING
    Human-readable explanation generated by Claude.
    Confidence scoring: HIGH (similarity >0.85, 1 council round),
    MEDIUM (>0.65, ≤2 rounds), LOW (otherwise).
    │
    ▼
[7] PROCESS DIAGRAM GENERATION
    Converts the validated FlowProposal + ChemistryPlan into a
    ProcessTopology with chemistry-aware labels, then renders
    a publication-quality SVG/PNG flowsheet showing actual
    chemical names, concentrations, and flow rates on each node.
```

### 3.2 FLORA-Design — Process Design from Scratch

**Purpose:** Design a flow process from a free-text chemistry goal — no batch protocol needed.

**When to use:** A researcher is planning a new flow campaign and hasn't done the reaction in batch.

**Pipeline (8 steps):**

```
Chemistry Goal (free text)
    │
    ▼
[1] CHEMISTRY CLASSIFIER
    LLM extracts structured features: reaction class, photocatalyst,
    wavelength, sensitivities, hazard level, phase regime.
    Post-processed with keyword lookup and photocatalyst database.
    │
    ▼
[2] UNIT OPERATION SELECTOR (rule-based, no LLM)
    Deterministic logic selects required unit operations:
    │
    │  ALWAYS: pumps, mixer, photoreactor, LED module, collector
    │  IF O₂-sensitive: deoxygenation unit
    │  IF gas generation OR high temp: back-pressure regulator
    │  IF solid catalyst: inline filter
    │  IF hazardous/exothermic: quench mixer
    │
    ▼
[3] TOPOLOGY AGENT (RAG + LLM)
    Retrieves 5 similar flow papers from ChromaDB.
    LLM refines unit operations, assigns stream connections,
    and produces a primary + alternative topology.
    │
    ▼
[4] PARAMETER AGENT
    Fills all numerical parameters using literature values
    or reaction-class defaults. Validates τ × Q = V consistency.
    │
    ▼
[5–8] ENGINE validation → Diagram → Explanation → Output
    (Same shared infrastructure as FLORA-Translate)
```

### 3.3 FLORA-Fundamentals — Handbook Knowledge Base

**Purpose:** Extract and inject foundational flow chemistry rules from textbooks and handbooks into every query — giving the agent deterministic, expert-level domain knowledge that does not depend on finding a relevant paper in the corpus.

**Why this matters:** RAG retrieval finds similar *reactions*. Fundamentals provides universal *engineering rules* (e.g., "FEP tubing required for photochemistry", "BPR needed when T > bp − 20°C") that apply regardless of what's in the corpus.

**How it works:**

```
Handbook PDFs (any size, outside or inside project)
    │
    ▼
[PASS 1 — Haiku scan, cheap]
    Each chunk (5 pages) is scanned by Claude Haiku.
    Scored 0–10 for relevance to flow chemistry rules.
    Chunks below threshold (default: 4/10) are SKIPPED.
    Typical skip rate: 60–70% of pages (intro, history,
    references, unrelated content).
    │
    ▼ (only relevant chunks proceed)
[PASS 2 — Sonnet extraction, detailed]
    Claude Sonnet reads each relevant chunk in full:
    text, tables, figures, formulas.
    Extracts structured rules:
      - Category (mixing, photochemistry, pressure, materials, ...)
      - Condition (WHEN this applies)
      - Recommendation (WHAT to do, with numbers)
      - Reasoning (WHY — the physics/chemistry)
      - Quantitative details (thresholds, formulas, ranges)
      - Severity (hard_rule | guideline | tip)
    │
    ▼
Saved to flora_fundamentals/data/rules.json
    │
    ▼ (automatic, no restart needed)
Agent injection (Chemistry Agent + ENGINE Council)
    Every translation query automatically loads relevant rules:
    - Chemistry Agent: rules filtered by mechanism type, solvent,
      photocatalyst, O₂ sensitivity, temperature.
    - ENGINE Council: each specialist agent receives 25 rules
      filtered by their domain (kinetics, fluidics, safety,
      photochemistry, reactor design, scale-up, materials).
```

**Cost savings:** Two-pass mode reduces Sonnet calls by ~60–70% on large handbooks.

| Handbook size | Single-pass cost | Two-pass cost | Savings |
|---|---|---|---|
| 100 pages | ~$1.00 | ~$0.55 | 45% |
| 300 pages | ~$3.00 | ~$1.40 | 53% |
| 500 pages | ~$5.00 | ~$2.10 | 58% |

**Input options:**
- Upload directly in the GUI
- Specify absolute file paths outside the project folder (for licensed books that cannot be committed to git)

### 3.4 Condition Optimization — Bayesian Optimization Helper

**Purpose:** Suggest the next experiment to run based on previous observations, using Gaussian Process regression and Expected Improvement acquisition.

**How it works:**
1. User defines a parameter space (e.g., residence time 1–30 min, temperature 0–100°C)
2. User enters experimental observations (conditions → yield)
3. FLORA fits a Gaussian Process surrogate model (Matern kernel, scikit-learn)
4. Computes Expected Improvement across 5,000 candidate points
5. Returns the next suggested experiment + GP surface plot + acquisition function plot

**Self-contained:** No LLM or ChromaDB dependency. Runs entirely on scikit-learn + scipy.

### 3.5 Protocol Diagnostics — Standalone ENGINE

**Purpose:** Validate any flow chemistry protocol against engineering constraints without running the full translation pipeline.

**Input:** Flow protocol parameters (tubing material, ID, flow rate, temperature, BPR, solvent, etc.)

**Output:** Full ENGINE council report — pressure drop, Reynolds number, material compatibility, safety flags, with specific warnings and suggested fixes.

### 3.6 Knowledge Pipeline — Literature Mining, Extraction, Indexing

The knowledge pipeline populates the ChromaDB corpus that powers RAG retrieval.

```
Literature Mining (SCOUT)
    OpenAlex API → ranked paper list by citation count
    │
    ▼
Knowledge Extraction (PRISM)
    5-pass PDF extraction pipeline:
    │
    │  Pass 1: Document Intelligence (text-only)
    │  Pass 2: Chemistry Extraction (text-only)
    │  Pass 3: Quantitative Conditions (vision @ 150 DPI)
    │  Pass 4: Figure-by-Figure Visual Extraction
    │           (Haiku classifies → Sonnet extracts non-"other")
    │  Pass 5: Synthesis + Validation (text-only merge)
    │
    │  Output: structured JSON record per paper
    │
    ▼
Indexing
    For each record:
    1. Claude generates a natural language summary
    2. OpenAI text-embedding-3-small produces a 1536-dim vector
    3. Record + vector stored in ChromaDB with filterable metadata:
       doi, chemistry_class, mechanism_type, phase_regime,
       photocatalyst, wavelength_nm, solvent, reactor_type,
       has_translation, confidence, year
```

---

## 4. The ENGINE — Multi-Agent Deliberation Council

The ENGINE is the shared engineering validation layer used by both Translate and Design. It implements a **multi-agent deliberation architecture** where five LLM-powered specialist agents independently analyze, debate, and converge on a validated flow design.

### 4.1 Architecture — LLM-Powered Specialist Agents

Each agent is a Claude LLM call with a deep domain-specific system prompt, injected with relevant fundamentals rules from the handbook knowledge base (2,537 rules across 6 textbooks). Agents receive the full proposal, chemistry plan, and 9-step engineering calculations.

| Agent | Domain | Capabilities |
|-------|--------|-------------|
| **Dr. Kinetics** | Reaction kinetics, residence time, conversion | PFR design equations, Arrhenius corrections, Damköhler analysis, intensification factor validation. Can PROPOSE alternative τ with mechanistic justification. |
| **Dr. Fluidics** | Pressure drop, mixing, mass transfer | Hagen-Poiseuille, Reynolds, Dean number, mixing time estimation. Can PROPOSE different tubing ID, mixer type, or reactor geometry. |
| **Dr. Safety** | Thermal safety, materials, pressure | Adiabatic temperature rise, thermal Damköhler, material compatibility, BPR sizing. Can PROPOSE concentration limits, material changes, safety additions. |
| **Dr. Chemistry** | Mechanism fidelity, streams, photochemistry | Beer-Lambert light budget, stream separation logic, wavelength matching, quench chemistry. Can PROPOSE stream reassignments, catalyst changes, concentration adjustments. |
| **Dr. Process** | Integration, scale-up, unit operations | Space-time yield, throughput, process architecture, numbering-up strategy. Can PROPOSE architectural changes, inline analytics, staging. |

Each agent outputs structured JSON with:
- **chain_of_thought**: Full reasoning — references calculator values by name, does not re-derive them
- **values_referenced**: Calculator values cited (e.g. "Re = 7.75 (from calculator)")
- **findings**: Bullet-point assessment results
- **proposals**: Structured field changes `{"field": "tubing_ID_mm", "value": "0.75", "reason": "..."}`
- **concerns**: Issues that need resolution
- **rules_cited**: Handbook fundamentals rules used
- **had_error**: True if the agent call failed — blocks convergence declaration

**Anti-hallucination design**: A shared preamble in every agent's system prompt prohibits
re-deriving values the calculator already computed. Agents read and interpret authoritative
physics — they never invent rate constants, extinction coefficients, or other ungrounded numbers.

### 4.2 Deliberation Protocol

```
Round 1 — Independent Analysis
  All 5 agents analyze the proposal independently.
  Each produces: chain_of_thought + values_referenced + findings
                + structured proposals + concerns
      │
      ▼
Chief Engineer — Sanity Check
  Reviews structured proposals from REVISE agents only.
  Resolves conflicts (safety > chemistry > physics > kinetics > integration).
  Outputs final_changes: only simple numeric/string fields — no lists.
  Applies changes → re-runs DesignCalculator (τ×Q=V enforced).
  Tracks all_changes_applied cumulative dict across rounds.
      │
      ▼
Round 2 — Cross-Agent Debate
  Each agent sees ALL other agents' Round 1 findings.
  Each agent also receives its OWN Round 1 concerns with explicit
  instruction to RESOLVE, DEFER, or ESCALATE each one — verbatim
  copy of Round 1 output is forbidden.
      │
      ▼
Chief Engineer — Final Sanity Check
  Synthesizes consensus. Applies final changes.
  If unresolved conflicts remain → Round 3 (max 3 rounds).
      │
      ▼
Convergence — gated on ACTUAL resolution, not round count
  CONVERGE when: all agents ACCEPT, OR changes applied with no REVISE remaining.
  DO NOT CONVERGE when: any agent had_error=True, OR REVISE proposals unaddressed.
      │
      ▼
DeliberationLog — complete, immutable audit trail
  Per-round per-agent: chain_of_thought, proposals, concerns, had_error.
  Sanity checks: conflicts, resolutions, final_changes per round.
  all_changes_applied: cumulative dict of every field modified.
  Consensus output shows ALL modified fields (★ marked), not just τ/Q/V_R.
```

**Convergence rules (precise):**
- An agent that throws an exception gets `had_error=True` → never treated as a pass → forces another round
- All ACCEPT (no errors) → converge immediately
- Round ≥ 2 AND changes were actually applied → converge (changes addressed the REVISE flags)
- Round ≥ 2 AND only WARNING flags (no REVISE, no errors) → converge (acknowledged risks)
- Otherwise → continue to next round (up to max 3)

### 4.3 Key Differentiator vs Single-Agent LLM

The multi-agent architecture produces designs that a single LLM cannot:

1. **Cross-domain conflict resolution**: When kinetics demands high τ but fluidics flags ΔP issues, the agents negotiate a solution (e.g., increase tubing ID to accommodate both). The Chief Engineer resolves by priority: safety > chemistry > physics > kinetics > integration.
2. **Independent verification**: Each agent checks the design from its own domain perspective, catching errors a single generalist would miss.
3. **Handbook-grounded reasoning**: Each agent receives 25 domain-specific rules from 6 flow chemistry textbooks (2,537 total) — making their reasoning expert-level, not generic.
4. **Traceable chain-of-thought**: Every decision is logged with reasoning, calculator values cited, rules used, and structured proposals — enabling human audit and paper reproducibility.
5. **Debate-driven refinement**: Round 2 debate — each agent sees all other agents' findings AND must explicitly resolve/defer/escalate its own Round 1 concerns — catches interaction effects independent analysis misses.
6. **Machine-applicable proposals**: Agents output `{"field": "tubing_ID_mm", "value": "0.75", "reason": "..."}` not free text. Changes are applied deterministically and verified by DesignCalculator re-run.
7. **Anti-hallucination architecture**: Calculator values are labeled AUTHORITATIVE. A shared preamble explicitly forbids agents from re-deriving physics or inventing spectroscopic/kinetic data. Agents reference, not recompute.

### 4.4 Robustness Design

- **JSON parse failures**: 4-strategy parser (markdown strip → direct parse → brace matching → text extraction). No exception propagates to the orchestrator — instead `had_error=True` blocks convergence.
- **Chief Engineer validation**: Only 10 simple numeric/string fields may appear in `final_changes`. List fields (`streams`, `pre_reactor_steps`) are blocked and logged as warnings. A fallback applies agent proposals directly if the Chief Engineer LLM call fails.
- **Circular reasoning prevention**: Dr. Kinetics is explicitly forbidden from back-calculating k from calculator τ and then using that k to "confirm" τ — this is circular and adds no information. The intensification factor IS the validation.

---

## 5. Chemistry Reasoning — The Three-Layer Architecture

The key architectural innovation in FLORA is the separation of **chemistry reasoning** from **hardware translation**, further grounded by a **fundamentals knowledge layer**:

**Fundamentals Layer (before everything)**
Handbook rules are loaded and injected before any LLM reasoning begins:
- "FEP/PFA tubing required for photochemistry (transparent to visible light)"
- "BPR required when T > solvent bp − 20°C"
- "Residence time = reactor_volume / flow_rate (always verify consistency)"
- Hundreds of similar hard rules extracted from expert textbooks

**Layer 1 — ChemistryPlan (before retrieval) — Claude Opus**
Uses Claude Opus (the most capable model) with a structured two-section output:
- **Section 1 (reasoning):** Free-text chain-of-thought working through: chemistry type, rate-limiting step, why flow helps (quantitatively), species roles, stream logic, sensitivity flags. This reasoning is logged and passed downstream.
- **Section 2 (JSON):** Structured ChemistryPlan consistent with the reasoning above.
- Token budget capped at 2,048 tokens for cost control (~$0.15/call).
- Fundamentals rules injected ranked by relevance to the specific chemistry type.

**Step 2b — 9-Step Engineering Design Calculator (before translation)**
Pre-computed before the translation LLM runs via `DesignCalculator.run()`.

The calculator operates in two modes depending on context:

**Mode A — Kinetics-based (initial estimate, before council):**
Step 2 derives τ from batch data and intensification factor:
1. **Kinetics (multi-method):** Analogy-derived IF (primary when ≥ 2 analogies have batch+flow data), class-level IF (fallback), Arrhenius T-correction if T_flow ≠ T_batch. Reports τ as a range covering all methods.
2. **Reactor sizing:** V_R = τ×Q, L = 4V_R/(πd²) — all geometry derived from τ
3. **Fluid dynamics:** Re = ρvd/μ — auto-adjusts d upward if turbulent
4. **Pressure drop:** ΔP = 128μLQ/(πd⁴) via Hagen-Poiseuille — auto-adjusts d if > pump max
5. **Mass transfer:** t_mix = d²/(D·π²), Da = k·d²/D — flags mass-transfer limited regime
6. **Heat transfer:** Q_gen, Q_rem = U·A·ΔT_lm, thermal Da — flags exotherm risk
7. **BPR:** Antoine equation for vapor pressure + system ΔP + 0.5 bar margin; gas-liquid systems always require BPR (≥ 5 bar)
8. **Process metrics:** STY, productivity, intensification factor

**Mode B — Council-approved override (after council):**
When `target_residence_time_min` is provided (or when the proposal already has a validated τ),
step 2 uses that τ **directly** via `_step2_override()` — skipping the kinetics-based
estimation entirely. The council's decision is the authoritative source. Steps 3-8 are then
computed FROM this τ, not from batch data. This eliminates the divergence between
`proposal.residence_time_min` and `design_calculations.residence_time_min`.

**τ Single Source of Truth — enforced at three levels:**
1. **DesignCalculator**: council-approved τ bypasses kinetics re-derivation
2. **Orchestrator re-runs**: passes `target_residence_time_min=current.residence_time_min` after every council change
3. **main.py post-sync**: after formatting, forces `design_calculations["residence_time_min"]`, `residence_time_s`, and `reactor_volume_mL` to match `proposal["residence_time_min"]` — catches any remaining divergence

All steps are internally consistent (τ=V/Q, L=4V/πd², Re=ρvd/μ verified). Results injected
into translation prompt as a concise engineering block.

**Layer 2 — Translation (receives ChemistryPlan + calculations + analogies)**
The translation LLM (Sonnet) receives three explicit reasoning steps:
1. **Analogy comparison:** For each analogy, must state: (a) what is chemically similar, (b) key difference, (c) how that difference adjusts the parameters.
2. **Calculation validation:** Must verify residence time is within calculated range, BPR set if calculation requires it, Re and Da are reasonable.
3. **Conditions justification:** Every numerical field must cite its source: analogy N, calculation, chemistry plan, or first principles.

**Layer 3 — Multi-Agent Deliberation (ENGINE council)**
Five LLM-powered specialist agents (Dr. Kinetics, Dr. Fluidics, Dr. Safety, Dr. Chemistry, Dr. Process) independently analyze and then debate the design across 2-3 rounds. Each agent receives domain-specific handbook rules, the full 9-step calculations, and all other agents' findings. A Chief Engineer sanity-checker resolves cross-agent conflicts and applies consensus changes. See Section 4 for full architecture.

---

## 6. Retrieval — Hard Metadata Filters

A key improvement over standard RAG: ChromaDB retrieval uses **hard metadata pre-filters** before any semantic comparison, ensuring mechanistically incompatible records are excluded entirely.

```
Query: "SET radical mechanism, single-phase liquid, Ir photocatalyst"
    │
    ▼
Tier 1 — Hard filter (ChromaDB $and):
    mechanism_type = "radical"
    AND phase_regime = "single_phase_liquid"
    AND has_translation = True
    AND confidence >= 2
    → Returns only records that ARE SET radical reactions
    → A gas-liquid singlet O₂ paper never appears, even if photocatalyst matches
    │
    ▼ if < 3 results:
Tier 2 — Relax hard filters
    Soft semantic + field reranking only
    │
    ▼ if still < 3:
Tier 3 — All records (no translation-pair requirement)
```

Metadata stored per record: `doi`, `chemistry_class`, `mechanism_type`, `phase_regime`, `photocatalyst`, `wavelength_nm`, `solvent`, `reactor_type`, `has_translation`, `confidence`, `year`.

---

## 7. Conversational Interface

The Batch-to-Flow translation page uses a multi-turn chat interface (`flora_translate/conversation_agent.py`) instead of a single-shot form.

**How it works:**
- `ConversationAgent` wraps the translate pipeline with intent classification (TRANSLATE | REVISE | ANSWER | ASK).
- **TRANSLATE:** first message with a batch protocol → runs full pipeline → shows result in expandable card.
- **REVISE:** "add a liquid-liquid extraction" → RevisionAgent applies targeted LLM patch to existing design → re-validates via ENGINE → rebuilds diagram (~10s vs ~60s for full pipeline).
- **ANSWER:** "why did you choose PFA tubing?" → answers from context, no re-run.
- **ASK:** when critical info is missing → returns 1-3 targeted clarifying questions before running.
- After every translation: auto-checks confidence and missing fields → proactively asks user for clarification.
- Reset button clears conversation and starts fresh.

---

## 8. Corpus DOI List

Full DOI list of all 465 papers in the corpus is at `dois/corpus_dois.csv`.
- 455/465 papers have resolved DOIs.
- 10 unresolved: `photochemflow (1-9)` internal lab PDFs.
- DOI sources: filename match (248), CrossRef API (137), paper_miner_history (70).

---

## 9. Cost Optimisation

### Paper Extraction (PRISM)

Multiple optimisations reduce extraction cost by approximately **68%** compared to a naive implementation:

| Optimisation | Mechanism | Savings |
|---|---|---|
| Text for Passes 1–2 | PyMuPDF text extraction instead of page images | ~90% input tokens |
| 150 DPI for Pass 3 | Reduced resolution still reads tables accurately | ~40% vision tokens |
| Haiku for figure classification | Trivial task uses cheapest model | ~91% per classification |
| Skip "other" figures | No Sonnet extraction for decorative figures | ~1 Sonnet call saved per irrelevant figure |
| Prompt caching | System instructions cached across calls | ~90% discount on cached tokens |
| Batch API | Folder processing submitted as async batch | 50% discount |

**Estimated cost per paper:** ~$0.42 (real-time) or ~$0.21 (batch mode)

### Handbook Extraction (Fundamentals)

Two-pass mode saves ~60% of Sonnet calls by using Haiku to pre-screen relevance:

| Handbook size | Single-pass | Two-pass | Savings |
|---|---|---|---|
| 100 pages | ~$1.00 | ~$0.55 | 45% |
| 300 pages | ~$3.00 | ~$1.40 | 53% |
| 500 pages | ~$5.00 | ~$2.10 | 58% |

**Cost tracking:** Every API call across all modules is logged with model, token counts, and USD cost. Viewable via `python paper_knowledge_extractor.py --cost` or in the dashboard.

---

## 8. Data Flow

```
┌─────────────────────────────────────────────────┐
│              KNOWLEDGE SOURCES                   │
│                                                   │
│  ┌──────────────┐      ┌───────────────────────┐ │
│  │ Research     │      │ Handbooks /           │ │
│  │ Paper PDFs   │      │ Textbooks             │ │
│  └──────┬───────┘      └──────────┬────────────┘ │
└─────────┼──────────────────────────┼─────────────┘
          │                          │
          ▼                          ▼
   ┌─────────────┐           ┌──────────────┐
   │ PRISM       │           │ Fundamentals │
   │ 5-pass      │           │ Two-pass     │
   │ extraction  │           │ extraction   │
   └──────┬──────┘           └──────┬───────┘
          │                          │
          ▼                          ▼
   ┌─────────────┐           ┌──────────────┐
   │ JSON records│           │ rules.json   │
   │ (per paper) │           │ (FlowRule[]) │
   └──────┬──────┘           └──────┬───────┘
          │                          │
          │  Claude summary           │  Injected into
          │  + OpenAI embed           │  Chemistry Agent
          ▼                          │  + ENGINE Council
   ┌────────────────────┐            │
   │ ChromaDB           │            ▼
   │ flora_records      │    ┌───────────────────┐
   │ flora_pairs        │    │  Chemistry Agent  │
   │ (+ mechanism_type, │    │  (Layer 1)        │
   │   phase_regime     │    └────────┬──────────┘
   │   metadata)        │             │
   └──────────┬─────────┘             │
              │                        │
              └──────────┬─────────────┘
                         ▼
              ┌─────────────────────┐
              │ TRANSLATE / DESIGN  │
              │ pipeline            │
              └──────┬──────────────┘
                     ▼
              ┌─────────────────────┐
              │ ENGINE Council      │ ← rules.json
              │ (5 LLM agents +    │   (25 rules/agent)
              │  Orchestrator)      │
              └─────────────────────┘
```

---

## 9. User-Facing Dashboard

The Streamlit dashboard organises all functionality into three sections:

### Section 1 — DESIGN
| Page | Input | Output |
|------|-------|--------|
| **Batch-to-Flow** | Batch protocol (text or form) | Flow proposal + chemical diagram + validation |
| **Process Design** | Chemistry goal (text) | Process topology + diagram + validation |

### Section 2 — EVALUATE
| Page | Input | Output |
|------|-------|--------|
| **Protocol Diagnostics** | Flow protocol parameters | ENGINE council report |
| **Condition Optimization** | Experimental observations | Next experiment suggestion + GP plot |

### Section 3 — KNOWLEDGE (admin)
| Page | Input | Output |
|------|-------|--------|
| **Fundamentals** | Handbook PDFs (upload or file path) | Extracted rules, auto-injected into queries |
| **Literature Mining** | Search topic | Ranked paper list (CSV) |
| **Knowledge Extraction** | Research PDF uploads | Structured JSON records |
| **Knowledge Base** | Filters | Browsable corpus table |

### Feedback System

Every design output includes an **Approve / Needs Correction** widget. Corrections capture per-field changes (residence time, BPR, tubing material, etc.) with free-text notes, building a feedback log (`data/feedback_log.json`) that compounds over time.

---

## 10. Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM (reasoning) | Claude Opus 4.6 (chemistry analysis) |
| LLM (translation + ENGINE council) | Claude Sonnet 4.6 (translation, 5 specialist agents, orchestrator, revision agent, extraction) |
| LLM (cheap scan) | Claude Haiku 4.5 (figure classification, handbook relevance scan) |
| Multi-agent orchestration | Custom deliberation orchestrator with inter-agent debate protocol |
| Domain knowledge | 2,537 handbook rules from 6 textbooks, category-filtered per agent |
| Embeddings | OpenAI text-embedding-3-small (1536 dimensions) |
| Vector database | ChromaDB (persistent, two collections: flora_records, flora_pairs) |
| PDF processing | PyMuPDF (text extraction + page rendering) |
| Physics calculations | 9-step first-principles design calculator (pure Python) |
| Bayesian optimisation | scikit-learn GaussianProcessRegressor + scipy EI |
| Web framework | Streamlit |
| Diagram generation | Graphviz (publication-quality PFD with equipment icons) |
| Data validation | Pydantic v2 |
| Literature search | pyalex (OpenAlex API) |

---

## 11. April 2026 Engineering Overhaul

### 11.1 — 9-Step First-Principles Design Calculator

`flora_translate/design_calculator.py` replaces the former lightweight `flow_calculator.py`.
The new calculator runs a complete, internally consistent engineering design from batch data:

```
Step 1 — Parse batch conditions     T, C₀, t_batch, X_target
Step 2 — Kinetics & residence time  Multi-method: analogy IF → class IF → Arrhenius correction
Step 3 — Reactor sizing             V_R = τ×Q, L = 4V_R/(πd²)
Step 4 — Fluid dynamics             Re = ρvd/μ, flow regime check, auto-adjust d if turbulent
Step 5 — Pressure drop              ΔP = 128μLQ/(πd⁴), auto-adjust d if ΔP > pump max
Step 6 — Mass transfer              t_mix = d²/(D·π²), Da = k·d²/D
Step 7 — Heat transfer              Q_gen = |ΔH_r|·r·V_R, Q_rem = U·A·ΔT_lm, Da_th
Step 8 — BPR sizing                 P_BPR = P_vap(T) + ΔP_sys + 0.5 bar (Antoine eq)
Step 9 — Process metrics            STY, productivity, intensification factor
```

#### Step 2 — Multi-Method Kinetics

Step 2 estimates residence time via three parallel methods, choosing the most data-rich
one as primary and reporting all as a range:

**Method A — Analogy-derived IF (primary when data available):**
`_extract_analogy_IFs()` pulls batch/flow time ratios from the top retrieved literature
analogies — either from `translation_logic.time_reduction_factor` or computed as
`batch_baseline.reaction_time_min / flow_optimized.residence_time_min`.
When ≥ 2 analogies have data, the median IF is used as the primary estimate.

**Method B — Class-level IF (fallback):**
Hardcoded table of intensification factors by reaction class (photoredox 48×, thermal 10×,
hydrogenation 50×, etc.). Used when no analogy data is available.

**Method C — Arrhenius temperature correction:**
Applied when T_flow ≠ T_batch:
```
k(T_flow)/k(T_batch) = exp(−Ea/R × (1/T_flow − 1/T_batch))
τ_corrected = τ_base / k_ratio
```
Ea estimated by reaction class: photoredox 25 kJ/mol (photon-limited, weak T dependence),
thermal 80 kJ/mol (strong), hydrogenation 45 kJ/mol, cross-coupling 65 kJ/mol.

**Priority logic:**

| Analogy data points | Method used | IF source |
|---|---|---|
| ≥ 2 | `analogy` | median of analogy IFs |
| 1 | `analogy+class` | average of (analogy IF, class IF) |
| 0 | `class` | hardcoded class table |

**Range reporting:**
```
τ = 21.2 min (range: 14.1–45.0 min), via analogy (IF = 68×)
```
Range = `[min(τ_methods)/1.5, max(τ_methods)×1.5]`, capturing inter-method uncertainty.

**Disagreement warning:**
When analogy and class methods disagree by > 3×, a warning is emitted recommending
experimental verification.

**Consistency guarantee** — verified after every run:
- τ = V_R / Q
- L = 4·V_R / (π·d²)
- Re = ρ·v·d / μ
- v = 4·Q / (π·d²)
- ΔP = 128·μ·L·Q / (π·d⁴)

**Council-approved override mode** — `_step2_override(tau_min, ...)`:
Triggered when `target_residence_time_min` is passed to `run()`, or when the proposal
already has a validated `residence_time_min > 0`.  Step 2 sets τ directly and marks
`kinetics_method = "council-approved"`.  The implied IF is back-calculated for display only
(`implied_IF = batch_time / τ`).  Steps 3-8 proceed normally, computing all geometry and
transport properties FROM this authoritative τ.

```python
# Before fix — three divergent values:
design_calculations.residence_time_min = 16.0   # kinetics re-derived
proposal.residence_time_min            = 15.0   # council decision
topology reactor node                   = 7.5   # per-stage (÷2 for 2-stage)

# After fix — single source of truth:
design_calculations.residence_time_min = 15.0   # council-approved override
proposal.residence_time_min            = 15.0   # council decision
topology reactor node                   = 15.0  # from proposal (single-step)
summary explanation                    = 15 min # anchored in OutputFormatter prompt
```

**OutputFormatter anchor injection** — `_generate_explanation()` prepends a block of
AUTHORITATIVE DESIGN NUMBERS (τ, Q, V_R, T, C, material, λ, BPR) to the explanation
system prompt.  The LLM must cite these exactly — preventing hallucination of wrong
residence times ("30 min") from inconsistent context.

**Gas-liquid detection** — `_is_gas_liquid()` inspects atmosphere, description, and stream
contents. If a reagent gas (O₂, H₂, CO₂, CO, etc.) is present, BPR is mandatory regardless
of boiling point, minimum 5 bar to maintain Henry's-law gas solubility. N₂/Ar used purely
as inert atmosphere are **not** counted as gas-liquid.

**Streamlit display** — `components/design_steps.py` renders each step with:
- Status badge (PASS / WARNING / FAIL / ADJUSTED / ESTIMATED / council-approved)
- LaTeX equations with real substituted numbers
- Computed values as metric cards
- Warnings, adjustments, and assumptions

**Backward compatibility** — `DesignCalculations` exposes `estimated_rt_min`, `bpr_minimum_bar`,
`damkohler_number`, `damkohler_interpretation`, and `to_prompt_block()` so all existing
prompt-builder and ENGINE code works without changes.

---

### 11.2 — LLM-Powered Multi-Agent Deliberation Council

`flora_translate/engine/orchestrator.py` + `flora_translate/engine/llm_agents.py` — complete
rewrite of the ENGINE architecture from rule-based validation to LLM-powered deliberation.

**Old architecture (rule-based):** Five Python-only agents with `if/else` checks. No agent
could reason about trade-offs or propose alternatives. The "debate" was mechanical field-patching.
Proposals were free text — never parsed or applied. Convergence was declared by round count
regardless of agent flags. Agent crashes were silently swallowed.

**New architecture (LLM-powered deliberation):**

Each of the 5 agents is now a Claude Sonnet LLM call with:
- A deep domain-specific system prompt (~120 lines of expert knowledge per agent)
- 25 domain-filtered rules from the handbook knowledge base (2,537 total)
- Full DesignCalculator results labeled **AUTHORITATIVE** — agents read and interpret, do not re-derive
- Structured `FieldProposal` output `{"field": "...", "value": "...", "reason": "..."}` — machine-applicable

**Key engineering decisions:**
- `had_error: bool` on `AgentDeliberation` — any exception → blocks convergence
- `all_changes_applied: dict` on `DeliberationLog` — cumulative record of all modified fields, shown with ★ in consensus output
- `_ALLOWED_CHANGE_FIELDS` whitelist (10 fields) — Chief Engineer cannot output list/nested fields
- 4-strategy JSON parser — markdown strip → direct parse → brace matching → text extraction
- Own-prior-concerns injection: each agent receives its Round 1 concerns in Round 2 and MUST resolve/defer/escalate each one — verbatim copy is explicitly forbidden
- Anti-circular-kinetics: Dr. Kinetics is prohibited from back-calculating k from calculator τ and using it to confirm τ
- τ single-source-of-truth: Orchestrator passes `target_residence_time_min` to DesignCalculator re-runs so council-approved τ is never overridden by kinetics re-derivation

**Key files:**
- `flora_translate/engine/llm_agents.py` — 5 specialist agents, rule loader, context builder, robust JSON parser
- `flora_translate/engine/orchestrator.py` — deliberation loop, Chief Engineer, proposal patcher, convergence logic
- `flora_translate/schemas.py` — `FieldProposal`, `AgentDeliberation` (with `had_error`), `SanityCheckResult`, `DeliberationLog` (with `all_changes_applied`)

**RevisionAgent** (`flora_translate/revision_agent.py`): For chat-based revision requests,
a single LLM call patches the existing design (no full pipeline re-run), then re-validates
through the ENGINE council. Latency: ~10-15s vs ~60s for full translate.

---

### 11.3 — Confidence Scoring Fix

`flora_translate/output_formatter.py` and `flora_translate/retriever.py` — two compounding
bugs were fixed that made confidence almost always LOW.

**Bug 1 — Semantic similarity formula.** ChromaDB returns L2 (Euclidean) distances.
The old formula `1/(1+L2)` severely underestimates similarity for normalised embeddings.
Fixed to use the correct cosine similarity approximation for unit vectors:
```
cosine_similarity = max(0, 1 − L²/2)
```

| L2 distance | Old `1/(1+L2)` | New `1−L²/2` |
|---|---|---|
| 0.5 | 0.67 | **0.875** |
| 0.8 | 0.56 | **0.68** |
| 1.0 | 0.50 | 0.50 |

**Bug 2 — HIGH confidence was unreachable.** The threshold `rounds == 1` could never be
satisfied because `MIN_COUNCIL_ROUNDS = 2`. Fixed thresholds:

| Confidence | Old condition | New condition |
|---|---|---|
| HIGH | score > 0.85 AND rounds = 1 (impossible) | score > 0.75 AND rounds ≤ 2 |
| MEDIUM | score > 0.65 AND rounds ≤ 2 | score > 0.50 AND rounds ≤ 3 |
| LOW | everything else | everything else |

---

### 11.3b — Residence Time Single Source of Truth

**Problem identified:** Four different UI locations displayed different residence time values
for the same design:

| Location | Source | Old value |
|---|---|---|
| Summary explanation | LLM-generated from full dict | "30 min" (hallucinated) |
| Process diagram reactor node | `proposal.residence_time_min` | 15 min |
| Engineering Design tab | `design_calculations.residence_time_min` | 16 min (kinetics re-derived) |
| Council consensus line | `proposal.residence_time_min` | 15 min |

**Root cause:** The `DesignCalculator._step2()` always recomputes τ from `batch_time / IF`
regardless of what the council approved. After the council set τ = 15 min, a DesignCalculator
re-run in the orchestrator produced τ = 16 min (different IF from analogies). The summary LLM
received both 15 and 16 in its context and hallucinated "30 min".

**Fix (three-layer):**

1. **`DesignCalculator.run(target_residence_time_min=...)`** — new parameter. When provided
   (or when the proposal already has a validated τ > 0), Step 2 uses `_step2_override()`:
   sets τ directly, marks `kinetics_method = "council-approved"`, and back-calculates the
   implied IF for display only. Steps 3-8 compute geometry/transport from this authoritative τ.

2. **`Orchestrator` re-runs** — after council applies changes, passes
   `target_residence_time_min=current.residence_time_min`. The calculator never re-derives τ
   from kinetics after the council has approved it.

3. **`main.py` post-sync** — after formatting, force-syncs:
   ```python
   result["design_calculations"]["residence_time_min"] = result["proposal"]["residence_time_min"]
   result["design_calculations"]["residence_time_s"]   = τ × 60
   result["design_calculations"]["reactor_volume_mL"]  = τ × Q
   ```
   Safety net that catches any remaining divergence.

4. **`OutputFormatter._generate_explanation()`** — prepends AUTHORITATIVE DESIGN NUMBERS block
   to the explanation system prompt, anchoring the LLM to the exact τ, Q, V_R, T, C values.
   Prevents hallucination of wrong residence times from inconsistent context.

---

### 11.4 — Process Flow Diagram Improvements

`flora_design/visualizer/flowsheet_builder.py`:

| Change | Detail |
|--------|--------|
| **Uniform icon sizes** | All nodes (pump, reactor, BPR, vial, mixer) rendered at `NODE_ICON_SIZE = 110` px |
| **Compact layout** | `nodesep` 0.9→0.35, `ranksep` 1.5→0.6, `arrowsize` 0.75→0.55 |
| **Flow rates on pumps** | Blue bold label (`#2563EB`) with `X.XX mL/min` below each pump; topology builder now always computes per-stream rate = total_Q / n_streams when individual rates are absent |
| **Clean reactor labels** | Labels now show only: `temperature · λ=wavelength · τ=residence time`. Verbose material/description strings removed. |

---

### 11.5 — Results Tabs Redesign

The Streamlit result view (`pages/flora_design_unified.py`) now has 8 tabs:

| Tab | Content |
|-----|---------|
| **Summary** | LLM-generated explanation, chemistry notes |
| **Engineering Design** | 9-step calculator + **per-reactor breakdown** (τ, V, ID, T per reactor for multi-step) |
| **Process Diagram** | Graphviz PFD with τ shown on each reactor node + unit operation list |
| **Chemistry Plan & Recipe** | Species, mechanism, sensitivities + step-by-step bench recipe |
| **Stream Assignments** | Per-pump contents, solvent, rationale |
| **Council Deliberation** | Round-by-round agent chain-of-thought, calculations, proposals, debate, and sanity checks |
| **Council Report** | Legacy agent messages grouped by agent |
| **Raw JSON** | Full result dict |

Note: **Conditions tab removed** — it showed flat single-reactor values that were misleading
for multi-reactor designs and duplicated Engineering Design.

---

### 11.6 — Council Deliberation Viewer

The Council Deliberation tab renders the full `DeliberationLog` directly — no additional LLM
call needed. Each agent's contribution is shown as an expandable card:

| Agent | Avatar | Error state |
|-------|--------|-------------|
| Dr. Kinetics | ⏱️ | Normal / ⚠️ WARNING / 🔄 REVISE / 💥 ERROR |
| Dr. Fluidics | 🌊 | Normal / ⚠️ WARNING / 🔄 REVISE / 💥 ERROR |
| Dr. Safety | 🛡️ | Normal / ⚠️ WARNING / 🔄 REVISE / 💥 ERROR |
| Dr. Chemistry | 🧪 | Normal / ⚠️ WARNING / 🔄 REVISE / 💥 ERROR |
| Dr. Process | 🏗️ | Normal / ⚠️ WARNING / 🔄 REVISE / 💥 ERROR |
| Chief Engineer | 👷 | Cross-agent sanity check, conflict resolution |

Cards with `had_error=True` show 💥 and `[ERROR — blocks convergence]` in the title and
are always expanded. Each card shows: chain-of-thought, values referenced (from calculator),
findings, structured proposals (`field → value` format), concerns, rules cited.

**Consensus output** shows ALL modified fields with ★ marker — not just τ/Q/V_R:
> Consensus reached after 2 rounds · τ = 12.5 min · Q = 0.5 mL/min · V_R = 6.25 mL · d = 0.75 mm ★
> (★ = modified by council: tubing_ID_mm)

If convergence was blocked by errors, the warning line shows the error count:
> Max rounds (2) reached — 1 agent error(s) prevented convergence · τ = 12.5 min ...

---

### 11.7 — Experimental Recipe Tab

`_render_recipe()` in `flora_design_unified.py` generates a 4-section step-by-step
experimental protocol for the bench chemist:

**Section A — Preparation:** Liquid vs gas streams handled separately.
- Liquid streams: weigh reagents, dissolve in solvent, deoxygenate if required.
- Gas streams: *connect cylinder via mass flow controller, set sccm, use gas-rated fittings and check valve* — never "dissolve gas in a vial".

**Section B — Flow System Assembly:** Tubing, reactor, LED (if photochem), bath temperature,
BPR (with context-aware reason: gas solubility vs. boiling point), mixer connections, syringe loading.

**Section C — Running:** Per-pump and per-MFC flow rates listed individually, priming sequence,
gas startup (stable slug flow formation), 3× residence time steady-state wait, collection.

**Section D — Shutdown & Workup:** Gas supply closed first, liquid pumps stopped, flush with
3× reactor volume, quench (if required), standard workup guidance.

A safety banner is shown when gas streams are present.

---

## 12. Limitations and Roadmap

### Current Limitations

1. **Corpus size** — Currently ~140 photochemistry papers. Accuracy scales with corpus quality and coverage.

2. **Chemistry scope** — Prompts and rules are optimised for photocatalytic flow chemistry. Extension to thermal, electrochemical, or enzymatic flow requires new rule sets.

3. **Kinetics assumptions** — The calculator assumes first-order kinetics when back-fitting from batch data. Higher-order reactions require Ea/A or explicit order input.

4. **ΔH_r and U estimates** — Heat-transfer calculations use estimated ΔH_r by reaction class and a fixed U = 300 W/m²·K. Experimental values give more accurate Da_th.

5. **Mechanism_type metadata** — Hard retrieval filters only activate after re-indexing. Records indexed before the mechanism_type addition fall back to soft filtering.

6. **Fundamentals quality** — An empty fundamentals knowledge base means no rule injection. Quality improves with each handbook added.

### Planned Improvements

| Priority | Feature | Impact |
|----------|---------|--------|
| 1 | Ground truth evaluation pipeline (10–15 held-out papers) | Enables quantitative accuracy measurement |
| 2 | Feedback loop weighting into retrieval | Improves over time from lab corrections |
| 3 | Arrhenius parameter input (A, Ea) for true temperature-dependent τ | Removes first-order assumption in calculator |
| 4 | Full optimization table extraction (not just optimal row) | Richer parameter landscape for BO |
| 5 | BO design space suggestion post-translation | Connects translation output to experimental planning |
| 6 | Automatic corpus growth via OpenAlex monitoring | Keeps knowledge current automatically |

---

## 13. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# One-time: extract research papers
python paper_knowledge_extractor.py pdfs/

# One-time: index papers into ChromaDB
# (also run this after adding new papers or after schema changes)
rm -rf flora_translate/data/chroma_db/*   # clear old index if re-indexing
python -m flora_translate.main index extraction_results/

# One-time (optional but recommended): ingest handbooks
# → Use the GUI Fundamentals page, or:
python -c "
from flora_fundamentals.handbook_reader import HandbookReader
reader = HandbookReader()
idx, rules = reader.read_handbook('/path/to/handbook.pdf', two_pass=True)
reader.save([idx], rules)
print(f'{len(rules)} rules extracted')
"

# Test the design calculator with hand-verification
python test_design_calculator.py

# Launch dashboard
streamlit run app.py
```

---

*Document maintained alongside the codebase. FLORA is under active development.*
*Last updated: April 2026 — Engineering overhaul (9-step calculator, calculation-driven ENGINE, diagram improvements, recipe tab, council deliberation).*
