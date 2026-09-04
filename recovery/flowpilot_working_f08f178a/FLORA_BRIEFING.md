# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing Document

**Version:** 1.5 | April 2026
**Status:** Active Development
**Codebase:** ~17,500+ lines across 65+ files
**Last major update:** April 2026 — Engineering Overhaul (9-step design calculator, calculation-driven ENGINE council, diagram improvements, recipe tab, council deliberation)

---

## 1. Executive Summary

FLORA is an AI-powered platform that automates the translation of batch chemistry protocols into validated continuous flow process designs. It combines **literature-grounded retrieval-augmented generation (RAG)**, **multi-agent engineering validation**, **deep chemistry reasoning**, and **handbook-level foundational knowledge** to produce proposals that a process chemist can take directly to the lab.

The system addresses a core bottleneck in flow chemistry adoption: designing a flow process from a batch protocol requires deep domain expertise in reaction engineering, fluid dynamics, photochemistry, and hardware compatibility — knowledge that is scattered across hundreds of journal articles and textbooks. FLORA consolidates this knowledge into a queryable, validated pipeline.

### What FLORA produces

Given a batch protocol like:

> *"Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative radical addition of N-Boc-proline to methyl vinyl ketone, K2HPO4, DMF, 0.1M, 25°C, 450nm LED, N2, 24h, 72% yield"*

FLORA returns:

- A **validated flow design** with named chemical streams, reactor specifications, and operating conditions
- A **process flow diagram** (SVG/PNG) showing actual chemicals in each stream
- An **engineering validation report** covering pressure drop, kinetics, safety, and hardware compatibility
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
│  │  ┌─────────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │  │ ENGINE      │  │ ChromaDB │  │ Lab Inventory          │  │
│  │  │ Council     │  │ Vector   │  │ (hardware specs)       │  │
│  │  │ (5 agents)  │  │ Store    │  └────────────────────────┘  │
│  │  └─────────────┘  └──────────┘                               │
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
[5] ENGINE VALIDATION COUNCIL (Layer 3)
    Five independent agents validate the proposal.
    Up to 3 revision rounds until convergence.
    │
    │  Chemistry Validator — wavelength matches catalyst absorption?
    │                        incompatible pairs in separate streams?
    │                        deoxygenation specified if O₂-sensitive?
    │
    │  Kinetics Agent     — residence time vs literature intensification
    │                        factor (within 0.3x–3x of median?)
    │
    │  Fluidics Agent     — Hagen-Poiseuille pressure drop < pump max?
    │                        Reynolds number in laminar regime?
    │
    │  Safety Critic      — tubing material compatible with solvent?
    │                        temperature within material limits?
    │                        BPR/light source available in lab inventory?
    │
    │  Process Architect  — builds ordered unit operations and P&ID
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
Chemistry Agent injection
    Every translation query automatically loads relevant rules
    and appends them to the Chemistry Agent's system prompt.
    Rules are filtered by category + keyword match to the
    specific reaction context (mechanism type, solvent,
    photocatalyst, O₂ sensitivity, temperature).
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

## 4. The ENGINE — Multi-Agent Engineering Validation

The ENGINE is the shared validation layer used by both Translate and Design. It runs a **deliberation council** of five specialised agents.

| Agent | What it checks | Method |
|-------|---------------|--------|
| **Chemistry Validator** | Wavelength ↔ photocatalyst match, incompatible reagent pairs in same stream, missing deoxygenation/quench, concentration sanity, light-sensitive feed lines | Rule-based against ChemistryPlan |
| **Kinetics Agent** | Residence time reasonableness via intensification factor (batch time / flow time) compared to literature median | Statistical comparison |
| **Fluidics Agent** | Pressure drop (Hagen-Poiseuille), Reynolds number, pump capacity | Physics calculation |
| **Safety Critic** | Tubing-solvent compatibility, temperature limits, BPR/light source/reactor availability in lab inventory | Lookup tables |
| **Process Architect** | Unit operation sequence, text-based P&ID | Construction |

**Convergence protocol — minimum 2 rounds always:**
- **Round 1 (validation):** All agents run. Hard violations are corrected. Calculations are used for precise validation (e.g. KineticsAgent uses computed Da and intensification factor directly).
- **Round 2 (refinement):** Even if Round 1 was clean, a second pass refines conditions. BPR is cross-checked against calculation results — if calculations say BPR is required but proposal has none, a REJECT is injected automatically.
- **Round 3 (if needed):** Only runs if Round 2 still has violations. Max 3 rounds total.

KineticsAgent now includes in every message: the computed intensification factor, the estimated RT from calculations, and the Damköhler number — making its feedback quantitatively grounded rather than rule-of-thumb only.

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
Pre-computed before the translation LLM runs via `DesignCalculator.run()`:

1. **Kinetics (multi-method):** Analogy-derived IF (primary when ≥ 2 analogies have batch+flow data), class-level IF (fallback), Arrhenius T-correction if T_flow ≠ T_batch. Reports τ as a range covering all methods.
2. **Reactor sizing:** V_R = τ×Q, L = 4V_R/(πd²) — all geometry derived from τ
3. **Fluid dynamics:** Re = ρvd/μ — auto-adjusts d upward if turbulent
4. **Pressure drop:** ΔP = 128μLQ/(πd⁴) via Hagen-Poiseuille — auto-adjusts d if > pump max
5. **Mass transfer:** t_mix = d²/(D·π²), Da = k·d²/D — flags mass-transfer limited regime
6. **Heat transfer:** Q_gen, Q_rem = U·A·ΔT_lm, thermal Da — flags exotherm risk
7. **BPR:** Antoine equation for vapor pressure + system ΔP + 0.5 bar margin; **gas-liquid systems always require BPR (≥ 5 bar) regardless of boiling point**
8. **Process metrics:** STY, productivity, intensification factor

All steps are internally consistent (τ=V/Q, L=4V/πd², Re=ρvd/μ verified). Results injected
into translation prompt as a concise engineering block.

**Layer 2 — Translation (receives ChemistryPlan + calculations + analogies)**
The translation LLM (Sonnet) receives three explicit reasoning steps:
1. **Analogy comparison:** For each analogy, must state: (a) what is chemically similar, (b) key difference, (c) how that difference adjusts the parameters.
2. **Calculation validation:** Must verify residence time is within calculated range, BPR set if calculation requires it, Re and Da are reasonable.
3. **Conditions justification:** Every numerical field must cite its source: analogy N, calculation, chemistry plan, or first principles.

**Layer 3 — Chemistry Validator (in ENGINE council)**
Cross-checks the final proposal against the ChemistryPlan:
- Does the wavelength match the photocatalyst?
- Are incompatible reagents actually in separate streams?
- Is deoxygenation specified for O₂-sensitive reactions?
- Is the quench reagent appropriate?

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
- **REVISE:** "add a liquid-liquid extraction" → appends `[REVISION: ...]` to original query → re-runs full pipeline.
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
          ▼                          │  system prompt
   ┌────────────────────┐            │
   │ ChromaDB           │            │
   │ flora_records      │            ▼
   │ flora_pairs        │    ┌───────────────────┐
   │ (+ mechanism_type, │    │  Chemistry Agent  │
   │   phase_regime     │    │  (Layer 1)        │
   │   metadata)        │    └────────┬──────────┘
   └──────────┬─────────┘             │
              │                        │
              └──────────┬─────────────┘
                         ▼
              ┌─────────────────────┐
              │ TRANSLATE / DESIGN  │
              │ pipeline            │
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
| LLM (reasoning) | Claude Opus 4.6 (chemistry analysis, council conversation) |
| LLM (translation) | Claude Sonnet 4.6 (translation, extraction, council summary) |
| LLM (cheap scan) | Claude Haiku 4.5 (figure classification, handbook relevance scan) |
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

**Gas-liquid detection** — `_is_gas_liquid()` inspects atmosphere, description, and stream
contents. If a reagent gas (O₂, H₂, CO₂, CO, etc.) is present, BPR is mandatory regardless
of boiling point, minimum 5 bar to maintain Henry's-law gas solubility. N₂/Ar used purely
as inert atmosphere are **not** counted as gas-liquid.

**Streamlit display** — `components/design_steps.py` renders each step with:
- Status badge (PASS / WARNING / FAIL / ADJUSTED / ESTIMATED)
- LaTeX equations with real substituted numbers
- Computed values as metric cards
- Warnings, adjustments, and assumptions

**Backward compatibility** — `DesignCalculations` exposes `estimated_rt_min`, `bpr_minimum_bar`,
`damkohler_number`, `damkohler_interpretation`, and `to_prompt_block()` so all existing
prompt-builder and ENGINE code works without changes.

---

### 11.2 — Calculation-Driven ENGINE Council

`flora_translate/engine/moderator.py` completely rewritten. The calculator is now the
**source of truth** for all physics. The council enforces this.

**Old behaviour:** agents ran independently, returned text-only warnings, revisions extracted
numbers from strings. Proposals were rubber-stamped unless blatantly wrong.

**New behaviour:**

```
For each round (max 3):
  1. Run DesignCalculator on the current proposal
  2. Physics sync:
       - If proposal τ differs from calculated τ by > 20% → REJECT + correct value
       - If V_R ≠ τ×Q → REJECT + enforce consistency
       - If d was adjusted (ΔP or Re) → REJECT + use calculated d
       - If BPR required but missing → REJECT + calculated value
  3. Run domain agents (all read calculator results):
       - KineticsAgent:   compares proposal τ to calculated τ, REJECT if > 50% off
       - FluidicsAgent:   reads step 4-5 results, reports pump/Re issues
       - SafetyCriticAgent: material compatibility, temperature limits, inventory
       - ChemistryValidator: stream assignments, wavelength, deoxygenation
  4. Apply all revisions → re-run calculator → verify consistency
  5. If no revisions in round ≥ 2 → converged
```

Every revision enforces τ×Q = V_R after application. No round ever leaves an inconsistent proposal.

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

The Streamlit result view (`pages/flora_design_unified.py`) now has 9 tabs:

| Tab | Content |
|-----|---------|
| **Summary** | LLM-generated explanation, chemistry notes |
| **Engineering Design** | Full 9-step calculator output with LaTeX equations |
| **Process Diagram** | Graphviz PFD + unit operation list |
| **Chemistry Plan & Recipe** | Species, mechanism, sensitivities + step-by-step bench recipe |
| **Stream Assignments** | Per-pump contents, solvent, rationale |
| **Conditions** | All flow parameters with reasoning |
| **Council Deliberation** | LLM-generated conversational meeting transcript |
| **Council Report** | Raw agent messages grouped by agent |
| **Raw JSON** | Full result dict |

---

### 11.6 — Council Deliberation Tab

`components/council_conversation.py` generates a genuine engineering review meeting transcript
using Claude Sonnet. Each agent has a persona:

| Agent | Persona |
|-------|---------|
| DesignCalculator | Physics Engine — cites equations and error percentages |
| KineticsAgent | Dr. Kinetics — residence times, intensification, literature comparison |
| FluidicsAgent | Dr. Fluidics — pressure drops, hardware limits, pragmatic |
| SafetyCriticAgent | Safety Officer — conservative, flags hazards |
| ChemistryValidator | Dr. Chemistry — mechanism, stream logic, sensitivity |
| ProcessArchitectAgent | Process Architect — synthesises to final design |

The conversation shows: initial proposal, disagreements, corrections being applied,
agents asking each other questions, and final consensus. Triggered on demand via a
"Generate Council Discussion" button. Cached in `st.session_state` to avoid re-generation.

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
