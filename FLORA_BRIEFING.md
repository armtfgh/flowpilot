# FLORA — Flow Literature Oriented Retrieval Agent

## Technical Briefing Document

**Version:** 1.2 | March 2026
**Status:** Active Development
**Codebase:** ~10,500 lines across 57+ files

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

**Convergence protocol:**
1. All agents run independently on the current proposal
2. Agents that flag `revision_required=True` provide a suggested fix
3. Moderator applies revisions programmatically
4. Repeat (max 3 rounds) until no revisions needed

---

## 5. Chemistry Reasoning — The Three-Layer Architecture

The key architectural innovation in FLORA is the separation of **chemistry reasoning** from **hardware translation**, further grounded by a **fundamentals knowledge layer**:

**Fundamentals Layer (before everything)**
Handbook rules are loaded and injected before any LLM reasoning begins:
- "FEP/PFA tubing required for photochemistry (transparent to visible light)"
- "BPR required when T > solvent bp − 20°C"
- "Residence time = reactor_volume / flow_rate (always verify consistency)"
- Hundreds of similar hard rules extracted from expert textbooks

**Layer 1 — ChemistryPlan (before retrieval)**
A dedicated Claude call that *only* thinks about chemistry, informed by the fundamentals:
- Identifies the reaction mechanism (radical, ionic, SET, HAT, energy transfer)
- Maps every species and its role
- Determines which reagents must be in separate streams and why
- Flags sensitivities (O₂, moisture, light, temperature)
- Generates retrieval keywords including mechanism type for smarter search

**Layer 2 — Translation (receives ChemistryPlan + analogies)**
The translation LLM is told: *"The ChemistryPlan is authoritative for chemistry. The analogies are authoritative for hardware."* Hard metadata filters (mechanism_type, phase_regime) ensure only mechanistically relevant analogies are retrieved.

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

## 7. Cost Optimisation

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
| LLM (reasoning) | Claude Sonnet 4 (translation, chemistry analysis, extraction) |
| LLM (cheap scan) | Claude Haiku 4.5 (figure classification, handbook relevance scan) |
| Embeddings | OpenAI text-embedding-3-small (1536 dimensions) |
| Vector database | ChromaDB (persistent, two collections: flora_records, flora_pairs) |
| PDF processing | PyMuPDF (text extraction + page rendering) |
| Physics calculations | Hagen-Poiseuille, Reynolds number (pure Python, no external lib) |
| Bayesian optimisation | scikit-learn GaussianProcessRegressor + scipy EI |
| Web framework | Streamlit |
| Diagram generation | Pure SVG (chemistry-aware node labels, no external diagram lib) |
| Data validation | Pydantic v2 |
| Literature search | pyalex (OpenAlex API) |

---

## 11. Limitations and Roadmap

### Current Limitations

1. **Corpus size** — Currently ~140 photochemistry papers. Accuracy scales directly with corpus quality and coverage. Reactions with no close analogy receive LOW confidence.

2. **Chemistry scope** — Prompts and rules are optimised for photocatalytic flow chemistry. Extension to thermal, electrochemical, or enzymatic flow requires new rule sets.

3. **No first-principles kinetics** — The Kinetics Agent uses statistical comparison to literature, not computational chemistry.

4. **Static diagrams** — Process flow diagrams are SVG images, not interactive.

5. **Mechanism_type metadata** — The new hard retrieval filters only activate after re-indexing. Records indexed before this change have empty mechanism_type and fall back to soft filtering.

6. **Fundamentals quality depends on handbooks ingested** — An empty fundamentals knowledge base means no rule injection. Quality improves with each handbook added.

### Planned Improvements

| Priority | Feature | Impact |
|----------|---------|--------|
| 1 | Ground truth evaluation pipeline (10–15 held-out papers) | Enables quantitative accuracy measurement |
| 2 | Feedback loop weighting into retrieval | Improves over time from lab corrections |
| 3 | Full optimization table extraction (not just optimal row) | Richer parameter landscape for BO |
| 4 | BO design space suggestion post-translation | Connects translation output to experimental planning |
| 5 | Automatic corpus growth via OpenAlex monitoring | Keeps knowledge current automatically |

---

## 12. How to Run

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

# Launch dashboard
streamlit run app.py
```

---

*Document maintained alongside the codebase. FLORA is under active development.*
