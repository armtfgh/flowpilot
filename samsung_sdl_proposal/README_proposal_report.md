# Self-Driving Laboratory for High-k Thin Films — Proposal Report

*Companion document to `SDL_HighK_Proposal.pptx` — July 2026, confidential draft.*

## 1. The problem

High-k dielectrics (HfO₂/ZrO₂ family: HZO, doped ZrO₂, nanolaminates) are developed by
tuning a large parameter space: composition and doping (set via ALD supercycle ratios),
deposition temperature, plasma conditions, pulse/purge timing, film thickness, and
post-deposition anneal (temperature, time, ambient).

The bottleneck is **not deposition — it is characterization**:

| Tier | Measurements | Cost | What it tells you |
|---|---|---|---|
| Fast / structural | Spectroscopic ellipsometry, XRR, GIXRD | minutes, non-destructive, automatable | thickness, density, roughness, crystalline phase |
| Slow / electrical | C–V and I–V on capacitor structures (after anneal + electrode deposition) | hours–days, delicate, human skill | **k, leakage, breakdown — the properties that actually matter** |

Blind (grid/random) sampling spends most of the expensive electrical tier on samples
that carry no new information about the process window.

## 2. The idea (what makes this proposal different)

Most self-driving labs (SDLs) are framed as *optimizers*: autonomously find the single
best material. This proposal reframes the SDL as an **efficient data engine**:

> Learn the mapping **(process params, anneal, structural fingerprint) → (k, leakage)** —
> and, crucially, learn **where that mapping is valid** — using the fewest possible
> expensive electrical measurements.

Two deliverables fall out of this framing:

1. **Virtual metrology.** Once the map is trained, a minutes-long optical/structural scan
   predicts electrical performance with quantified confidence. Electrical probing becomes
   the exception, not the routine. (Fabs already do this for mature processes; extending
   it to *new* high-k materials in R&D is the gap.)
2. **Validity mapping.** The model reports where fast measurements reliably predict
   electrical behavior, and flags regions where they cannot — because hidden variables
   (phase mixtures, grain-boundary leakage paths, interfacial SiOₓ, trap states) dominate.
   Those flagged regions are exactly where the interesting physics and the reliability
   risks live, and they are routed to deep characterization (TEM/XPS).

**Important framing note:** avoid saying "correlation zones / non-correlation zones."
Physics doesn't switch off. The rigorous statement is: fast observables are sometimes
*sufficient* to predict electrical outcomes and sometimes not. The algorithm must
distinguish **epistemic uncertainty** (model needs more samples → keep sampling) from
**aleatoric / hidden-variable uncertainty** (more samples won't help → flag for deep
analysis). A heteroscedastic model whose learned noise floor is itself a deliverable is
the methodological novelty.

## 3. Why ALD is the right platform

- **The run is already digital.** An ALD recipe is a parameter file (pulse/purge times,
  temperature, plasma power, cycle count). Optimizer → recipe file → tool. No synthesis
  robotics needed — unlike spin-coating SDLs, which had to build their automation.
- **Composition is software.** Hf:Zr ratio and dopant level are set by supercycle cycle
  ratios — an integer in the recipe, not a new solution or sputter target.
- **In-situ metrology for free.** In-chamber ellipsometry/QCM give growth-per-cycle and
  film data every run without unloading — a fast, fully autonomous inner loop.
- **Precedent.** Autonomous ALD *process* tuning with in-situ feedback has been
  demonstrated (Argonne National Laboratory, Yanguas-Gil group — verify exact citation
  before submission). This proposal extends autonomy to film *properties* (k, leakage),
  which no published SDL has closed the loop on.
- **Honest constraint.** One run = one composition (uniformity is ALD's virtue — no
  combinatorial gradient libraries). Compensated by sample-efficient algorithms and rich
  in-situ data, not brute-force throughput.

## 4. Architecture: two nested loops around one decision engine

```
INNER LOOP (fully autonomous, ~minutes)
  PLAN (active learning) → DEPOSIT (ALD) → FAST METROLOGY (SE·XRR·GIXRD) → back to PLAN
                                                    |
                                            ESCALATE?  (info gain vs cost gate)
                                                    ↓
OUTER LOOP (algorithm-selected samples only, human-in-the-loop)
  ANNEAL (RTP) → ELECTRICAL TEST (C–V / I–V, pre-patterned substrates, human-executed)
       → MODEL UPDATE → back to PLAN

DATA BACKBONE: full provenance per sample (recipe → tool logs → structural fingerprint
→ anneal → electrical result), with drift-control reference runs scheduled automatically.
```

The human-in-the-loop electrical tier is a deliberate design choice, not a compromise:
probe automation is where thin-film SDLs historically fail. Pre-patterned test substrates
(Chipmetrics-style) or a mercury probe keep the human step fast and reproducible.

## 5. The decision engine

- **Model:** probabilistic map (multi-fidelity Gaussian-process family) from structural
  fingerprint + anneal parameters to k and leakage, with calibrated uncertainty. Note:
  the fast data are not a "low-fidelity version of k" — they are *intermediate features*,
  so the model is feature-based, not classic co-kriging of one output at two fidelities.
- **Acquisition:** this is **active learning / Bayesian experimental design, not MFBO** —
  there is no optimum to find, there is a map to learn. The policy chooses the next
  recipe AND whether a sample earns escalation to electrical testing, maximizing
  information gain per unit cost (the human's time is in the cost model).
- **Drift-aware by design:** scheduled reference depositions and replicate measurements
  separate real materials physics from tool drift and metrology noise. Without this, an
  apparent "unpredictable region" may just be the lab's own reproducibility limit — the
  single biggest trap in this project.
- **Validated in silico first (Phase 0):** build a synthetic HfO₂/HZO ground-truth
  landscape from literature (thickness–anneal–phase–leakage behavior, including a
  hidden-variable region) and show the acquisition policy reaches a fixed map accuracy
  with ≥3× fewer electrical tests than grid/random sampling (placeholder target — refine).
  The same benchmark becomes the acceptance test for the physical system.

## 6. Phased roadmap

| Phase | When | Scope | Standalone deliverable |
|---|---|---|---|
| 0 — Foundations | months 0–6 | data infrastructure + provenance schema; retrospective modeling on existing ALD data; in-silico algorithm benchmark | validated decision engine + data backbone |
| 1 — Human-in-the-loop SDL | months 6–18 | automated recipes + fast structural loop; algorithm-selected, human-executed electrical tests; one material system | virtual-metrology model v1 + mapped process window |
| 2 — Closing the loop | months 18+ | automated sample handling + anneal integration; expanded palette (dopants, laminates, electrodes); transfer validation on device-representative stacks | autonomous platform + reusable materials dataset |

De-risked by design: if Phase 2 automation slips, Phases 0–1 have already delivered the
dataset, the models, and the process windows.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tool drift & metrology noise mimic physics | scheduled reference runs + replicates; explicit noise terms in the model |
| Probe automation is fragile | human-in-the-loop tier + pre-patterned test substrates from day one |
| Lab optimum ≠ device stack (planar test structure vs real 3D device) | Phase 2 transfer validation on device-representative structures |
| Deposition-temperature changes are slow (30–60 min stabilization) | cost-aware scheduling batches experiments by thermal budget |
| Precursor changes are manual | fix the chemistry palette (e.g., TEMA-Hf, TEMA-Zr, one dopant) at proposal time |

## 8. One-paragraph pitch

> We propose a phased self-driving laboratory that treats expensive electrical
> characterization as a scarce resource allocated by an information-theoretic policy.
> The system autonomously grows and structurally fingerprints ALD high-k films,
> escalates only maximally informative samples to human-executed electrical testing,
> and learns not just a structure→property model but the boundaries of its own
> validity — delivering (1) a virtual-metrology model for new high-k materials,
> (2) mapped safe operating windows, and (3) automatically flagged regions where
> hidden variables control the physics, i.e., where the interesting science is.

## 9. Before submission — verify these

- Exact citation for the Argonne autonomous-ALD work (Yanguas-Gil group) and any claimed
  speed-up numbers.
- Which ALD tool/vendor Samsung has in mind (API access determines Phase 1 scope).
- Material system: DRAM capacitor dielectric (ZrO₂ family) vs ferroelectric HZO vs
  gate stack — changes the objective function and metrology details.
- The "≥3× fewer electrical tests" benchmark target — set after the in-silico study design.

## In-silico demonstration (done)

The decision engine has been validated on a synthetic, physics-inspired
landscape (`benchmark/`): 4 escalation policies × 6 campaigns × 150 electrical
tests. Ours (joint info-gain, epistemic-aware, with replicate-anchored noise
estimation and an ε-exploration safeguard) achieved the best map accuracy
(RMSE 0.56 vs 0.62–0.76), the best hidden-zone detection (AUC 0.74), and
wasted half the budget of any baseline in the unlearnable zone (16% vs
30–35%). It reaches the baselines' final accuracy with ~2× fewer tests.
Full details, honest limitations, and the methodological findings (replicates
are mandatory; noise-map lock-in needs an exploration safeguard):
`benchmark/RESULTS.md`. Deck slide 7 carries the composite figure.

## Files

- `SDL_HighK_Proposal.pptx` — the 9-slide deck (slide 7 = in-silico demo)
- `make_deck.py` — regenerates the deck (`python3 make_deck.py`)
- `README_proposal_report.md` — this report
- `benchmark/` — synthetic landscape (`synth.py`), heteroscedastic GP
  (`gp.py`), policy benchmark (`run_benchmark.py`), figures
  (`make_figures.py`), results (`RESULTS.md`, `results.json`, `fig*.png`)
