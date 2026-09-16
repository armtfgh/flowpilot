# FlowPilot 2.0 - private scientific-design enhancement

This is an opt-in evolution of the current pipeline, not a replacement agent.
The legacy default and its benchmark evidence remain available. Source baseline:
`d6fd9a307d1d170a7c44e5df7ccd0f3ebf614adf`.

## Scope and acceptance tasks

| Task | Status | Acceptance evidence |
| --- | --- | --- |
| 0. Freeze legacy source and failing DPDTC run | Verified | `outputs/releases/flowpilot_legacy_20260907_d6fd9a3`; source SHA256 `3f981082d8b2f94c917531a2a1c032dda185c14b53fdfc26d9b9f2f9442c9ff8` |
| 1. Preserve stage-specific reagent introduction | Focused tests passed | 5 tests in `test_scientific_evidence.py`; saved DPDTC replay preserves benzylamine in stage 2 |
| 2. Separate evidence from kinetic assumptions | Focused tests passed | Same tests; unknown rate, yield and IF; class-mismatched analogy cannot supply kinetics |
| 3. Keep a 12-candidate scientific council | Unit tests and preliminary live run passed | 12 distinct complete designs, 48 domain reviews, then skeptic/chief; full live transcripts retained |
| 4. Preserve selected design through finalization | Verified on supplied baseline | All 12 physical candidates matched during full-pipeline replay; selected candidate preserved; final pressure-path check passed |
| 5. Expose private mode and scientific provenance in GUI | Verified on desktop/mobile | 24 browser checks passed; 2 additional SVG bounds/overlap checks passed; screenshots inspected and warning layout corrected |
| 6. Evaluate supplied DPDTC baseline | Completed, software evaluation only | `live_04`, `verified_04`, `evaluation_final`; 14 independent checks passed; no measured yield improvement claimed |

## Scientific boundaries

- The supplied protocol contains two 30-minute heating periods at 95 C. The
  user's separate report of a 20-30 minute manuscript flow result is an external
  comparison, not input to candidate ranking or a fitting target.
- There are no new wet-lab measurements in this task. Improvement means corrected
  provenance, stage semantics, numerical/hardware consistency and honest uncertainty,
  not established yield improvement or publication readiness.
- Unknown kinetics permit an explicitly identified exploratory screen, not a
  manufactured conversion prediction. Physical limits cannot be overruled by a model.
- Candidate diversity and reviewer coverage are required. Repeating a failing
  generation until a desired result appears is not an evaluation strategy.
- All changes, test failures, fixes and run outcomes will be appended here or
  linked from the validation output directory. Existing result JSONs stay unchanged.

## Change and evaluation log

### Initial audit

The recorded legacy failure has a total inlet-based time of 2.25 min (1.25 + 1.00),
at 80 C instead of the protocol's 95 C. A substring `pd` inside `DPDTC` selected
the cross-coupling default. An unrelated photochemical analogy supplied IF=60;
the calculator blended it with IF=15, giving a 1.6-minute anchor. The council's
short-time policy explored only about 2.0-2.7 min. A stage-1 restoration function
also copied the global reagent list, introducing benzylamine before its specified
stage. These are release-blocking scientific defects, not plotting errors.

### Preserved development attempts

- `outputs/flowpilot2/20260907_dpdtc/live_01`: source identity check rejected
  reversed acronym/full-name aliases. Fixed explicit alias matching in either
  order and sentence punctuation; added regression coverage. No identities are
  guessed from unrelated chemical names.
- `live_02`: fresh Opus 4.6 upstream / Sonnet 4.6 downstream completed a full
  12-candidate council, selected candidate 4 (33.5411 + 26.8328 min), preserved
  benzylamine addition in stage 2 and generated the icon topology. Initial 13
  representation/arithmetic checks passed. **Further inspection found its
  8-bar BPR at the SF-10 pump's 8-bar maximum, without reactor pressure-drop
  headroom. This preliminary result is NOT the recommended final design.**
  `evaluation_02_headroom_audit` records that failure without deleting the earlier
  evaluation. Candidate generation now tests BPR plus downstream coil pressure
  losses against pump and reactor ratings, and explores available BPR settings.
- `live_03`: adding honest unknown thermal-performance fields revealed a prompt
  renderer that formatted `None` as a float. Fixed the renderer and tested the
  actual generated prompt, not only the calculator object.
- `live_04`: full fresh Opus 4.6 / Sonnet 4.6 run completed in 416 s. All four
  domain reviewers covered all 12 candidates; skeptic and chief also responded.
  Selected candidate 4 at 2 bar, with the selected physical design preserved.
  Eighty-eight trial configurations were rejected during pressure-feasibility
  construction before the final 12 were reviewed.
- `verified_04`: the saved upstream, translation and six council responses from
  `live_04` were replayed through the finalized code. All 12 physical candidates
  were matched at every reviewer call (72 comparisons) before reusing responses.
  This is **not another independent model repeat**. It verifies final engineering
  annotations without generating or subsequently hiding a default conversion,
  reaction rate, Damkohler number or heat-release estimate.
- `evaluation_final`: all 14 independent representation/stoichiometry/geometry/
  pressure/inventory/provenance checks passed. No experimental yield was measured.
- `browser_01_stale_server`: UI interaction tests correctly blocked submissions
  after code changed while the test server was running. Restarting the test
  server resolved this; `browser_02.json` records 22 passing tests and 12 skipped
  tests requiring live-run fixtures. No deployment safeguard was disabled.

### Current implementation boundaries

- The private preview was validated for the supplied explicit, two-stage thermal
  liquid sequence. Other chemistries require separate validation. Gas, heterogeneous and photochemical design-space policies
  have not been validated here and are not silently routed through the new policy.
- The 12 screening patterns bracket source batch holds using predefined stage
  multipliers. They are experimental choices, **not predicted kinetic optima**.
  Discrete reactor assignments and common stoichiometric flow scaling are solved
  before council review. A narrower inventory may not support 12 distinct points;
  it raises a diagnostic instead of padding the pool with duplicate candidates.
- Four domain agents must each review all 12 IDs. A negative preference is not
  a hard veto; explicit hard violations and the skeptic's hard vetoes exclude
  candidates. The chief selects an eligible existing ID, not rewritten numbers.
- Final selection preservation and pressure-path checks are deterministic.
  Unknown rate constants, conversion, yield, reaction heat and thermal safety
  are not replaced with a default target or a class intensification factor.
- Existing implicit-accessory inventory behavior remains: a compatible generic
  T-mixer can be a **verification-required assumption**, not confirmed KHU stock.
  The GUI warns about these items. No laboratory safety certification is implied.
- Thermal UA and hydraulic loss remain engineering estimates. Minor pressure
  losses, solvent/feed solubility, precipitation and temperature control need
  experimental verification. The saved KHU PFA maximum is 80 C; it does not
  authorize the source protocol's 95 C operation.
- Legacy scalar experiment-loop updates are disabled for the scientific preview;
  measured feedback requires a new intake and council review. Stage-resolved
  kinetic learning remains future work, not a capability claimed by this release.
- OpenAI embedding credentials were absent in these runs. Existing retrieval
  used its documented lexical fallback; low-match analogies were not allowed to
  establish kinetics. Anthropic upstream/council requests returned HTTP 200.

### Verification locations

- Legacy archive: `outputs/releases/flowpilot_legacy_20260907_d6fd9a3`.
- All baseline checks: `outputs/flowpilot2/20260907_dpdtc`.
- Upstream/translation snapshots: `outputs/scientific_pipeline`.
- Full candidate matrices, prompts, raw reviewer responses and selection:
  `outputs/scientific_council`.
- Re-run scripts: `scripts/run_flowpilot2_baseline.py` and
  `scripts/evaluate_flowpilot2_baseline.py`.

## Final DPDTC screening design

This is an experiment to evaluate, not a claim of high amide yield. It is not a
replication of the reported 20-30-minute literature flow process.

| Parameter | Stage 1 | Stage 2 |
| --- | --- | --- |
| Transformation | Acid activation / thioester formation | Benzylamine addition / amidation |
| Available reactor | KHU PFA 5 mL, 0.03-inch ID | KHU PFA 5 mL, unit 1 |
| ID (mm) | 0.762 | 1.016 |
| Temperature (C) | 80 | 80 |
| Cumulative liquid flow (mL/min) | 0.149071 | 0.186339 |
| Residence time (min) | 33.5411 | 26.8328 |
| BPR (bar gauge, shared downstream) | 2 | 2 |

Total reaction-zone residence time: **60.3739 min**. Stream A is 0.5 M substrate
at 0.149071 mL/min; stream B is nominally 2.1 M benzylamine at 0.037268 mL/min,
introduced only before Stage 2. Independently recomputed benzylamine/acid ratio:
1.050007, consistent with 1.05 equivalents within flow-setpoint rounding.
With a 0.5 M acid-feed basis, nominal DPDTC and DMAP concentrations are **0.525 M**
and **0.050 M**, respectively. Stock preparation and solubility require confirmation.

The chief's free-text assessment remains model output, not a verified procedure.
For example, it loosely refers to DPDTC as 0.5 M; the stoichiometric value is
0.525 M. It also suggests a 34-bar mixer rating; this is not a derived minimum
for the selected 2-bar system. Do not interpret these comments as authoritative
inventory specifications. Component quantities, physical setpoints and source
facts take precedence; a chemist must review any proposed experiment.

Calculated upstream pump pressure requirement is about 2.0185 bar (BPR plus
coil losses), below its 8-bar rating; the second pump path is about 2.0034 bar,
below its 42-bar rating. These estimates exclude uncharacterized minor losses.
The generic T-mixer is still **assumed, not confirmed equipment**. Verify it,
temperature control, feed/byproduct solubility, phase stability and an appropriate
sampling/operating procedure before laboratory execution.

## Evaluation conclusion

The software improvements demonstrated here are: source-linked stage ordering,
absence of forced IF or invented kinetics, complete 12-candidate review, joint
inventory/flow/pressure checks, preservation of the selected design, and matching
GUI tables/topology. **Longer reaction time by itself is not a success criterion.**
No statistical superiority, wet-lab yield improvement, general chemistry coverage,
or publication readiness is established by this single-case development check.

Verification: 376 Python tests passed (`regression_06.xml`), production frontend
build passed, 24 desktop/mobile browser tests passed (10 unrelated live-fixture
tests skipped), and both SVG caption bounds/overlap checks passed. Screenshots
are under `browser_final_layout`. Direct screenshot inspection caught a warning
text layout defect that assertions alone missed; it was fixed and given a
width/height regression check.

## Running the private preview

Open `http://localhost:8513` (or `http://10.13.24.95:8513` from your laboratory
network while this server is running). Choose **Design policy: FlowPilot 2.0 -
private scientific preview**. The default direct API remains legacy.

The verified saved result is available at:
`http://localhost:8513/?run=20260907_162137_scientific_verified`.

To restart on a free port:

```bash
FLOWPILOT_PORT=8513 ./flowpilot_webapp/run_dev.sh
```

Use another free port if 8513 is occupied; do not run duplicate servers on the
same port. The old source archive is preserved separately, without API keys or
ignored external retrieval-service state.
