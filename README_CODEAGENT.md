# FlowPilot code-agent handoff

**Repository snapshot:** `modded_v2`, inspected 20 October 2026
**Purpose:** orient a new developer or coding agent to the active application, its code paths, research utilities, user interfaces, data boundaries and verification workflow.

This is a research codebase with several generations of tools, manuscript work and archived benchmark campaigns. The active batch-to-flow implementation is the `flora_translate` package. The newer browser/desktop workspace is in `flowpilot_webapp`; the root Streamlit application is an additional interface. Do not infer that every similarly named directory is part of the current production path.

## Start here

- Main batch-protocol design function: `flora_translate/main.py::translate`.
- Main structured data contracts: `flora_translate/schemas.py`.
- Main multi-agent design council: `flora_translate/engine/council_v4/`.
- Main browser/desktop app: `flowpilot_webapp/`.
- Main development briefing: `FLORA_BRIEFING.md`. It has useful architecture detail but predates several current finalization and scientific-mode changes; verify behavior against current source and tests. For example, current `main.py` explicitly keeps the design-space winner as a council seed and preserves the proposal geometry for deliberation.
- Private scientific-mode notes: `docs/flowpilot_2_development.md`. These describe an opt-in development path and a case-specific evaluation, not a general claim that every chemistry is validated.

The main operating principle is that language models interpret chemistry and assess design trade-offs, while deterministic code owns numerical closure, inventory binding, final validation and executable-artifact generation. Treat the final contract—not free-text agent reasoning or an intermediate proposal—as the user-facing source of truth.

## Architecture from input to result

```text
Researcher / GUI
  -> intake questions, objectives, operating limits and laboratory inventory
  -> flora_translate.main.translate(...)
       1. parse and normalize batch protocol
       2. create chemistry plan and reconcile it against source facts
       3. preflight required unit operations against available inventory
       4. retrieve and rank literature analogies
       5. calculate engineering anchors and design-space candidates
       6. generate a proposal and run ENGINE Council v4 review
       7. realize the selected proposal against actual inventory
       8. compile/check topology, disposition and final-design contract
       9. render run-local diagram and publish result artifacts
  -> final_design + result views + saved run record
```

In source, the main sequence is:

1. `flora_translate.main.translate` resolves either a raw protocol or a ready `DesignInputPackage`, then calls `parse_batch_input` and `analyze_batch_chemistry`.
2. `reconcile_chemistry_plan` anchors model-derived chemistry to protocol facts and hard constraints. If strict inventory assignment is enabled, `analyze_topology_requirements` checks required capabilities before expensive design calls; unresolved equipment is surfaced as a confirmation or infeasibility result.
3. `VectorRetriever.retrieve` builds a chemistry-plan-aware query, applies metadata filtering, falls back to softer retrieval when needed and reranks records by semantic and selected field agreement. `AnalogySelector` chooses the analogies passed downstream.
4. `DesignCalculator.run` calculates the engineering anchor. `DesignSpaceSearch.run` explores a deterministic set of feasible geometry/time candidates; the council may also receive feasible design-space points as seeds.
5. `TranslationPromptBuilder` and `TranslationLLM.generate` produce the initial structured proposal. That proposal is a candidate, not the final equipment-bound recipe.
6. `CouncilV4.run` coordinates the Designer, Chemistry, Kinetics, Fluidics and Safety reviews, Skeptic audit and Chief selection. Bounded edits are recalculated and checked rather than trusted as arbitrary numeric text.
7. `finalize_design` and `realize_executable_design` bind feeds, stages, pumps, reactor geometry, gas settings and pressure settings to the selected laboratory inventory. The pipeline validates stream/stage consistency and may return `BLOCK`, `SCREEN` or a review-required disposition.
8. `compile_inventory_topology` creates the inventory-assigned process graph. `render_topology_artifacts` saves topology JSON, SVG/PNG and a render manifest. A blocked candidate may have a diagnostic diagram, but must not be described as executable.
9. `build_final_design_contract` creates `result.final_design`. The frontend result views use this contract as the canonical numerical/design representation. A blocked final contract suppresses executable procedure artifacts.

`PipelineRuntimeOptions` in `flora_translate/pipeline_runtime.py` carries per-run settings such as model routes, design policy, objective, retrieval mode, excluded record IDs, candidate budget and review controls. Keep benchmark/test-specific overrides run-scoped; do not silently change global defaults to make a single run pass.

Two policy settings are easy to conflate. `design_policy` is `legacy` or opt-in `scientific_v2`; `FLOW_TRANSLATION_POLICY` is the flow-value policy (`evidence_first`, `intensify` or `neutral`). Scientific mode uses stage-resolved source evidence and a constrained 12-candidate council path. Its code and evaluation notes are in `flora_translate/engine/council_v4/scientific.py` and `docs/flowpilot_2_development.md`; that case-specific verification does not validate every chemistry class.

## Important code map

| Path | Responsibility |
|---|---|
| `flora_translate/schemas.py` | Pydantic contracts for protocols, intake, chemistry plans, inventory, proposals, topologies, results and executable artifacts. Schema changes often require updates in API/UI serialization and tests. |
| `flora_translate/intake_agent.py` | Standardized intake questions, conditional questions, answer merging, package readiness and conversion from intake to protocol context. |
| `flora_translate/protocol_input.py`, `input_parser.py`, `batch_normalization.py` | Normalize raw protocol input and extract evidence-backed batch quantities. |
| `flora_translate/lightweight_upstream.py`, `chemistry_agent.py` | Lightweight/local and full chemistry-planning routes; produce structured `ChemistryPlan` data. |
| `flora_translate/chemistry_contract.py`, `protocol_identity.py`, `component_identity.py`, `source_component_aliases.py` | Reconcile model interpretation and component identity against the source protocol; protect chemical identity, roles, stages and stoichiometry. |
| `flora_translate/retriever.py`, `vector_store.py`, `embedding_engine.py`, `analogy_selector.py` | RAG corpus access, embedding/lexical search, metadata and field-aware ranking, and final analogy selection. |
| `flora_translate/design_calculator.py`, `residence_time_basis.py`, `gas_settings.py`, `pump_settings.py` | Deterministic engineering calculations and explicit flow/residence-time bases, gas setpoints and supported pump settings. |
| `flora_translate/engine/design_space.py`, `engine/sampling.py` | Candidate-grid enumeration, physical calculations, feasibility checks and candidate metrics shared with the council. |
| `flora_translate/prompt_builder.py`, `translation_llm.py`, `conversation_agent.py`, `revision_agent.py` | Build translation prompts, parse model output, route user chat intent and apply targeted revision requests. |
| `flora_translate/engine/council_v4/chief.py` | Council orchestration, weighted selection, revision coordination and final selection. `scoring.py`, `designer.py`, `skeptic.py`, `review_contract.py`, `execution_config.py`, `physics_tools.py` and `backflow.py` implement the bounded review parts. |
| `flora_translate/design_realizer.py`, `final_design_validator.py`, `design_disposition.py`, `final_design_contract.py`, `executable_artifacts.py` | Post-council hardware realization, checks, disposition and the canonical final-design contract/artifacts. |
| `flora_translate/topology_compiler.py`, `topology_semantics.py`, `topology_preflight.py`, `multistage_inventory.py` | Convert process requirements into an inventory-bound, stage-aware topology and assess required capabilities. |
| `flora_translate/diagram_artifacts.py`, `flora_design/visualizer/flowsheet_builder.py` | Render topology to run-local SVG/PNG, with a deterministic fallback and artifact provenance. |
| `flora_translate/experiment_loop.py` | Interpret measured experimental outcomes and propose evidence-calibrated next conditions. Review stage/time-basis handling before extending it. |
| `flora_translate/gui_autosave.py`, `output_formatter.py`, `result_reporting.py` | Persist and format run results for interfaces and research workflows. |
| `flora_translate/inventory_profiles.py`, `inventory_resolution.py`, `inventory_constraints.py`, `inventory_allocator.py` | Versioned laboratory profiles, ingestion/validation, hardware matching and constraint enforcement. |

### Contracts and invariants to preserve

- `result.final_design` is canonical. `result.proposal`, council text and early calculations can be preliminary.
- Keep protocol facts, user preferences, model hypotheses, retrieved precedent and deterministic calculations distinguishable in the data model and UI.
- Do not invent equipment when a strict inventory is active. Ask for confirmation or return an explicit blocked/diagnostic state.
- Preserve stage-specific additions and feed assignments in telescoped processes. A global reagent list must not silently move a later-stage reagent upstream.
- Gas feeds need explicit identity, phase, flow basis, composition and residence-time basis. The code has separate standard/inlet and in-channel values; do not collapse them into one unlabeled flow or time.
- The compiled diagram, final proposal, instrument manifest and operating procedure must describe the same design. Update contract tests whenever one representation changes.
- LLM review is not experimental validation, safety certification or a yield guarantee.

## User interfaces

### React + FastAPI workspace — preferred interface

`flowpilot_webapp/` is the HILO-inspired browser and desktop shell over the shared Python pipeline. The frontend does not implement separate chemistry logic.

- `frontend/src/main.tsx`: workspace state, navigation, job submission/polling and main screens (`design`, `inventory`, `runs`, `system`).
- `frontend/src/result_views.tsx`: result tabs and views for summary, topology, engineering, responses, recipe, inventory, council and JSON.
- `frontend/src/inventory_workspace.tsx`: inventory/profile editor and import/review flow.
- `backend/app.py`: FastAPI routes for runtime/model health, intake, inventory review/import/extraction/save, design jobs, refinement, saved runs and artifacts.
- `backend/jobs.py`: serialized background design work, progress logs, request/result snapshots and per-job persistence under `outputs/webapp_jobs/`.
- `backend/deployment.py`: running-build identity and stale frontend/backend protection.
- `backend/desktop.py`, `launcher.py`, `FlowPilot.spec`, `build_windows.ps1`: local desktop shell and Windows packaging.

Run locally from the repository root:

```bash
./flowpilot_webapp/run_dev.sh
```

The script prepares `.venv-flowpilot` from `flowpilot_webapp/backend/requirements.txt`, installs frontend dependencies with `npm ci` when needed, builds the client and starts Uvicorn on port 8510 (override with `FLOWPILOT_PORT`). Open `http://localhost:8510`. `http://localhost:8510/?demo=1` shows a UI-only sample and does not call an LLM. Health and model availability are exposed at `/api/health` and `/api/models`.

The FastAPI backend loads the root `.env` when `python-dotenv` is installed; the Streamlit process must receive credentials through its launch environment or an approved server-side configuration. The frontend does not hold FlowPilot provider secrets. Current cloud routes use `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. Semantic RAG embeddings use OpenAI `text-embedding-3-small`; if embeddings fail, retrieval can fall back to lexical search. Model routes are listed in `flora_translate/model_catalog.py`; local OpenAI-compatible endpoints can be supplied with `FLOWPILOT_QWEN36_BASE_URL` and `FLOWPILOT_QWEN38_BASE_URL`. The older general Ollama route uses `OLLAMA_BASE_URL` in `flora_translate/config.py`. Never commit `.env` or provider tokens.

The backend currently enables permissive CORS and the dev launcher binds to `0.0.0.0`. It is intended for local/research use; public deployment requires authentication, origin restrictions and deployment-specific secret handling.

### Root Streamlit dashboard — additional/older interface

`app.py` routes through `components/sidebar.py` to the `pages/` modules. `pages/flora_design_unified.py` is the primary FlowPilot design page in this interface. Other sidebar areas are inventory management, diagnostics, optimization, benchmark figure studio, fundamentals extraction, literature scouting, knowledge extraction and corpus management. Most design views call the same `flora_translate.main.translate` pipeline. Start with `streamlit run app.py` in an environment that includes Streamlit and the backend dependencies.

### Other interface and design packages

- `flora_design/` contains an older goal-to-topology flow (`flora_design.main.design`) with rule-based unit-op selection, RAG topology generation and Council v3. It is useful legacy context, but it is not the main batch-protocol pipeline and does not use Council v4.
- `hilo_webapp/` is a separate HILO/Ask-Tell optimization prototype with its own API and frontend. Do not treat it as FlowPilot’s production webapp; it uses a separate data path and service layer.

## Supporting and research code

| Directory/files | What they are for |
|---|---|
| `flora_fundamentals/` | PDF handbook reading and structured rule extraction/storage. |
| `flora_scout/` | Literature-search helper exposed in the Streamlit interface. |
| `flora_prism/` | GUI wrapper around `paper_knowledge_extractor.py` for extracting structured chemistry/process records from papers. |
| `flora_optimize/` | Gaussian-process expected-improvement helper; separate from the design-generation council. |
| `paper_miner.py`, `downloader.py`, `processing.py`, `paper_knowledge_extractor.py`, `doi_merger.py` | Research utilities for literature search/download, extraction, normalization and knowledge-base preparation. They are not called on every design run. |
| `flora_translate/data/` | Default inventory and runtime corpus assets. `records/`, `chroma_db/` and user inventory profiles are ignored/private data and may be absent from a Git checkout. |
| `ablation_test/` | Benchmark protocols, configurations, reusable benchmark source, test suite and campaign scripts. Some scripts make paid provider calls; inspect the script and target run directory before running. |
| `benchmark/` | Earlier benchmark framework and historical experiment helpers; check dates and manifests before using results as current evidence. |
| `ablation_results/` | Research outputs, tables, figures and saved benchmark results; this is not runtime code. |
| `manuscript/`, `figures_paper/`, `figures/` | Manuscript versions, submissions, figures and editorial work. These are research records, not application source; preserve originals and do not “clean up” without explicit direction. |
| `new direction/`, `inventory_khu/` | Separate KHU sulfoxidation study materials and inventory/source-review files. Keep this study data distinct from the core app and other experiments. |
| `case_papers_new/`, `dois/`, `archive dois/`, `extraction_results/`, `data/` | Literature inputs, DOI records and extracted knowledge. Some are ignored or local-only. |
| `scripts/`, `visualization/` | Operational scripts, research/benchmark runners, audits and plotting code. Inspect each script’s arguments and output path; many are campaign-specific. |
| `output/`, `outputs/`, `deliverables/`, `tmp/`, `recovery/`, `.local_artifacts_backup_*/` | Generated artifacts, intermediate results, recovery snapshots and backups. Not authoritative source unless a README/manifest identifies a specific frozen artifact. Some paths are symlinks to ignored archives. |

The `.gitignore` deliberately omits the paper corpus, Chroma index, user inventory profiles, generated jobs, benchmark data and many manuscript exports. A collaborator receiving only Git source may therefore need a separately authorized corpus, inventory profile and other required runtime inputs. Do not recreate or substitute scientific data silently.

## Development and verification

Python core and API tests:

```bash
python -m pytest flora_translate/tests flowpilot_webapp/tests
```

Frontend typecheck/build:

```bash
npm --prefix flowpilot_webapp/frontend ci
npm --prefix flowpilot_webapp/frontend run build
```

Browser end-to-end tests use Playwright and the script in `flowpilot_webapp/frontend/package.json`:

```bash
npm --prefix flowpilot_webapp/frontend run test:e2e
```

Run only relevant tests while iterating, then run the affected Python suite and production frontend build. Browser tests require Playwright browser binaries and the test server setup in `playwright.config.ts`. Benchmark scripts are not substitutes for unit tests and may make network calls or incur model charges.

When changing a field that crosses the pipeline, follow it end to end: schema -> parser/agent -> deterministic realization -> final contract -> API serialization -> frontend view -> persisted run artifact. Add or update a focused regression test at the boundary where the defect occurred. For scientific changes, also retain provenance for model route, runtime options, inventory profile, corpus/retrieval state and source facts.

## Instructions for incoming code agents

1. Begin by checking `git status`, the active branch and the exact target behavior. Read the relevant module and its tests before editing; use this guide as orientation, not as a substitute for source inspection.
2. Keep changes in the active package and preserve a focused diff. Do not edit saved benchmark outcomes, historical runs, manuscript versions, raw KHU inputs or recovery archives as a side effect of a code task.
3. Do not make live model/API calls or launch a paid benchmark unless the project owner explicitly asks for that run. Prefer deterministic unit/integration tests and saved-run replays first.
4. When a change touches a scientific decision or executable procedure, report the evidence, test coverage and remaining assumptions. Do not present a passing software test as laboratory validation.
5. Keep credentials server-side and out of prompts, source control, screenshots, reports and handoff documents.

## Handoff cautions

- `flora_translate/` is still named with the original FLORA project namespace. This is expected; user-facing branding is FlowPilot. Do not rename the package without updating imports, packaging and tests.
- `flora_translate/main.py` is the current batch-to-flow entry point. Root `main.py` is an old one-off literature-query/notebook-export file, not the application launcher; review it before execution or transfer because it contains an inline third-party credential. Do not copy that credential into documentation or new code; rotate it if it is still active.
- There is no root Python `pyproject.toml` or single pinned requirements file. The maintained webapp backend requirements and frontend lockfile are the clearest install references; confirm the collaborator’s Python version and test environment before promising a reproducible deployment.
- License status remains undecided. Confirm licensing with the project owner before external redistribution or incorporation into another package.
- Preserve archived benchmark/manuscript artifacts. They are evidence of prior work, but current scientific or software claims must be checked against the matching source commit, run manifest and raw records.

