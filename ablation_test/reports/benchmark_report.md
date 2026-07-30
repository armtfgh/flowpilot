# FlowPilot Ablation Benchmark Report

Generated: 2026-07-28T10:11:45.369464+00:00

## Scope

- Frozen literature-derived cases: 12
- Adversarial engineering cases available: 8
- THQ is excluded by an automated content audit.
- Hidden source records are excluded from retrieval on a per-case basis.
- Machine-extracted literature records require verification against original papers before manuscript use.

## Execution

- Run summaries found: 178
- Completed cells: 164
- Status counts: `{"completed": 164, "failed": 14}`

## Execution Limits

- Literature retrieval was available during these runs.
- Completed GPT-4o cells: 105.
- Primary matched architecture matrix: 7 variants x 6 protocols = 42 completed cells with `gpt4o`.
- The primary architecture table uses one fixed model and the exact case intersection shared by every included variant.
- Full-pipeline portability contains one smoke case per available model and is not a publication-level model ranking.

## Primary Results

Architecture composite scores (fixed model: `gpt4o`):

| variant                 |     mean |       std |   count |
|:------------------------|---------:|----------:|--------:|
| no_council              | 0.972617 | 0.0365522 |       6 |
| no_retrieval            | 0.953967 | 0.0507048 |       6 |
| full                    | 0.938083 | 0.0588613 |       6 |
| no_engineering          | 0.924983 | 0.0604239 |       6 |
| general_one_shot        | 0.9004   | 0.0920844 |       6 |
| structured_single_agent | 0.8992   | 0.0544504 |       6 |
| no_inventory            | 0.805933 | 0.072167  |       6 |

**Important metric limitation:** the deterministic composite does not include formal schema validity or the benefit of council rejection/failure handling. A schema-invalid one-shot response can therefore receive a high composite score. These rankings are screening results, not evidence that a simpler architecture is scientifically superior.

### Candidate Quality and Assurance Score v2

| variant                 |   quality assurance |   quality std |   deployment readiness |   deployment-ready rate |   cases |
|:------------------------|--------------------:|--------------:|-----------------------:|------------------------:|--------:|
| full                    |            0.91125  |     0.0651681 |               0.69     |                0.166667 |       6 |
| no_council              |            0.87125  |     0.038365  |               0.87125  |                1        |       6 |
| no_inventory            |            0.8525   |     0.0595609 |               0.6      |                0        |       6 |
| no_retrieval            |            0.851667 |     0.0759221 |               0.77875  |                0.5      |       6 |
| no_engineering          |            0.747083 |     0.0480213 |               0.692917 |                0.666667 |       6 |
| general_one_shot        |            0.550417 |     0.0638635 |               0.35     |                0        |       6 |
| structured_single_agent |            0.538333 |     0.0545588 |               0.35     |                0        |       6 |

The v2 score uses output properties only: 65% direct design quality, 10% evidence provenance, 20% decision assurance, and 5% actionability/calibrated uncertainty. It does not award points from the architecture label or use the hidden literature reference.

- Full FlowPilot ranked first at 0.911; the strongest comparator was `no_council` at 0.871.
- Paired full-minus-`no_council` delta: 0.040 (95% case-bootstrap CI 0.017 to 0.065); Full won 5/6 cases.
- Full ranked first in 75.7% of 20,000 plausible weight sets. The winner is directionally robust but not independent of value judgments.
- Deployment readiness gives a different result: `no_council` had the highest mean readiness and Full had no completely ungated cases because gas, tubing, or `SCREEN_REQUIRED` gates remained.

**Prospective-validation warning:** v2 was designed after the current outputs were available. This is exploratory reanalysis, not preregistered confirmatory evidence. Freeze the included specification and validate it on new holdout protocols before making a superiority claim.

Full-pipeline model portability:

| bundle_name   |   mean |   std |   count |
|:--------------|-------:|------:|--------:|
| claude        | 0.7143 |   nan |       1 |
| gemma         | 0.5714 |   nan |       1 |
| qwen          | 0.5714 |   nan |       1 |

## Automated Findings

- Formal schema-valid rate by variant: `full` 100%, `general_one_shot` 0%, `no_council` 100%, `no_engineering` 100%, `no_inventory` 100%, `no_retrieval` 100%, `structured_single_agent` 0%.
- Geometry pass rate by variant: `full` 100%, `general_one_shot` 100%, `no_council` 100%, `no_engineering` 100%, `no_inventory` 100%, `no_retrieval` 100%, `structured_single_agent` 100%.
- Required gas-bookkeeping pass rate: `full` 50%, `general_one_shot` 0%, `no_council` 100%, `no_engineering` 0%, `no_inventory` 100%, `no_retrieval` 100%, `structured_single_agent` 0%.
- Unexpected gas streams in matched non-gas cases: 0.
- Matched architecture mean execution cost: `full` 43.2 calls / 165908 tokens; `general_one_shot` 1.0 calls / 1414 tokens; `no_council` 3.0 calls / 11650 tokens; `no_engineering` 3.2 calls / 11982 tokens; `no_inventory` 40.7 calls / 159503 tokens; `no_retrieval` 40.3 calls / 154596 tokens; `structured_single_agent` 1.0 calls / 1289 tokens.
- Full-pipeline cost and geometry: `claude`: 63 calls, 392490 tokens, geometry pass=False; `gemma`: 74 calls, 122994 tokens, geometry pass=False; `qwen`: 25 calls, 107546 tokens, geometry pass=False.
- These are automated screening findings. Expert scores remain pending.

## Interpretation Rules

- Architecture claims must use the fixed-model architecture arm only.
- Model claims must use the full-pipeline portability arm only.
- Surrogate or partial references are reported separately from primary reference-accuracy statistics.
- Keyword coverage is an automated screening metric, not expert validation.
- The deterministic composite excludes `schema_valid`; inspect schema validity and expert review separately.
- Positive ablated-minus-full deltas do not by themselves prove that removing a component improves scientific design quality.
- Quality Assurance Score v2 is exploratory on this dataset and must be frozen before prospective holdout validation.
- A high assurance score does not override a failed deployment gate.
- Failed and credential-blocked cells remain in the denominator and failure table.

## Artifacts

- `tables/all_runs.csv`: one row per discovered run.
- `tables/aggregate_architecture.csv`: architecture summary.
- `tables/aggregate_portability.csv`: model summary.
- `expert_scoring/expert_scoring_blinded.csv`: blinded review form.
- `figures/`: PNG and vector PDF versions of every figure.
- `tables/agent_call_events.csv`: call-level model/component proof with prompt and response hashes.
- `tables/architecture_module_matrix.csv`: explicit module presence/removal for all seven architecture variants.
- `tables/quality_score_v2_matched.csv`: matched quality and deployment summary.
- `tables/quality_score_v2_pairwise.csv`: paired Full-versus-ablation differences and bootstrap intervals.
- `tables/deployment_gate_rates_v2.csv`: architecture-level hard-gate frequencies.
- `reports/quality_score_v2_specification.json`: candidate metric definition to freeze for future holdout validation.
- `reports/quality_score_v2_methodology.md`: score rationale, equations, gates, and statistical plan.
- `reports/agent_trace_summary.md`: concise full-pipeline call trace.
- `runs/`: raw prompts, completions, snapshots, metrics, logs, and checksums.
