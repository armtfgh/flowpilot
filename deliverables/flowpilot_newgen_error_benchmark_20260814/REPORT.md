# FlowPilot NewGen Error-First Benchmark

## Purpose

This report replaces the weighted quality score as the primary comparison. Each output is audited with the same 20 frozen questions. Python deterministically recalculates numerical closure and checks structured protocol, topology, safety, and inventory evidence. A failed applicable criterion counts as one error; `NOT_APPLICABLE` is excluded.

## Headline Results

| Model | Architecture | Errors | Applicable checks | Error rate | Critical-error-free runs |
|---|---:|---:|---:|---:|---:|
| Qwen 27B | One-shot | 8 | 53 | 15.09% | 1/3 |
| Qwen 27B | FlowPilot | 0 | 53 | 0.00% | 3/3 |
| GPT-5.4 | One-shot | 4 | 53 | 7.55% | 1/3 |
| GPT-5.4 | FlowPilot | 0 | 53 | 0.00% | 3/3 |

Qwen one-shot produced 8 errors; Qwen FlowPilot produced 0. GPT-5.4 one-shot produced 4 errors; GPT-5.4 FlowPilot produced 0.

FlowPilot was error-free under this frozen deterministic rubric in all 6 model-case runs. The one-shot outputs were also error-free for CuAAC, so the result is not that one-shot always fails. The architecture difference appeared in the gas-liquid-solid and multistage cases.

## Observed Errors

### Qwen 27B One-shot: Hydrogenolysis

Errors: **7** (`NG-01;NG-02;NG-15;NG-16;NG-17;NG-19;NG-20`).

- `NG-01`: missing groups=[['3-azetidinol'], ['hydrogenolysis']]
- `NG-02`: missing groups=[['pd(oh)2/al2o3', 'palladium hydroxide', 'pd(oh)2']]
- `NG-15`: reported=0.00238; ideal-gas expected=2.9424286106534874
- `NG-16`: reported=23.8; calculated=1770.4354279708969
- `NG-17`: inlet reported/calculated=300.0/0.059988002399520096; channel=300.0/242.32633279483036
- `NG-19`: inventory errors=['stream Liquid Feed flow 0.01 mL/min is outside pump limits']
- `NG-20`: missing safety groups=[]; missing topology=['temperature_control']

### Qwen 27B One-shot: Two-stage amidation

Errors: **1** (`NG-13`).

- `NG-13`: closure errors=['total tau=15.04, sum stages=10.68']

### GPT-5.4 One-shot: Hydrogenolysis

Errors: **3** (`NG-01;NG-16;NG-17`).

- `NG-01`: missing groups=[['hydrogenolysis']]
- `NG-16`: reported=2.95; calculated=3.5408708559417934
- `NG-17`: inlet reported/calculated=30.0/2.727272727272727; channel=12.0/18.598884066955982

### GPT-5.4 One-shot: Two-stage amidation

Errors: **1** (`NG-13`).

- `NG-13`: closure errors=['stage 2: tau=22.53, V/Q=23.78181818181818']

## Interpretation

The strongest architecture effect is numerical and operational closure. The most severe one-shot example was Qwen hydrogenolysis: the reported pressure-corrected hydrogen flow was inconsistent with the ideal-gas conversion, the hydrogen-equivalent calculation did not close, both residence-time bases were wrong, and the liquid flow was below the declared pump minimum. Both one-shot models also made a total-residence-time error in the two-stage case.

These data support the claim that deterministic engineering realization and inventory validation reduce detectable design errors. They do not establish superior reaction yield prediction, broad chemical generalization, or statistical significance because this package contains three cases and one run per model-architecture cell.

## Reproducibility

- Evaluator: `ablation_test/src/error_audit.py`
- Fixed expectations: `ablation_test/benchmarks/newgen_error_expectations_v1.json`
- Builder: `ablation_test/scripts/build_newgen_error_benchmark.py`
- Per-run audit JSONs: `audits/`
- Machine-readable tables: `tables/`
- Figures: `figures/`

The expectation file is retrospective for this current package but is derived only from the public protocol, hard constraints, inventory, and held-out source oracle. It must be frozen unchanged before any confirmatory reruns or additional models are tested.
