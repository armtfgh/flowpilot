# Verification Record

## Regression

Command:

```bash
python -m pytest flora_translate/tests ablation_test/tests -q
```

Result: `72 passed in 0.63s`

## Development Matrix

- Experiment: `pilot_readiness_dev_frozen_v2_gpt4o_v2_20260728`
- Planned cells: 42
- Completed result JSONs: 42
- Completed metric JSONs: 42
- Status: all completed

## Untouched Holdout Matrix

- Experiment: `pilot_readiness_holdout_frozen_v2_gpt4o_20260728`
- Planned cells: 42
- Completed result JSONs: 42
- Completed metric JSONs: 42
- Status: all completed

## Analysis

- Matched runs: 84
- Architectures: 7
- Development protocols: 6
- Untouched holdout protocols: 6
- PNG figures: 14
- PDF figures: 14
- CSV tables: 5
- Primary score: frozen Quality and Assurance Score v2
- Scorer SHA-256: `e86509c9a913249a7bccf73c49d9ae884a2748b74ddc5fcda74fe5f5f254b5e8`

## Result

Full FlowPilot ranked first on the primary architecture-blind quality score:

- Development: `0.920`
- Untouched holdout: `0.858`
- Best holdout comparator (`no_council`): `0.793`
- Full holdout paired record: 6 wins, 0 losses against every comparator

Deployment readiness is reported separately. Full FlowPilot scored `0.684`
on holdout readiness and marked five of six cases for additional screening.
