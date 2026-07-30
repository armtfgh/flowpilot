# Package Contents

The release archive is rooted at the project directory and contains:

- `ablation_test/readiness_package_20260728/`: report, summary, score specification, figures, tables, and verification.
- `ablation_test/runs/pilot_readiness_dev_frozen_v2_gpt4o_v2_20260728/`: 42-cell development matrix with raw JSON, LLM events, stage events, council snapshots, metrics, and checksums.
- `ablation_test/runs/pilot_readiness_holdout_frozen_v2_gpt4o_20260728/`: 42-cell untouched holdout matrix with the same raw artifacts.
- `ablation_test/src/metrics.py`: frozen architecture-blind v2 scorer.
- `ablation_test/src/runner.py`: benchmark execution paths.
- `ablation_test/scripts/`: benchmark runner and readiness report generator.
- `ablation_test/protocols/literature_cases.json`: public protocol suite and hidden-reference metadata.
- `flora_translate/`: changed production calculation, inventory, validation, and council modules.
- `benchmark/pipeline.py`: production benchmark finalization path.
- `flora_translate/tests/` and `ablation_test/tests/`: regression suites used for verification.

Earlier benchmark directories and release archives remain in the workspace and were not removed.
