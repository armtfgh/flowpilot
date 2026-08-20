# FlowPilot Ablation Results

This directory stores generated ablation artifacts and is intentionally excluded
from Git, except for this file. The code-only benchmark package is in
`../ablation_test/`.

Default generated subdirectories include:

- `runs/`: raw prompts, responses, logs, JSON results, and checksums.
- `studies/`: staged benchmark runs and confirmatory analyses.
- `figures/`, `tables/`, `reports/`: aggregate analysis outputs.
- `packages/`: collaborator packages and compressed archives.

Set `FLOWPILOT_ABLATION_RESULTS_DIR` to use a different results location. For
example:

```bash
export FLOWPILOT_ABLATION_RESULTS_DIR=/data/flowpilot_ablation_results
python -m ablation_test.scripts.run --profile smoke
```

The split was applied on 2026-08-03. Historical manifests were left unchanged
to preserve their recorded evidence; active readers rebase legacy
`ablation_test/runs/...` and `ablation_test/studies/...` paths automatically.
