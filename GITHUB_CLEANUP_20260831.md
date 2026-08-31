# GitHub code-only cleanup

## Current benchmark scope

Only these generator models are part of the publication benchmark:

- Qwen3.6-27B
- Qwen3.8-27B
- GPT-4o
- Claude Sonnet 4.6
- Claude Opus 4.6

The active model list is locked by
`ablation_test/tests/test_current_benchmark_model_scope.py`.

## Removed from the publishable tree

- Superseded model-specific benchmark runners, configurations, and tests.
- Generated `benchmark/data/` campaigns.
- Generated `outputs/` application and benchmark runs.
- Generated `deliverables/` figures, reports, and office documents.
- Local retrieval records and Chroma database files.

The generated data were preserved locally under
`.local_artifacts_backup_20260831/`. Ignored symbolic links keep the original
local paths working, but neither the links nor their targets are published.

## Commit the cleanup

```bash
git add -A
git status --short
git commit -m "Clean benchmark model scope and remove generated artifacts"
git push
```

Do not force-add ignored result folders. They are local evidence archives, not
source code.
