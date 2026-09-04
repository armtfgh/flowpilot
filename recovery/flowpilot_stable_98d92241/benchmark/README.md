# Benchmark Workspace

This folder is isolated from the Streamlit app and is intended for council
benchmarking only.

Contents:

- `cases.py`: built-in benchmark case library.
- `recorder.py`: on-disk recorder that writes JSONL/JSON while the run is in progress.
- `pipeline.py`: frozen upstream preparation + council execution helpers.
- `run_candidate_budget_benchmark.py`: benchmark runner for `N = 1 / 6 / 12 / 24`.
- `summarize.py`: collates run-level and LLM-level manifests into CSV files.

All benchmark data is written under `benchmark/data/...`.
