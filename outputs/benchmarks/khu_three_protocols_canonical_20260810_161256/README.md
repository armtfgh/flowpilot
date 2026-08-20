# KHU Three-Protocol FlowPilot Package

This folder contains the canonical outputs for one photoredox case and two DPDTC amide-coupling cases from `2 protocols.pdf`, all constrained to `khu_inventory_updated_flowpilot.json`.

- `manifest.json`: provenance, accepted source runs, and package status.
- `summary.json` / `summary.csv`: cross-case final parameters.
- `<case>/protocol.txt`: exact protocol passed to FlowPilot.
- `<case>/intake_package.json`: frozen standardized intake.
- `<case>/run.log` and `llm_events.jsonl`: complete model and pipeline logs.
- `<case>/raw_run_result.json`: untouched result from the fresh benchmark.
- `<case>/audit.json`: stage arithmetic, topology, and inventory audit.
- `<case>/final/<run>/`: canonical result, final design, allocation, topology, PNG, and SVG.

The two PDF cases excluded the publication's flow conditions from the model prompt. All outputs are screening recommendations, not wet-lab validation. LOW-confidence cases require conservative experimental screening.
