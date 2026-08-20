# FlowPilot Model Repair Priorities

## P0: One Numerical Authority

Make `final_design` the sole numerical source after realization. Regenerate explanation, topology, instrument manifest, recipe, engineering metrics, and validation from that object. Do not carry model-written scalar values forward.

Add deterministic equality checks with explicit tolerances before publication:

- General repeated scalar values: relative tolerance 1% unless an exact inventory setpoint is required.
- Material-balance and residence-time closure: 5%.
- Gas-law and gas-equivalent closure: 5% with explicit STP and absolute-pressure definitions.
- Inventory IDs, reactor volumes, stage count, and topology connectivity: exact.

Priority examples: `FP-AUDIT-071`, `FP-AUDIT-075` through `FP-AUDIT-084`, `FP-AUDIT-088`, `FP-AUDIT-089` through `FP-AUDIT-091`, and `FP-AUDIT-095`.

## P0: Gas Calculation Contract

Create one typed gas-basis object containing gas identity, STP definition, inlet flow, molar flow, absolute and gauge pressure, channel temperature, pressure-corrected channel flow, equivalents, holdup model, inlet-basis residence time, and channel-basis residence time. All displays and calculations must read this object.

Reject any design when two pressure bases or two gas-flow values survive final validation.

## P1: Superseded Record Handling

Every council or inventory revision must emit a machine-readable supersession record with old value, new value, reason, authority, and downstream artifacts regenerated. Intermediate records remain available for provenance but must be labeled `rejected` or `superseded`, never presented as current instructions.

Priority examples: `FP-AUDIT-072`, `FP-AUDIT-084`, `FP-AUDIT-085`, `FP-AUDIT-088`, `FP-AUDIT-092`, and `FP-AUDIT-095`.

## P1: Assumption And Evidence Control

Require provenance classes for every kinetic, transport, heat-transfer, and performance value: measured, protocol fact, inventory fact, literature evidence, calculation, or model assumption. Conversion predictions and disposition must carry uncertainty from assumed values.

Reject invented DOI, literature, equipment, or conversion claims. Irrelevant analogies should be omitted rather than used as support.

Priority examples: `FP-AUDIT-073`, `FP-AUDIT-074`, `FP-AUDIT-086`, `FP-AUDIT-087`, `FP-AUDIT-093`, and `FP-AUDIT-094`.

## P1: Stage-Specific Engineering

For multistage designs, calculate flow, volume, residence time, geometry, pressure drop, heat transfer, startup waste, and productivity per stage. Aggregate values may be displayed only after stage closure and must state the aggregation rule.

Priority examples: `FP-AUDIT-090`, `FP-AUDIT-091`, and `FP-AUDIT-095`.

## P2: Executable Operations

Generate topology and operating procedures from allocated inventory. Explicitly type gas and liquid streams and include startup, steady-state qualification, shutdown, emergency shutdown, depressurization, quench, waste, and collection controls when applicable.

Priority examples: `FP-AUDIT-082` and the Qwen FlowPilot safety/operations tickets in `tables/model_fix_tickets.csv`.

## Regression Gate

After implementing these fixes, rerun the same 12 frozen source outputs through deterministic validation first. Then regenerate new outputs from the same public inputs and run the holistic audit. Keep the old campaign immutable for paired before/after comparison. Do not optimize prompts against individual case answers; fixes must operate on typed contracts and universal validators.
