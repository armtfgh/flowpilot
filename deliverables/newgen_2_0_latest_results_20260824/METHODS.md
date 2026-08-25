# Benchmark methods

The benchmark used three frozen flow-chemistry cases: hydrogenolysis, photochemical oxidation, and CuAAC. Each generator model produced a general one-shot outcome and a FlowPilot outcome for each case in three independent generation repeats. The same protocol, objective, authoritative inventory, architecture budget, and repeat identifiers were used within each matched model comparison.

Every delivered outcome was blinded to generator identity and evaluated against the same 14-criterion NewGen 2.0 rubric by three judge families: Qwen, OpenAI, and Claude. Applicable criterion scores ranged from 0 to 4 and were normalized to 0-1 before averaging. NOT_APPLICABLE criteria were excluded from the denominator. Deterministic verification supplied numerical, inventory, topology, and schema checks to every judge.

For each model and architecture, case scores were first averaged within each generation repeat. The reported campaign mean and sample standard deviation were then calculated across the three repeat-level means (n=3; ddof=1). Paired architecture effects were calculated as FlowPilot minus one-shot for the same model, case, and repeat, then averaged by repeat. Judge disagreement is reported separately and is not treated as generation-repeat variability. Error bars represent sample standard deviation, not confidence intervals.

All candidate, criterion-level, repeat-level, and provenance records are included in this folder. The completeness audit must pass before figures are generated.
