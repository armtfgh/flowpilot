# Claude Opus 5 diagnostic

Claude Opus 5 is the sole model in this matched campaign for which FlowPilot scored below its one-shot condition: `0.772 ± 0.087` versus `0.936 ± 0.023`. This result is retained as observed; it was not removed or replaced.

Five of nine Opus 5 FlowPilot outcomes were deterministically blocked:

- All three hydrogenolysis repeats introduced duplicate hydrogen stream representations. The realization layer consequently detected unresolved MFC/mixer allocation and withheld an executable design.
- Two CuAAC repeats introduced chemistry-inconsistent hardware requirements: one requested an undeclared light source and one introduced an unsupported oxygen stream. The third repeat preserved the expected CuAAC process and was executable.
- All three photochemical oxidation repeats were executable and scored consistently (`0.953 ± 0.004`).

The low aggregate therefore reflects upstream model-to-contract incompatibility and correctly conservative inventory gates, rather than arithmetic failure hidden by the report. It identifies a production issue to address through canonical chemistry-plan normalization and duplicate-stream reconciliation. Because these causes were diagnosed after observing the frozen campaign, they were not used to alter or rerun the scored outputs.
