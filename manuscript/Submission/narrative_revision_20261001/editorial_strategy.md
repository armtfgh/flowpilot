# Editorial Strategy

## Central Argument

FlowPilot is an agentic research assistant for laboratory-constrained, end-to-end
flow-process design. A batch procedure is its chemical starting information, not
the definition of the scientific contribution. The contribution is to coordinate
chemical intent, evidence, candidate exploration, quantitative engineering and
equipment realization into an inspectable process proposal that can be tested
and revised by the researcher.

End-to-end refers to the design workflow. It must not imply autonomous physical
execution, demonstrated global optimization, validated kinetics, or a safety
guarantee.

## Narrative Sequence

1. Introduction: establish process design as a coupled scientific problem;
   situate synthesis planning, automation and recent chemical agents fairly;
   introduce FlowPilot's researcher-facing purpose, integrated design mechanism
   and evaluation strategy in more than a closing list of components.
2. Figure 1: explain how decisions propagate through the connected process and
   why the chemist, specialist council and deterministic tools have different
   responsibilities.
3. Figures 2-3: explain the role of literature and handbook knowledge, then test
   retrieval alignment without presenting metadata agreement as independent
   chemical validation.
4. Figure 4: ask whether the architecture improves delivered designs relative
   to the same model operating in one shot; explain models, tasks, matching,
   scoring, repeats and interpretation. Separate architecture effects from
   model capability and from additional computation. Discuss the internal
   ablation as a distinct, exploratory experiment, including configurations
   that do not improve on the simpler alternatives.
5. Figures 5-6: move from design-quality assessment to experimental process
   development. Explain common response-set logic once, then connect each
   chemistry's requirements, realized apparatus and measured outcomes. Preserve
   distinctions between proposed settings, actual implementation and reference
   literature. Treat all experimental work as the authors' work.
6. Discussion: synthesize what the architecture and experiments jointly support;
   identify the practical role of traceable decisions and experimental feedback
   without repeating the entire Results section or claiming autonomous success.
7. Conclusion: state the contribution and what the combined evidence establishes.
   Methods retains the detailed reproducible procedures; ESI retains exact
   inputs, scores, archived discussions, apparatus and analytical records.

## Evidence Boundaries

- Five-model aggregate: 90 outcomes, three chemistries, three generation repeats.
  Figure 4 additionally contains an archived sixth generator; disclose this
  instead of changing the figure or silently altering the cohort.
- Fixed 14-criterion rubric; separate Qwen/OpenAI/Claude judging calls, equal
  applicable-criterion and judge weights, integer ratings divided by four.
  Judge flags are not counts of independently confirmed physical failures.
- Module screen: 15 conditions, three cases, one generation per cell. Its SD
  describes between-case variation, not repeatability or significance.
- Retrieval source exclusion does not establish absence from model pretraining.
- Experimental response sets: integrated yield-focused design, conversion-
  focused shortest justified times, and throughput/compactness-oriented design.
  These are preferences/hypotheses, not independent experimental replicates.
- Each chemistry has two distinct reported operating points: Giese Sets 1/3
  share an entry; amidation Sets 2/3 share an entry.
- Pure oxygen was specified by the researchers in the revised inputs, not
  autonomously discovered by FlowPilot.
- Giese Stage 2's STP-based index is not an operating-pressure residence time.
- Preserve NMR yield bases, sample-loop qualification and missing metadata.

## Preservation and Verification

Use the latest inventory-GUI manuscript/ESI as immutable sources. Preserve all
embedded images, their geometry, tables, author details, bibliography, styles,
page setup and existing reference numbering. Change prose and selected headings;
align ESI framing without rewriting verbatim benchmark inputs or archived
conversations. Produce clean and yellow-highlighted Word copies, rendered PDFs,
an edit log and checks for citations, source numbers, image preservation and
layout. Inspect the rendered manuscript, including figure-caption placement.
