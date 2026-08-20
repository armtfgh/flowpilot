# Supporting Figure Captions

## Figure S1

Extended FlowPilot architecture and authority boundaries. (a) The current pipeline converts a standardized intake package into a chemistry plan, retrieves literature analogies, performs deterministic engineering and design-space calculations, generates and audits candidates through the multi-agent council, reconciles the selected design against laboratory inventory, validates a single final-design contract, and renders topology and provenance artifacts. (b) Evidence authority is ordered from measured experimental evidence to model inference. Higher-authority information may constrain or override lower-authority suggestions. (c) Deterministic final gates separate executable screening designs from inventory-confirmation and blocked diagnostic outputs. LLM modules interpret chemistry; deterministic modules own numerical closure and feasibility.

## Figure S2

Typed contracts and field ownership. (a) Major Pydantic data objects passed through FlowPilot and their field counts in the current code. (b) Primary ownership of representative design fields. A filled cell denotes the module that creates, computes, constrains, or publishes the field. (c) Candidate revisions are not published directly: editable fields are recomputed, inventory-reconciled, and checked before entering the final contract.

## Figure S3

Reproducible standardized intake. (a) Fixed question bank with stable IDs, required/optional status, and the design context populated by each answer. (b) Readiness state machine. Batch protocol and objective must be answered; history, inventory, limits, and hypotheses must be answered or explicitly marked unavailable. (c) Pending question IDs are a deterministic function of package state, while the LLM is limited to extracting content and cannot invent new IDs.

## Figure S4

Frozen DesignInputPackage and evidence propagation. (a) Section-level status for the representative photoredox case used in this package. (b) Authority-labeled influence matrix showing where protocol facts, measured evidence, inventory, operating limits, hypotheses, and preferences enter the pipeline. (c) The package is serialized with schema version and content hash before design, providing a reproducible boundary between chemist input and model inference. Empty historical data are recorded as unavailable rather than silently omitted.

## Figure S5

GUI workflow and single-source result rendering. (a) User workflow from protocol intake and inventory selection to design execution and review. (b) Ten result tabs in the standardized-intake design view. All numerical tabs are rebuilt from the post-validation FinalDesignContract; supporting traces remain diagnostic. (c) Executable and blocked paths use different rendering rules: blocked runs expose requirements topology and reconciliation reasons but withhold run parameters.

## Figure S6

Run-level provenance and artifact integrity. (a) Artifact classes stored for the representative photoredox run. (b) Provenance chain from frozen input and model events through raw and canonical results, deterministic audit, topology, and checksums. (c) All nine final validation checks passed in the selected executable example. (d) Distribution of 57 recorded LLM events by pipeline component. Artifact counts describe this stored case and are not architecture requirements.

## Figure S7

Composition of the frozen flow-chemistry corpus (n = 464 classified records). Distributions are shown for (a) reaction class, (b) reactor type, (c) reactor material, (d) bond type, (e) number of inlet streams, and (f) available batch and optimized-flow yields. Unknown/other categories are retained to expose metadata incompleteness. Percentages use the available denominator for each field; yield distributions therefore do not imply complete yield coverage.

## Figure S8

Engineering rule-base structure (2,537 rules). (a) Counts by category and severity for the 14 largest categories. (b) Fraction of rules containing a machine-detected quantitative expression; this is an expression-coverage indicator, not proof that every expression is an independently validated equation. (c) Rule-category associations across chemistry classes. Cell values are association counts and may exceed category totals because one rule can map to multiple chemistry classes. (d) Co-occurrence network of the most frequent engineering concepts, with node size proportional to frequency and edge width proportional to co-occurrence.

## Figure S9

Current plan-aware retrieval workflow. (a) ChemistryPlan fields enrich the query before embedding or deterministic lexical retrieval. (b) Retrieval starts with mechanism/phase filters on paired records, relaxes filters when fewer than three hits are found, and finally searches all records when needed. (c) Candidate records are reranked by 0.60 semantic similarity and 0.40 field similarity. Field similarity combines photocatalyst, solvent, wavelength, temperature, and concentration terms using the current configuration. Leave-one-source-out evaluation excludes hidden reference record IDs after candidate construction and before top-k selection.

## Figure S10

Retrieval benchmark and leakage controls. (a) Top-5 photocatalyst-family match rates for semantic retrieval and FlowPilot reranking. (b) Distribution of rank changes across 1,600 frozen query-result pairs; 25.1% changed rank. (c) Mean and nonzero rates for available field-score components. (d) One iridium-query example showing the family composition of the top five before and after reranking. (e) Leave-one-source-out control and deterministic test coverage. These retrieval metrics measure metadata alignment, not downstream chemical yield or design optimality.
