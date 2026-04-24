"""FLORA-Translate — All Pydantic v2 data models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hardware / Lab Inventory
# ---------------------------------------------------------------------------


class PumpSpec(BaseModel):
    name: str
    type: str  # "HPLC" / "syringe" / "peristaltic"
    max_pressure_bar: float
    max_flow_rate_mL_min: float
    min_flow_rate_mL_min: float


class TubingSpec(BaseModel):
    material: str  # "FEP" / "PFA" / "SS" / "PTFE"
    ID_mm: float
    max_pressure_bar: float
    max_temperature_C: float
    transparent: bool


class LightSourceSpec(BaseModel):
    name: str
    wavelength_nm: float
    power_W: float
    compatible_reactor: str  # "coil" / "chip" / "both"


class ReactorSpec(BaseModel):
    type: str
    material: str
    volume_mL: float
    ID_mm: float


class LabInventory(BaseModel):
    pumps: list[PumpSpec] = []
    tubing: list[TubingSpec] = []
    BPR_available: list[float] = []
    light_sources: list[LightSourceSpec] = []
    reactors: list[ReactorSpec] = []

    @classmethod
    def from_json(cls, path: str) -> LabInventory:
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        return cls(**data)


# ---------------------------------------------------------------------------
# Input — Batch Record
# ---------------------------------------------------------------------------


class BatchRecord(BaseModel):
    reaction_description: str = ""
    photocatalyst: Optional[str] = None
    catalyst_loading_mol_pct: Optional[float] = None
    base: Optional[str] = None
    solvent: Optional[str] = None
    temperature_C: Optional[float] = None
    reaction_time_h: Optional[float] = None
    concentration_M: Optional[float] = None
    scale_mmol: Optional[float] = None
    yield_pct: Optional[float] = None
    light_source: Optional[str] = None
    wavelength_nm: Optional[float] = None
    additives: Optional[list[str]] = None
    atmosphere: Optional[str] = None
    raw_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Process Records (from PRISM / paper_knowledge_extractor)
# ---------------------------------------------------------------------------


class ConditionsRecord(BaseModel):
    photocatalyst: Optional[str] = None
    catalyst_loading_mol_pct: Optional[float] = None
    base: Optional[str] = None
    solvent: Optional[str] = None
    temperature_C: Optional[float] = None
    reaction_time_h: Optional[float] = None
    residence_time_min: Optional[float] = None
    flow_rate_mL_min: Optional[float] = None
    concentration_M: Optional[float] = None
    yield_pct: Optional[float] = None
    light_source: Optional[str] = None
    wavelength_nm: Optional[float] = None


class ProcessDesignRecord(BaseModel):
    reactor_type: Optional[str] = None
    tubing_material: Optional[str] = None
    tubing_ID_mm: Optional[float] = None
    tubing_length_m: Optional[float] = None
    reactor_volume_mL: Optional[float] = None
    pump_type: Optional[str] = None
    mixer_type: Optional[str] = None
    BPR_bar: Optional[float] = None
    BPR_required: Optional[bool] = None
    phase_regime: Optional[str] = None
    light_setup: Optional[str] = None
    deoxygenation_method: Optional[str] = None


class EngineeringLogicRecord(BaseModel):
    batch_limitation: Optional[str] = None
    flow_advantage: Optional[str] = None
    safety_concern: Optional[str] = None
    intensification_factor: Optional[float] = None
    batch_to_flow_yield_delta: Optional[float] = None


class TranslationRecord(BaseModel):
    batch_baseline: Optional[ConditionsRecord] = None
    flow_optimized: Optional[ConditionsRecord] = None
    flow_process_design: Optional[ProcessDesignRecord] = None
    reasoning: Optional[str] = None


class ProcessRecord(BaseModel):
    record_id: str = ""
    doi: str = ""
    title: str = ""
    year: int = 0
    chemistry_class: str = ""
    mechanism_type: str = ""        # "radical" | "SET" | "EnT" | "HAT" | "ionic" | ""
    phase_regime: str = "single_phase_liquid"  # "single_phase_liquid" | "gas_liquid" | "liquid_liquid" | "solid_liquid"
    process_mode: str = ""  # "batch" / "flow" / "both"
    conditions: ConditionsRecord = Field(default_factory=ConditionsRecord)
    process_design: Optional[ProcessDesignRecord] = None
    engineering_logic: Optional[EngineeringLogicRecord] = None
    translation_record: Optional[TranslationRecord] = None
    confidence: int = 1
    embedding_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Chemistry Plan (Layer 1 — generated before retrieval and translation)
# ---------------------------------------------------------------------------


class ReagentRole(BaseModel):
    """A single chemical species and its role in the reaction."""
    name: str = ""                       # e.g. "Ir(ppy)3", "sulfenamide 1a"
    role: str = ""                       # "photocatalyst" | "substrate" | "oxidant" | "base" | "additive" | "solvent" | "sensitizer" | "co-catalyst" | "quencher"
    equiv_or_loading: str = ""           # "1.0 equiv" or "2 mol%" — free text
    smiles: Optional[str] = None         # if known
    notes: str = ""                      # e.g. "light-sensitive", "moisture-sensitive"


class MechanismStep(BaseModel):
    """One elementary step in the proposed mechanism."""
    step_number: int = 0
    description: str = ""                # e.g. "Photoexcitation of Ir(III) to *Ir(III)"
    species_involved: list[str] = Field(default_factory=list)
    is_photon_dependent: bool = False    # does this step require light?
    is_rate_limiting: bool = False


class StreamLogic(BaseModel):
    """Chemistry-driven reasoning for which reagents go in which stream."""
    stream_label: str = ""               # "A", "B"
    reagents: list[str] = Field(default_factory=list)   # names of reagents in this stream
    reasoning: str = ""                  # WHY these go together
    molar_equiv: float = 1.0             # stoichiometric equivalents relative to limiting reagent
    concentration_M: Optional[float] = None  # concentration of this stream (for Q calculation)


class ProcessStage(BaseModel):
    """One stage of a multi-step flow process.

    Each stage = one reactor zone with its own feeds, conditions, and outlet.
    Stages are connected in sequence: stage N outlet feeds into stage N+1
    (possibly via a quench mixer or solvent switch point).
    """
    stage_number: int = 1
    stage_name: str = ""                  # e.g. "Grignard formation", "Reduction"
    reaction_type: str = ""               # e.g. "nucleophilic addition", "photoredox"
    reactor_type: str = "coil"            # "coil" | "packed_bed" | "chip" | "CSTR"
    temperature_C: Optional[float] = None
    requires_light: bool = False
    wavelength_nm: Optional[float] = None

    # What feeds into this stage
    feed_streams: list[StreamLogic] = Field(default_factory=list)
    # What comes from the previous stage (empty for stage 1)
    inlet_from_previous: str = ""         # e.g. "crude ArMgBr in THF"

    # Conditions and sensitivities for THIS stage
    solvent: str = ""
    atmosphere: str = ""                  # "N2" | "Ar" | "air" | "H2"
    oxygen_sensitive: bool = False
    moisture_sensitive: bool = False
    deoxygenation_required: bool = False

    # What happens between THIS stage and the NEXT
    post_stage_action: str = ""           # "quench with NH4Cl" | "solvent switch to DMF" | "inline filter" | ""
    post_stage_reasoning: str = ""


class ChemistryPlan(BaseModel):
    """Pure chemistry analysis — no hardware decisions.

    Generated by the Chemistry Reasoning Agent (Layer 1) before
    retrieval or hardware translation.

    For multi-step reactions, the `stages` list describes the process
    as an ordered sequence of stages, each with its own feeds, reactor
    type, conditions, and inter-stage actions. The flat fields
    (stream_logic, deoxygenation_required, etc.) describe the OVERALL
    process or the first stage for backward compatibility.
    """
    # Reaction identity
    reaction_name: str = ""
    reaction_class: str = ""
    mechanism_type: str = ""
    bond_formed: str = ""
    bond_broken: str = ""

    # Multi-step stages (empty for single-step — backward compatible)
    stages: list[ProcessStage] = Field(default_factory=list)
    n_stages: int = 1

    # All species (across all stages)
    reagents: list[ReagentRole] = Field(default_factory=list)

    # Mechanism
    mechanism_steps: list[MechanismStep] = Field(default_factory=list)
    key_intermediate: str = ""
    excited_state_type: str | None = ""
    energy_transfer_or_redox: str | None = ""

    # Sensitivity and constraints (overall)
    oxygen_sensitive: bool = False
    moisture_sensitive: bool = False
    temperature_sensitive: bool = False
    light_sensitive_reagents: list[str] = Field(default_factory=list)

    # Stream separation logic (overall / stage-1 for backward compat)
    stream_logic: list[StreamLogic] = Field(default_factory=list)
    mixing_order_reasoning: str = ""
    incompatible_pairs: list[list[str]] = Field(default_factory=list)

    # Pre/post reactor chemistry (overall / stage-1)
    deoxygenation_required: bool = False
    deoxygenation_reasoning: str = ""
    quench_required: bool = False
    quench_reagent: str = ""
    quench_reasoning: str = ""

    # Retrieval hints
    retrieval_keywords: list[str] = Field(default_factory=list)
    similar_reaction_classes: list[str] = Field(default_factory=list)

    # Wavelength recommendation
    recommended_wavelength_nm: Optional[float] = None
    wavelength_reasoning: str = ""

    # Confidence
    confidence_notes: str = ""


# ---------------------------------------------------------------------------
# Output — Flow Proposal
# ---------------------------------------------------------------------------


class StreamAssignment(BaseModel):
    """Describes what goes into a single pump/stream."""
    stream_label: str = ""              # "A", "B", "C"
    pump_role: str = ""                 # e.g. "substrate + photocatalyst", "oxidant"
    contents: list[str] = Field(default_factory=list)  # e.g. ["sulfenamide (0.1M)", "Ir(ppy)3 (1 mol%)"]
    solvent: str = ""
    concentration_M: Optional[float] = None
    flow_rate_mL_min: Optional[float] = None
    molar_equiv: float = 1.0            # stoichiometric equivalents relative to limiting reagent (substrate=1.0)
    reasoning: str = ""                 # why these go together


class FlowProposal(BaseModel):
    # Proposed flow conditions
    residence_time_min: float = 0
    flow_rate_mL_min: float = 0
    temperature_C: float = 25
    concentration_M: float = 0.1
    BPR_bar: float = 0
    reactor_type: str = "coil"
    tubing_material: str = "FEP"
    tubing_ID_mm: float = 1.0
    reactor_volume_mL: float = 0
    light_setup: str = ""
    wavelength_nm: Optional[float] = None
    deoxygenation_method: Optional[str] = None

    # Chemistry-aware stream assignments
    streams: list[StreamAssignment] = Field(default_factory=list)
    mixer_type: str = "T-mixer"
    mixing_order_reasoning: str = ""     # why this mixing order matters
    pre_reactor_steps: list[str] = Field(default_factory=list)  # e.g. ["degas stream A with N2"]
    post_reactor_steps: list[str] = Field(default_factory=list)  # e.g. ["inline quench with Na2S2O3"]
    chemistry_notes: str = ""            # mechanism-specific design notes

    # Per-stage parameters (populated by council Chief for multi-step processes)
    # Each dict: {stage_number, tau_fraction, d_mm, Q_inlet_mL_min, V_R_mL}
    stage_parameters: list[dict] = Field(default_factory=list)

    # Reasoning
    reasoning_per_field: dict[str, str] = Field(default_factory=dict)
    literature_analogies: list[str] = Field(default_factory=list)

    # Status
    engine_validated: bool = False
    safety_flags: list[str] = Field(default_factory=list)
    confidence: str = "LOW"  # "HIGH" / "MEDIUM" / "LOW"


# ---------------------------------------------------------------------------
# FLORA-Design — Chemistry Features, Unit Operations, Topology
# ---------------------------------------------------------------------------


class ChemFeatures(BaseModel):
    """Structured chemistry features extracted from user goal text."""
    reaction_class: str = "unknown"
    photocatalyst: Optional[str] = None
    photocatalyst_class: Optional[str] = None
    wavelength_nm: Optional[float] = None
    base: Optional[str] = None
    solvent: Optional[str] = None
    temperature_C: Optional[float] = None
    concentration_M: Optional[float] = None
    scale: str = "lab"
    phase_regime: str = "single_phase_liquid"
    generates_gas: bool = False
    has_solid_catalyst: bool = False
    generates_precipitate: bool = False
    O2_sensitive: bool = True
    light_sensitive_product: bool = False
    hazard_level: str = "low"
    exothermic: bool = False
    multi_step: bool = False
    number_of_steps: int = 1
    classifier_confidence: float = 0.0
    ambiguous_fields: list[str] = Field(default_factory=list)


class UnitOperation(BaseModel):
    """A single unit operation in the process."""
    op_id: str = ""
    op_type: str = ""
    label: str = ""
    parameters: dict = Field(default_factory=dict)
    required: bool = True
    rationale: str = ""


class StreamConnection(BaseModel):
    """A stream connecting two unit operations."""
    stream_id: str = ""
    from_op: str = ""
    to_op: str = ""
    stream_type: str = "liquid"
    label: Optional[str] = None


class ProcessTopology(BaseModel):
    """Complete ordered process topology from inlet to outlet."""
    topology_id: str = ""
    unit_operations: list[UnitOperation] = Field(default_factory=list)
    streams: list[StreamConnection] = Field(default_factory=list)
    total_flow_rate_mL_min: float = 0
    residence_time_min: float = 0
    reactor_volume_mL: float = 0
    pid_description: str = ""
    literature_support: list[str] = Field(default_factory=list)
    topology_confidence: str = "MEDIUM"


class DesignResult(BaseModel):
    """Final output of FLORA-Design."""
    goal_text: str = ""
    chem_features: ChemFeatures = Field(default_factory=ChemFeatures)
    topology: ProcessTopology = Field(default_factory=ProcessTopology)
    design_candidate: Optional[DesignCandidate] = None
    svg_path: str = ""
    png_path: str = ""
    explanation: str = ""
    retrieved_records: list[str] = Field(default_factory=list)
    alternatives: list[ProcessTopology] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ENGINE — Council Messages (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------


class CouncilMessage(BaseModel):
    agent: str
    status: str  # "ACCEPT" / "WARNING" / "REJECT"
    field: str = ""
    value: str = ""
    concern: str = ""
    revision_required: bool = False
    suggested_revision: Optional[str] = None


# ---------------------------------------------------------------------------
# ENGINE — Deliberation Log (new: LLM-powered multi-agent council)
# ---------------------------------------------------------------------------


class FieldProposal(BaseModel):
    """A single concrete, machine-applicable design change proposed by an agent."""
    field: str = ""                       # FlowProposal field name, e.g. "residence_time_min"
    value: str = ""                       # Target value as string, e.g. "15.0"
    reason: str = ""                      # Why this change is needed


class AgentDeliberation(BaseModel):
    """One agent's contribution in a single deliberation round.

    Agents READ the DesignCalculator output (authoritative physics) and
    INTERPRET it — they do not re-derive Re, ΔP, Da, etc.  Their role
    is to assess whether the design is adequate in their domain and
    propose concrete field-level changes when it is not.
    """
    agent: str = ""                      # "KineticsSpecialist", etc.
    agent_display_name: str = ""         # "Dr. Kinetics", etc.
    round: int = 1
    chain_of_thought: str = ""           # Full reasoning (shown to user)
    values_referenced: list[str] = Field(default_factory=list)  # Calculator values cited
    findings: list[str] = Field(default_factory=list)       # Bullet-point findings
    proposals: list[FieldProposal] = Field(default_factory=list)  # Structured changes
    concerns: list[str] = Field(default_factory=list)       # Issues raised
    status: str = "ACCEPT"               # "ACCEPT" / "WARNING" / "REVISE"
    had_error: bool = False              # True if agent LLM call failed — blocks convergence
    references_to_agents: list[str] = Field(default_factory=list)
    rules_cited: list[str] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)  # [{"tool": name, "input": {...}, "result": {...}}, ...]


class SanityCheckResult(BaseModel):
    """Central orchestrator's cross-agent consistency check."""
    round: int = 1
    consistent: bool = True
    chain_of_thought: str = ""
    conflicts_found: list[str] = Field(default_factory=list)
    resolutions: list[str] = Field(default_factory=list)
    # Only simple numeric/string FlowProposal fields — no lists, no nested objects
    final_changes: dict[str, str] = Field(default_factory=dict)


class DeliberationLog(BaseModel):
    """Complete record of the multi-agent deliberation process."""
    rounds: list[list[AgentDeliberation]] = Field(default_factory=list)
    sanity_checks: list[SanityCheckResult] = Field(default_factory=list)
    total_rounds: int = 0
    consensus_reached: bool = False
    # Cumulative record of all field changes applied across all rounds
    all_changes_applied: dict[str, str] = Field(default_factory=dict)
    summary: str = ""                    # Human-readable summary of deliberation
    trade_off_summary: str = ""   # Skeptic's cross-pick comparative narrative
    trade_off_matrix: str = ""    # Pre-computed surviving-picks comparison table (Chief input)


class DesignCandidate(BaseModel):
    proposal: FlowProposal
    chemistry_plan: Optional[ChemistryPlan] = None
    council_messages: list[CouncilMessage] = Field(default_factory=list)
    council_rounds: int = 0
    safety_report: dict = Field(default_factory=dict)
    unit_operations: list[str] = Field(default_factory=list)
    pid_description: str = ""
    human_explanation: str = ""
    deliberation_log: Optional[DeliberationLog] = None
