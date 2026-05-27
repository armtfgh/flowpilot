from __future__ import annotations

from flora_translate.design_calculator import DesignCalculator
from flora_translate.engine.sampling import compute_metrics, generate_candidates, hard_filter
from flora_translate.engine.council_v4.chief import _intensification_feasibility_precheck
from flora_translate.engine.council_v4.skeptic import _verify_v_r_equals_tau_q
from flora_translate.main import (
    _build_multistep_topology,
    _build_singlestep_topology,
    _reconcile_final_bpr,
    _sync_final_stream_flowrates,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    ProcessStage,
    StreamAssignment,
    StreamLogic,
)


def test_gas_liquid_calculation_corrects_volume_and_adds_o2_metrics():
    batch = BatchRecord(
        reaction_description="Photoredox aerobic oxidation with molecular oxygen from air.",
        solvent="EtOH / water",
        temperature_C=25,
        reaction_time_h=6,
        concentration_M=0.1,
        scale_mmol=0.2,
        atmosphere="air",
    )
    plan = ChemistryPlan(
        reaction_class="photoredox oxidation",
        mechanism_type="gas-liquid photoredox oxidation",
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Aerobic oxidation",
                atmosphere="air",
                requires_light=True,
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["sulfide solution"], concentration_M=0.1),
                    StreamLogic(stream_label="G", reagents=["air"], phase="gas", reasoning="O2 feed"),
                ],
            )
        ],
    )
    proposal = FlowProposal(
        residence_time_min=10,
        flow_rate_mL_min=1.0,
        concentration_M=0.1,
        temperature_C=25,
        BPR_bar=5.0,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(stream_label="A", pump_role="liquid substrate", solvent="EtOH/water", concentration_M=0.1),
            StreamAssignment(stream_label="G", pump_role="air gas feed", contents=["air"], phase="gas"),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    proposal = DesignCalculator.annotate_proposal_with_calculations(proposal, calc)
    proposal.BPR_bar = 0.0
    topology = _build_multistep_topology(proposal, plan, batch)

    assert calc.is_gas_liquid
    assert calc.gas_species == "air"
    assert calc.gas_flow_sccm > 0
    assert calc.gas_flow_actual_mL_min > 0
    assert calc.gas_holdup > 0
    assert calc.reactor_volume_mL > calc.liquid_holdup_volume_mL
    assert calc.o2_equiv_supplied >= 2.9
    assert calc.UA_W_K > 0
    bprs = [op for op in topology.unit_operations if op.op_type == "bpr"]
    assert bprs
    assert bprs[-1].parameters["pressure_bar"] >= 5.0


def test_n2_blanket_does_not_trigger_multiphase_and_quench_is_not_reactor():
    batch = BatchRecord(
        reaction_description="Alpha bromination under N2 followed by thiosulfate quench. Aldehyde is O2-sensitive.",
        solvent="MeCN",
        temperature_C=25,
        reaction_time_h=0.5,
        concentration_M=0.667,
        scale_mmol=10,
        atmosphere="N2",
    )
    plan = ChemistryPlan(
        reaction_class="electrophilic alpha bromination",
        mechanism_type="thermal",
        quench_required=True,
        quench_reagent="Na2S2O3",
    )
    proposal = FlowProposal(
        residence_time_min=5,
        flow_rate_mL_min=1.0,
        concentration_M=0.667,
        temperature_C=25,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(stream_label="A", pump_role="aldehyde in MeCN", solvent="MeCN", concentration_M=0.667, flow_rate_mL_min=0.5),
            StreamAssignment(stream_label="B", pump_role="Br2 in MeCN", solvent="MeCN", concentration_M=0.667, flow_rate_mL_min=0.5),
            StreamAssignment(stream_label="Q", pump_role="inline quench", contents=["Na2S2O3"], solvent="water", flow_rate_mL_min=0.1),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    topology = _build_singlestep_topology(proposal, plan, batch)

    assert not calc.is_gas_liquid
    assert not any(op.op_type == "mfc" for op in topology.unit_operations)
    reaction_reactors = [
        op for op in topology.unit_operations
        if op.op_type in {"coil_reactor", "photoreactor", "chip_reactor"}
    ]
    assert len(reaction_reactors) == 1
    assert any(op.op_id == "quench_mixer" for op in topology.unit_operations)


def test_multistage_packed_bed_is_reactor_not_filter():
    batch = BatchRecord(
        reaction_description="Two-stage liquid reaction followed by H2 packed-bed hydrogenation.",
        solvent="MeCN",
        temperature_C=25,
        reaction_time_h=2,
        concentration_M=0.5,
        scale_mmol=5,
    )
    plan = ChemistryPlan(
        reaction_class="cascade hydrogenation",
        mechanism_type="gas-liquid hydrogenation",
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Liquid activation",
                reactor_type="coil",
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["substrate"], concentration_M=0.5),
                    StreamLogic(stream_label="B", reagents=["activator"], concentration_M=0.5),
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Hydrogenation",
                reactor_type="packed_bed",
                feed_streams=[
                    StreamLogic(stream_label="D", reagents=["H2 gas"], phase="gas"),
                ],
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=20,
        flow_rate_mL_min=0.5,
        concentration_M=0.5,
        temperature_C=25,
        BPR_bar=7.0,
        tubing_ID_mm=0.75,
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate", concentration_M=0.5),
            StreamAssignment(stream_label="B", pump_role="activator", concentration_M=0.5),
            StreamAssignment(stream_label="D", pump_role="H2 gas", contents=["H2 gas"], phase="gas"),
        ],
        multiphase_metrics={
            "gas_holdup": 0.5,
            "gas_flow_sccm": 10.0,
            "gas_flow_actual_mL_min": 2.0,
        },
    )

    topology = _build_multistep_topology(proposal, plan, batch)

    assert any(op.op_type == "packed_bed_reactor" for op in topology.unit_operations)
    assert not any(
        op.op_type == "inline_filter" and "hydrogenation" in op.label.lower()
        for op in topology.unit_operations
    )


def test_gas_stream_with_none_solvent_stays_single_mfc():
    batch = BatchRecord(
        reaction_description="Activated alkene chlorination with Cl2 gas feed followed by thiosulfate quench.",
        solvent="DCM",
        temperature_C=-20,
        reaction_time_h=0.5,
        concentration_M=0.2,
        scale_mmol=5,
    )
    plan = ChemistryPlan(
        reaction_class="chlorination",
        mechanism_type="gas-liquid electrophilic addition",
    )
    proposal = FlowProposal(
        residence_time_min=3,
        flow_rate_mL_min=1.0,
        concentration_M=0.2,
        temperature_C=-20,
        tubing_ID_mm=0.5,
        BPR_bar=7.0,
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate pump", contents=["alkene"], solvent="DCM", flow_rate_mL_min=1.0),
            StreamAssignment(stream_label="G", pump_role="gas mass flow controller", contents=["Cl2 gas"], solvent="none", flow_rate_mL_min=0.05),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    proposal = DesignCalculator.annotate_proposal_with_calculations(proposal, calc)
    topology = _build_singlestep_topology(proposal, plan, batch)

    assert calc.is_gas_liquid
    assert calc.gas_species == "Cl2"
    gas_ops = [op for op in topology.unit_operations if op.parameters.get("stream") == "G"]
    assert len(gas_ops) == 1
    assert gas_ops[0].op_type == "mfc"


def test_singlestep_packed_bed_topology_preserves_reactor_type():
    proposal = FlowProposal(
        residence_time_min=4,
        flow_rate_mL_min=1.0,
        concentration_M=0.2,
        temperature_C=25,
        tubing_ID_mm=0.75,
        reactor_type="packed_bed",
        tubing_material="stainless steel",
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate", contents=["nitroarene"], solvent="EtOH", flow_rate_mL_min=1.0),
            StreamAssignment(stream_label="B", pump_role="H2 gas feed", contents=["H2 gas"], phase="gas", flow_rate_mL_min=0.1),
        ],
    )

    topology = _build_singlestep_topology(proposal, ChemistryPlan(), BatchRecord())

    assert any(op.op_type == "packed_bed_reactor" for op in topology.unit_operations)


def test_gas_liquid_hard_filter_rejects_high_pressure_microtube():
    metrics = {
        "L_m": 10.0,
        "V_R_mL": 5.0,
        "Re": 50.0,
        "delta_P_bar": 120.0,
        "Q_mL_min": 1.0,
        "tau_min": 5.0,
        "d_mm": 0.28,
        "Da_mass": 0.1,
        "r_mix": 0.01,
        "expected_conversion": 0.9,
    }

    feasible, violations, _ = hard_filter(
        metrics,
        is_photochem=False,
        is_gas_liquid=True,
        pump_max_bar=200.0,
        BPR_bar=120.0,
    )

    assert not feasible
    assert any("gas-liquid ΔP" in v for v in violations)
    assert any("gas-liquid slug-flow practical floor" in v for v in violations)


def test_quench_stream_mentioning_gas_reagent_is_not_mfc():
    proposal = FlowProposal(
        residence_time_min=5,
        flow_rate_mL_min=1.0,
        concentration_M=0.2,
        temperature_C=25,
        tubing_ID_mm=0.75,
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate feed", contents=["alkene"], solvent="DCM", flow_rate_mL_min=1.0),
            StreamAssignment(stream_label="G", pump_role="Cl2 gas feed", contents=["Cl2"], phase="gas", flow_rate_mL_min=0.1),
            StreamAssignment(stream_label="Q", pump_role="quench stream for excess Cl2 destruction", contents=["Na2S2O3 aq"], flow_rate_mL_min=0.2),
        ],
        multiphase_metrics={"gas_flow_sccm": 10.0, "gas_flow_actual_mL_min": 0.1},
    )
    plan = ChemistryPlan(quench_required=True)

    topology = _build_singlestep_topology(proposal, plan, BatchRecord())
    quench = next(op for op in topology.unit_operations if op.parameters.get("stream") == "Q")

    assert quench.op_type == "pump"


def test_gas_stream_allows_null_solvent_from_llm_json():
    proposal = FlowProposal(
        residence_time_min=30,
        flow_rate_mL_min=0.1,
        streams=[
            {"stream_label": "A", "pump_role": "substrate in DMSO", "solvent": "DMSO"},
            {
                "stream_label": "G",
                "pump_role": "O2 gas feed",
                "contents": "O2",
                "phase": "gas",
                "solvent": None,
            },
        ],
    )

    assert proposal.streams[1].solvent == ""
    assert proposal.streams[1].contents == ["O2"]
    assert proposal.streams[1].phase == "gas"


def test_gas_liquid_candidate_metrics_include_holdup_pressure_and_bpr_gate():
    metrics = compute_metrics(
        tau_min=20.7,
        d_mm=0.5,
        Q_mL_min=0.1518,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=20.0,
        pump_max_bar=400.0,
        is_photochem=True,
        is_gas_liquid=True,
        BPR_bar=7.0,
    )

    assert metrics["V_R_mL"] > metrics["liquid_holdup_volume_mL"]
    assert metrics["two_phase_multiplier"] > 1.0
    assert metrics["required_bpr_bar"] > 10.0
    feasible, violations, _ = hard_filter(
        metrics,
        is_photochem=True,
        is_gas_liquid=True,
        pump_max_bar=400.0,
        BPR_bar=7.0,
    )
    assert not feasible
    assert any("routine gas-liquid ceiling" in v for v in violations)


def test_final_bpr_reconciliation_does_not_promote_stale_calculator_bpr():
    result = {
        "proposal": {"BPR_bar": 7.0},
        "design_calculations": {
            "is_gas_liquid": True,
            "bpr_pressure_bar": 23.4,
        },
    }

    _reconcile_final_bpr(result)

    assert result["proposal"]["BPR_bar"] == 7.0
    assert result["design_calculations"]["bpr_pressure_bar"] == 7.0
    assert "validation warning" in result["design_calculations"]["bpr_reconciliation_note"]


def test_gas_liquid_candidate_generation_uses_total_tube_volume_basis():
    feasible, infeasible = generate_candidates(
        tau_center_min=8.3,
        tau_lit_min=None,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=16.67,
        pump_max_bar=400.0,
        is_photochem=True,
        is_gas_liquid=True,
        BPR_bar=7.0,
        tau_low_factor=0.3,
        tau_high_factor=2.0,
        n_tau=5,
        d_exclude_above_mm=1.0,
        L_fractions=[0.4, 0.6, 0.8, 0.95],
        N_target=12,
    )

    assert feasible
    assert all(c["L_m"] <= 20.0 for c in feasible)
    assert all(c["required_bpr_bar"] <= 10.0 for c in feasible)
    assert len(infeasible) == 0


def test_skeptic_volume_audit_uses_liquid_holdup_for_gas_liquid():
    errors = _verify_v_r_equals_tau_q([
        {
            "id": 1,
            "tau_min": 25.0,
            "Q_mL_min": 0.1272,
            "liquid_holdup_volume_mL": 3.18,
            "V_R_mL": 21.2,
            "gas_holdup": 0.85,
        }
    ])

    assert errors == []


def test_final_stream_flowrates_sync_to_authoritative_liquid_q():
    proposal = FlowProposal(
        residence_time_min=93.8,
        flow_rate_mL_min=0.13397,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution feed",
                contents=["substrate"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.433,
            ),
            StreamAssignment(
                stream_label="B",
                pump_role="oxidant gas feed",
                contents=["O2"],
                phase="gas",
                flow_rate_mL_min=0.0793,
            ),
        ],
    )
    holder = type("Holder", (), {"proposal": proposal})()
    result = {
        "proposal": proposal.model_dump(),
        "design_calculations": {
            "is_gas_liquid": True,
            "liquid_flow_rate_mL_min": 0.13397,
            "gas_flow_sccm": 4.504,
            "gas_flow_actual_mL_min": 0.0793,
        },
    }

    _sync_final_stream_flowrates(result, holder)

    streams = result["proposal"]["streams"]
    assert streams[0]["flow_rate_mL_min"] == 0.13397
    assert streams[1]["flow_rate_mL_min"] == 0.0793
    assert streams[1]["gas_flow_sccm"] == 4.504
    assert holder.proposal.streams[0].flow_rate_mL_min == 0.13397


def test_marginal_intensification_conflict_proceeds_as_screen_required():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=187.5,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "KINETIC_ANCHOR_UNCERTAIN_SCREEN_REQUIRED"
    assert diagnostic["hard_block"] is False
    assert diagnostic["required_to_ceiling_ratio"] == 1.042


def test_candidate_under_ceiling_prevents_pre_council_hard_block():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=187.5,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
        candidate_tau_min=93.8,
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "KINETIC_ANCHOR_UNCERTAIN_SCREEN_REQUIRED"
    assert diagnostic["hard_block"] is False
    assert diagnostic["candidate_tau_min"] == 93.8
    assert diagnostic["candidate_projected_conversion"] >= 0.5


def test_large_intensification_conflict_still_blocks_before_council():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=300.0,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "INFEASIBLE_WITH_CURRENT_KINETIC_ANCHOR"
    assert diagnostic["hard_block"] is True


def test_bad_low_conversion_candidate_does_not_bypass_hard_block():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=300.0,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
        candidate_tau_min=40.0,
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "INFEASIBLE_WITH_CURRENT_KINETIC_ANCHOR"
    assert diagnostic["hard_block"] is True
    assert diagnostic["candidate_projected_conversion"] < 0.5
