from flora_translate.main import _build_multistep_topology
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    ProcessStage,
    StreamAssignment,
    StreamLogic,
)


def test_multistage_topology_allocates_global_council_tau_and_keeps_air_as_mfc():
    batch = BatchRecord(
        reaction_description="One-pot photoredox Giese addition + aerobic oxidation",
        reaction_time_h=10.0,
        concentration_M=0.1,
        scale_mmol=0.2,
        solvent="EtOH / pH 9 buffer",
        temperature_C=25,
    )
    plan = ChemistryPlan(
        reaction_name="One-pot photoredox Giese addition + aerobic oxidation",
        reaction_class="photoredox aerobic oxidation",
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Giese radical addition",
                reaction_type="photoredox radical addition",
                reactor_type="coil",
                temperature_C=25,
                requires_light=True,
                wavelength_nm=452,
                batch_time_h=4.0,
                solvent="EtOH / pH 9 buffer",
                atmosphere="Ar",
                oxygen_sensitive=True,
                deoxygenation_required=True,
                feed_streams=[
                    StreamLogic(
                        stream_label="A",
                        reagents=["PMPSCH2TMS", "Ir photocatalyst"],
                        reasoning="O2-free substrate/catalyst solution",
                        molar_equiv=1.0,
                    ),
                    StreamLogic(
                        stream_label="B",
                        reagents=["acrylonitrile"],
                        reasoning="Michael acceptor",
                        molar_equiv=2.0,
                    ),
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Aerobic sulfide oxidation",
                reaction_type="photoredox aerobic oxidation",
                reactor_type="coil",
                temperature_C=25,
                requires_light=True,
                wavelength_nm=452,
                batch_time_h=6.0,
                solvent="EtOH / pH 9 buffer",
                atmosphere="air",
                feed_streams=[
                    StreamLogic(
                        stream_label="C",
                        reagents=["Air"],
                        reasoning="O2 gas feed",
                        molar_equiv=1.0,
                    )
                ],
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=38.38,
        flow_rate_mL_min=0.2234,
        tubing_ID_mm=0.5,
        reactor_volume_mL=8.574,
        temperature_C=25,
        concentration_M=0.1,
        BPR_bar=5,
        tubing_material="FEP",
        wavelength_nm=452,
        streams=[
            StreamAssignment(
                stream_label="A",
                contents=["PMPSCH2TMS", "Ir photocatalyst"],
                solvent="EtOH / pH 9 buffer",
                flow_rate_mL_min=0.0745,
            ),
            StreamAssignment(
                stream_label="B",
                contents=["acrylonitrile"],
                solvent="EtOH / pH 9 buffer",
                flow_rate_mL_min=0.1489,
            ),
            StreamAssignment(
                stream_label="C",
                pump_role="gas injection",
                contents=["Air"],
                flow_rate_mL_min=0.0,
            ),
        ],
        # Legacy bad state: only stage 1 received the full council tau.
        stage_parameters=[{"stage_number": 1, "residence_time_min": 38.38, "d_mm": 0.5}],
    )

    topology = _build_multistep_topology(proposal, plan, batch)
    reactors = [
        op for op in topology.unit_operations
        if op.op_type == "coil_reactor"
    ]
    assert [r.parameters["residence_time_min"] for r in reactors] == [15.35, 23.03]
    assert [r.parameters["Q_inlet_mL_min"] for r in reactors] == [0.2234, 0.2234]

    gas_feed = next(op for op in topology.unit_operations if op.op_id == "pump_c")
    assert gas_feed.op_type == "mfc"
    assert gas_feed.parameters["flow_rate_mL_min"] == 0.0
