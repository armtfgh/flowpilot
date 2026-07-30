from pathlib import Path

from flora_translate.final_design_validator import finalize_design
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
    StreamAssignment,
)


INVENTORY_PATH = Path("flora_translate/data/lab_inventory.json")


def test_finalizer_atomically_enforces_inventory_and_geometry_closure():
    inventory = LabInventory.from_json(str(INVENTORY_PATH))
    batch = BatchRecord(
        reaction_description="Knoevenagel condensation in ethanol for 1 h.",
        solvent="EtOH",
        temperature_C=40,
        reaction_time_h=1,
        concentration_M=0.2,
    )
    plan = ChemistryPlan(
        reaction_class="Knoevenagel condensation",
        mechanism_type="thermal",
    )
    council_proposal = FlowProposal(
        residence_time_min=12,
        flow_rate_mL_min=0.31,
        reactor_volume_mL=3.72,
        concentration_M=0.2,
        temperature_C=40,
        BPR_bar=2,
        tubing_ID_mm=0.75,
        tubing_material="PTFE",
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                solvent="EtOH",
                concentration_M=0.2,
                flow_rate_mL_min=0.31,
            )
        ],
    )

    final, calculations, report = finalize_design(
        council_proposal,
        batch_record=batch,
        chemistry_plan=plan,
        analogies=[],
        inventory=inventory,
    )

    available_reactors = {
        (reactor.volume_mL, reactor.ID_mm, reactor.material)
        for reactor in inventory.reactors
    }
    assert (
        final.reactor_volume_mL,
        final.tubing_ID_mm,
        final.tubing_material,
    ) in available_reactors
    assert report["status"] == "ready"
    assert report["checks"]["reactor_inventory_match"]
    assert report["checks"]["pump_flow_feasible"]
    assert report["checks"]["tubing_feasible"]
    assert report["checks"]["geometry_closure"]
    assert calculations.tubing_ID_mm == final.tubing_ID_mm
    assert abs(calculations.reactor_volume_mL - final.reactor_volume_mL) < 0.02


def test_finalizer_adds_missing_air_stream_and_both_residence_times():
    inventory = LabInventory.from_json(str(INVENTORY_PATH))
    batch = BatchRecord(
        reaction_description="The substrate solution was exposed to air for 2 h.",
        solvent="water",
        temperature_C=25,
        reaction_time_h=2,
        concentration_M=0.1,
    )
    plan = ChemistryPlan(
        reaction_class="aerobic oxidation",
        mechanism_type="oxidation",
    )
    council_proposal = FlowProposal(
        residence_time_min=20,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.1,
        reactor_volume_mL=2,
        concentration_M=0.1,
        temperature_C=25,
        BPR_bar=0,
        tubing_ID_mm=0.75,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                solvent="water",
                concentration_M=0.1,
                flow_rate_mL_min=0.1,
            )
        ],
    )

    final, _, report = finalize_design(
        council_proposal,
        batch_record=batch,
        chemistry_plan=plan,
        analogies=[],
        inventory=inventory,
    )

    gas = next(stream for stream in final.streams if stream.phase == "gas")
    assert gas.contents == ["air"]
    assert gas.gas_flow_sccm > 0
    assert gas.gas_flow_actual_mL_min > 0
    assert gas.molar_equiv > 0
    assert final.residence_time_inlet_min > 0
    assert final.residence_time_in_channel_min > 0
    assert report["checks"]["gas_bookkeeping_complete"]
