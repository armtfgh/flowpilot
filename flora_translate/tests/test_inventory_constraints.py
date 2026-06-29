from pathlib import Path

from flora_translate.engine.design_space import DesignSpaceSearch
from flora_translate.inventory_constraints import enforce_reactor_inventory
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory, StreamAssignment


THQ_INVENTORY = Path("flora_translate/data/lab_inventory_thq.json")


class _StubCalc:
    is_gas_liquid = True
    residence_time_min = 67.68
    tau_analogy_min = None
    tau_class_min = None
    tau_kinetics_min = 67.68
    rate_constant = None
    intensification_factor = 13.3
    concentration_M = 0.5
    bpr_pressure_bar = 6.0
    extinction_coefficient_M_cm = None


def _thq_batch() -> BatchRecord:
    return BatchRecord(
        reaction_description="Photochemical aerobic oxidation of tetrahydroquinoline to quinoline with O2.",
        solvent="DMSO",
        temperature_C=40,
        reaction_time_h=15,
        concentration_M=0.5,
        scale_mmol=0.2,
        atmosphere="O2",
        wavelength_nm=450,
    )


def _thq_plan() -> ChemistryPlan:
    return ChemistryPlan(
        reaction_class="photochemical aerobic oxidation",
        mechanism_type="O2-mediated photochemical oxidation",
        o2_is_reagent=True,
        recommended_wavelength_nm=450,
    )


def test_design_space_uses_only_inventory_reactor_volumes_for_thq():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    candidates = DesignSpaceSearch().run(
        batch_record=_thq_batch(),
        chemistry_plan=_thq_plan(),
        calculations=_StubCalc(),
        inventory=inventory,
        reaction_class="photochemical aerobic oxidation",
    )

    assert candidates
    volumes = {round(c.V_R_mL) for c in candidates}
    assert volumes <= {10, 15, 20}
    assert {round(c.d_mm, 3) for c in candidates} == {1.0}
    assert all(c.inventory_reactor_name for c in candidates)


def test_inventory_enforcement_snaps_krict_like_design_to_available_reactor():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    proposal = FlowProposal(
        residence_time_min=67.68,
        residence_time_in_channel_min=67.68,
        residence_time_inlet_min=15.277,
        residence_time_basis="in-channel total actual flow",
        flow_rate_mL_min=0.02835,
        reactor_volume_mL=10.74,
        temperature_C=40,
        concentration_M=0.5,
        BPR_bar=6,
        tubing_ID_mm=0.75,
        tubing_material="FEP",
        wavelength_nm=450,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["6-methyl-THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.02835,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                flow_rate_mL_min=0.13034,
                gas_flow_actual_mL_min=0.13034,
                gas_flow_sccm=0.67469,
            ),
        ],
    )

    revised, report = enforce_reactor_inventory(proposal, inventory)

    assert report["applied"]
    assert revised.reactor_volume_mL == 10.0
    assert revised.tubing_ID_mm == 1.0
    assert revised.inventory_selection["system"] == "Vapourtec System"
    assert revised.residence_time_in_channel_min == 67.68
    assert revised.flow_rate_mL_min < proposal.flow_rate_mL_min
    assert revised.streams[1].gas_flow_sccm is not None
    assert revised.residence_time_inlet_min is not None
