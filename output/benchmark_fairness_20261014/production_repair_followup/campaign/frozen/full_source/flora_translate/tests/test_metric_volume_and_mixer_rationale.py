"""Metric normalization and canonical mixer rationale regression tests."""
import pytest

from flora_translate import design_calculator as metrics
from flora_translate import main
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal


def calculate(*, q, c, x, volume, tau):
    calc = metrics.DesignCalculations(flow_rate_mL_min=q, concentration_M=c,
        target_conversion=x, reactor_volume_mL=volume, residence_time_min=tau,
        residence_time_s=tau * 60, batch_time_s=300)
    metrics.DesignCalculator()._step9(calc, BatchRecord(reaction_time_h=5 / 60))
    return calc


@pytest.mark.parametrize("time_index", [0.1713396, 0.5125876, 2.0])
def test_photo_sty_is_invariant_to_gas_volume_time_reporting_basis(time_index):
    calc = calculate(q=.5, c=.1, x=.99, volume=1, tau=time_index)
    assert calc.productivity_mmol_h == pytest.approx(2.970)
    assert calc.space_time_yield_mol_L_h == pytest.approx(2.970)
    assert calc.steps[-1].values["STY_volume_basis"] == "total nominal reactor volume"
    assert "conditional projections" in calc.steps[-1].assumptions[0]


def test_hydrogenolysis_sty_uses_total_bed_volume_and_same_conversion():
    calc = calculate(q=.5, c=.126, x=.381, volume=3, tau=1.569036)
    assert calc.productivity_mmol_h == pytest.approx(1.440)
    assert calc.space_time_yield_mol_L_h == pytest.approx(.4801)


def test_liquid_only_cuaac_value_is_preserved():
    calc = calculate(q=.125, c=.25, x=.94, volume=.3, tau=2.4)
    assert calc.space_time_yield_mol_L_h == pytest.approx(5.875)


def test_metric_does_not_invent_reactor_volume_or_measured_conversion():
    calc = calculate(q=.5, c=.1, x=.5, volume=0, tau=2)
    assert calc.productivity_mmol_h == pytest.approx(1.5)
    assert calc.space_time_yield_mol_L_h is None
    assert calc.steps[-1].values["conversion_basis"] == "assumed target_conversion"


def test_geometry_changes_sty_at_fixed_feed_and_assumed_conversion():
    one = calculate(q=.5, c=.1, x=.99, volume=1, tau=2)
    two = calculate(q=.5, c=.1, x=.99, volume=2, tau=2)
    assert one.productivity_mmol_h == two.productivity_mmol_h
    assert one.space_time_yield_mol_L_h == pytest.approx(2 * two.space_time_yield_mol_L_h)


def test_mixer_rationale_cannot_publish_unchecked_downstream_order():
    bad = "Place the BPR downstream of the vented gas-liquid separator."
    proposal = FlowProposal(reactor_type="packed-bed", reactor_volume_mL=3,
        flow_rate_mL_min=.5, concentration_M=.126, temperature_C=60,
        BPR_bar=21, tubing_ID_mm=4.35, tubing_material="stainless steel",
        mixing_order_reasoning=bad,
        streams=[dict(stream_label="A", contents=["substrate"], solvent="methanol",
                            flow_rate_mL_min=.5, concentration_M=.126, phase="liquid"),
                 dict(stream_label="B", contents=["H2"], phase="gas", gas_flow_sccm=1.4,
                            gas_flow_actual_mL_min=.08, flow_rate_mL_min=.08)])
    topology = main._build_singlestep_topology(proposal, ChemistryPlan(), BatchRecord())
    mixer = next(op for op in topology.unit_operations if op.op_id == "mixer_1")
    assert "final process diagram" in mixer.rationale
    assert bad not in mixer.rationale
    assert proposal.mixing_order_reasoning == bad  # Original reasoning remains archived.
