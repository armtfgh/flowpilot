import pytest

from flora_translate.component_identity import component_key, component_name, is_solvent_component, unique_components, protocol_component_quantity
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.design_realizer import _resolve_component_quantity_assumptions, _normalized_streams
from flora_translate.lightweight_upstream import _build_reagent_roles, LIGHTWEIGHT_CHEMISTRY_USER_TEMPLATE
from flora_translate.executable_artifacts import _stream_components, _stream_preparation_instruction
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, ReagentRole, StreamLogic, StreamAssignment


@pytest.mark.parametrize("role", ["solvent", "co-solvent", "cosolvent", "co solvent", "solvent / carrier"])
def test_medium_only_roles_do_not_demand_equivalents(role):
    assert is_solvent_component("ethanol", "EtOH:buffer (5:1)", role)
    components = _stream_components(
        {"chemistry_plan": {"reagents": [{"name": "ethanol", "role": role}]}},
        [{"stream_label": "A", "contents": ["ethanol"]}],
    )
    assert components[0].quantified
    assert not components[0].quantification_required


@pytest.mark.parametrize("role", ["co-solvent / base", "solvent and reagent", "substrate", "hydrogen donor"])
def test_reactive_medium_is_not_exempted_by_name_or_solvent_field(role):
    assert not is_solvent_component("ethanol", "ethanol", role)


def test_inline_dual_role_is_not_exempted():
    assert not is_solvent_component("water (solvent / reactant)", "water", "unknown")


@pytest.mark.parametrize("name", ["(((4-methoxyphenyl)thio)methyl)trimethylsilane", "[Ir(dF(CF3)ppy)2(dtbpy)]PF6", "copper(II)"])
def test_chemical_parentheses_are_preserved(name):
    assert component_name(name) == name
    assert component_name(name + " (0.5 mol%)") == name
    assert component_key(name) == component_key(name + " (0.5 mol%)")


def test_dedup_keeps_quantities_and_does_not_hide_conflicting_entries():
    assert unique_components(["acceptor", "acceptor (2 equiv)", "acceptor"]) == ["acceptor (2 equiv)"]
    assert unique_components(["acceptor (1 equiv)", "acceptor (2 equiv)"]) == ["acceptor (1 equiv)", "acceptor (2 equiv)"]


def test_reconciliation_does_not_add_unquantified_duplicates():
    batch = BatchRecord(raw_text="Substrate and acrylonitrile were charged together in ethanol.")
    plan = ChemistryPlan(
        reaction_name="Addition",
        reagents=[ReagentRole(name="substrate", role="substrate", equiv_or_loading="1 equiv"),
                  ReagentRole(name="acrylonitrile", role="acceptor", equiv_or_loading="2 equiv")],
        stream_logic=[StreamLogic(stream_label="A", reagents=["substrate (1 equiv, 0.1 M)", "acrylonitrile (2 equiv, 0.2 M)"])],
    )
    normalized, _ = reconcile_chemistry_plan(batch, plan)
    assert normalized.stream_logic[0].reagents == plan.stream_logic[0].reagents
    repeated, _ = reconcile_chemistry_plan(batch, normalized)
    assert repeated.stream_logic[0].reagents == normalized.stream_logic[0].reagents


def test_known_catalyst_loading_and_acceptor_equiv_are_not_overwritten():
    plan = ChemistryPlan(reaction_name="Addition", reagents=[
        ReagentRole(name="substrate", role="substrate", equiv_or_loading="1 equiv"),
        ReagentRole(name="acrylonitrile", role="acceptor", equiv_or_loading="2 equiv"),
        ReagentRole(name="[Ir(dF(CF3)ppy)2(dtbpy)]PF6", role="photocatalyst", equiv_or_loading="0.5 mol%"),
        ReagentRole(name="ethanol", role="co-solvent", equiv_or_loading="5 parts by volume"),
    ])
    proposal = FlowProposal(
        residence_time_min=10, flow_rate_mL_min=0.1, temperature_C=40,
        concentration_M=0.1, BPR_bar=3, reactor_type="coil", tubing_material="PFA",
        tubing_ID_mm=1, reactor_volume_mL=1,
        streams=[StreamAssignment(stream_label="A", pump_role="feed", concentration_M=0.1,
                                  contents=[item.name for item in plan.reagents])],
    )
    decisions = []
    _resolve_component_quantity_assumptions(proposal, decisions, plan)
    assert not decisions
    components = _stream_components({"chemistry_plan": plan.model_dump()}, [proposal.streams[0].model_dump()])
    assert [item.concentration_M for item in components] == [0.1, 0.2, 0.0005, None]
    assert all(item.quantified for item in components)


def test_screening_quantity_is_not_printed_as_a_dispensing_instruction():
    stream = {"stream_label": "A", "contents": ["substrate (0.1 M)", "buffer (0.1 M screening assumption)"], "phase": "liquid"}
    components = _stream_components({}, [stream])
    text = _stream_preparation_instruction(stream, components, parameters={"residence_time_min": 10})
    assert "DRAFT ONLY" in text
    assert "buffer: 1 mmol" not in text


def test_local_adapter_preserves_per_component_facts_and_medium_roles():
    batch = BatchRecord(
        raw_text="Substrate (0.2 mmol, 1.0 equiv), acceptor (0.4 mmol, 2.0 equiv) and Ir catalyst (0.5 mol%) were charged together.",
        photocatalyst="Ir catalyst", catalyst_loading_mol_pct=0.5,
        solvent="EtOH:pH 9 buffer (5:1, v/v)", concentration_M=0.1,
    )
    feed = StreamLogic(stream_label="A", reagents=["Substrate", "acceptor", "Ir catalyst", "EtOH", "pH 9 buffer"], concentration_M=0.1)
    roles = _build_reagent_roles(batch, {}, [feed], "")
    assert [item.equiv_or_loading for item in roles[:3]] == ["1.0 equiv", "2.0 equiv", "0.5 mol%"]
    assert all(item.role == "solvent" for item in roles[3:])
    plan, _ = reconcile_chemistry_plan(batch, ChemistryPlan(reagents=roles, stream_logic=[feed]))
    streams = _normalized_streams([], plan, batch.raw_text)
    assert streams[0].molar_equiv == 1.0
    components = _stream_components({"chemistry_plan": plan.model_dump()}, [item.model_dump() for item in streams])
    by_name = {item.name: item for item in components}
    assert by_name["Substrate"].concentration_M == 0.1
    assert by_name["acceptor"].concentration_M == 0.2
    assert by_name["Ir catalyst"].concentration_M == 0.0005
    assert by_name["EtOH"].quantification_required is False


def test_local_adapter_accepts_explicit_structured_roles_instead_of_discarding_them():
    roles = _build_reagent_roles(BatchRecord(), {"reagents": [{"name": "Z", "role": "base", "equiv_or_loading": "3 equiv"}]}, [StreamLogic(stream_label="A", reagents=["Z"])], "")
    assert roles[0].role == "base"
    assert roles[0].equiv_or_loading == "3 equiv"
    assert '"equiv_or_loading"' in LIGHTWEIGHT_CHEMISTRY_USER_TEMPLATE.format(batch_json="{}")


def test_protocol_quantity_does_not_match_chemical_suffixes_or_choose_conflicting_doses():
    assert protocol_component_quantity("ethanol", "methanol (2 equiv)") == ""
    assert protocol_component_quantity("X", "X (1 equiv). X (3 equiv).") == ""
