"""Stock quantities and operating instructions must close with the final schedule."""

from copy import deepcopy

import pytest

from flora_translate.executable_artifacts import (
    _feed_stock_requirement,
    _initial_run_schedule,
    _operating_procedure,
    _safety_requirements,
    _stage_liquid_flow,
    _stream_preparation_instruction,
)
from flora_translate.final_design_contract import build_final_design_contract
from flora_translate.schemas import CanonicalStreamComponent, ExecutableProcessGraph
from flora_translate.tests.test_executable_artifact_contract import _hydrogen_result


def test_stock_covers_both_actual_commissioning_and_collection_intervals():
    result = _hydrogen_result()
    original = deepcopy(result)
    contract = build_final_design_contract(result)
    steps = {step["step_id"]: step for step in contract["operating_procedure"]}
    prep = steps["PREP-A"]["parameter_bindings"]
    stabilization = steps["STEADY-STATE"]["parameter_bindings"]["minimum_initial_stabilization_time_min"]
    collection = steps["COLLECT-PRODUCT"]["parameter_bindings"]["minimum_initial_collection_time_min"]
    assert stabilization == pytest.approx(60)
    assert collection == pytest.approx(20)
    assert prep["minimum_feed_stock_volume_mL"] == pytest.approx(
        prep["flow_rate_mL_min"] * (stabilization + collection)
    ) == pytest.approx(12)
    assert "1.512 mmol (0.126 M final)" in steps["PREP-A"]["instruction"]
    assert "12 mL final-volume basis" in steps["PREP-A"]["instruction"]
    assert prep["additional_feed_volume_mL"] is None
    assert prep["additional_feed_volume_requires_measurement"] is True
    assert prep["stock_availability_confirmed"] is False
    assert "no dead-volume allowance or available stock has been assumed" in steps["PREP-A"]["instruction"]
    assert "minimum_feed_stock_volume_mL" not in steps["PREP-G"]["parameter_bindings"]
    assert result == original


@pytest.mark.parametrize("flow", [0.01, 0.16, 0.5, 2.0])
def test_each_feed_uses_its_own_flow_and_preserves_multicomponent_ratios(flow):
    schedule = {"minimum_scheduled_feed_duration_min": 24}
    requirement = _feed_stock_requirement(flow, schedule)
    components = [
        CanonicalStreamComponent(stream_label="A", name="acid", concentration_M=0.5),
        CanonicalStreamComponent(stream_label="A", name="activator", concentration_M=0.525),
        CanonicalStreamComponent(stream_label="A", name="catalyst", concentration_M=0.05),
    ]
    recipe = _stream_preparation_instruction(
        {"stream_label": "A", "phase": "liquid", "solvent": "2-MeTHF"},
        components,
        parameters={},
        stock_requirement=requirement,
    )
    expected_mL = 24 * flow
    assert requirement["recipe_basis_volume_mL"] == pytest.approx(expected_mL)
    for component in components:
        assert f"{component.name}: {component.concentration_M * expected_mL:g} mmol" in recipe
    assert "Scale every solute amount and final solvent volume by the same factor" in recipe
    assert "Provide priming and shutdown solvent separately" in recipe


def test_larger_declared_time_and_liquid_turnover_time_both_bound_schedule():
    contract = build_final_design_contract(_hydrogen_result())
    graph = ExecutableProcessGraph.model_validate(contract["process_graph"])
    # A gas/STP index is shorter than the liquid-volume index here.
    assert _initial_run_schedule(graph, {"residence_time_min": 0.6452})["timing_reference_min"] == pytest.approx(20)
    # A separately justified longer total stage time must not be shortened.
    assert _initial_run_schedule(graph, {"residence_time_min": 30})["minimum_scheduled_feed_duration_min"] == pytest.approx(120)


def test_staged_addition_uses_local_stage_flows_not_final_outlet_flow():
    contract = build_final_design_contract(_hydrogen_result())
    graph = ExecutableProcessGraph.model_validate(contract["process_graph"])
    first = graph.stages[0]
    reactor = next(op for op in graph.topology.unit_operations if op.op_id == first.reactor_operation_id)
    reactor.parameters["Q_liquid_mL_min"] = 0.25
    second = first.model_copy(update={"stage_number": 2, "reactor_operation_id": "reactor_2"})
    graph.stages.append(second)
    graph.topology.unit_operations.append(reactor.model_copy(update={
        "op_id": "reactor_2", "parameters": {"Q_liquid_mL_min": 0.5}
    }))
    graph.total_liquid_flow_mL_min = 0.5
    schedule = _initial_run_schedule(graph, {"residence_time_min": 1.0})
    assert schedule["nominal_stage_liquid_volume_time_sum_min"] == pytest.approx(3 / 0.25 + 3 / 0.5)
    assert schedule["minimum_scheduled_feed_duration_min"] == pytest.approx(72)
    assert _feed_stock_requirement(0.25, schedule)["minimum_feed_stock_volume_mL"] == pytest.approx(18)


def test_legacy_local_flow_can_be_reconstructed_but_unresolved_split_is_rejected():
    contract = build_final_design_contract(_hydrogen_result())
    graph = ExecutableProcessGraph.model_validate(contract["process_graph"])
    reactor_id = graph.stages[0].reactor_operation_id
    reactor = next(op for op in graph.topology.unit_operations if op.op_id == reactor_id)
    reactor.parameters.pop("Q_liquid_mL_min")
    assert _stage_liquid_flow(graph, reactor_id) == pytest.approx(0.15)
    edge = next(edge for edge in graph.topology.streams if edge.from_op == graph.feeds[0].operation_id)
    graph.topology.streams.append(edge.model_copy(update={"stream_id": "bypass", "to_op": "collector_1"}))
    with pytest.raises(ValueError, match="resolved split allocation"):
        _stage_liquid_flow(graph, reactor_id)


def test_unresolved_procedure_is_reported_as_blocked_without_numeric_repair(monkeypatch):
    import flora_translate.final_design_contract as compiler

    message = "Local liquid flow for reactor R2 requires a resolved split allocation"

    def fail_compile(*args, **kwargs):
        raise ValueError(message)

    monkeypatch.setattr(compiler, "compile_executable_artifacts", fail_compile)
    contract = compiler.build_final_design_contract(_hydrogen_result())
    assert contract["status"] == "blocked"
    assert contract["consistency"]["passed"] is False
    assert contract["operating_procedure"] == []
    assert contract["diagnostic"]["procedure_compilation_error"] == {
        "type": "ValueError", "message": message
    }
    assert {"code": "FINAL-PROCEDURE-COMPILATION-FAILED", "message": message} in contract["consistency"]["issues"]


def test_specific_shutdown_controls_survive_and_follow_product_collection():
    contract = build_final_design_contract(_hydrogen_result())
    steps = contract["operating_procedure"]
    by_id = {step["step_id"]: step for step in steps}
    order = {step["step_id"]: i for i, step in enumerate(steps)}
    assert len(by_id) == len(steps)
    assert order["COLLECT-PRODUCT"] < order["H2-SHUTDOWN"] < order["NORMAL-SHUTDOWN"] < order["PRESSURE-DEPRESSURIZE"]
    assert "Isolate hydrogen first" in by_id["H2-SHUTDOWN"]["instruction"]
    assert "re-establish the inventory-declared inert purge" in by_id["H2-SHUTDOWN"]["instruction"]
    assert "verify zero pressure before opening any fitting" in by_id["PRESSURE-DEPRESSURIZE"]["instruction"]
    assert "On leak, blockage, pressure excursion, or temperature excursion" in by_id["GENERAL-EMERGENCY-STOP"]["instruction"]
    assert "do not combine incompatible waste streams" in by_id["GENERAL-WASTE"]["instruction"]


def test_photochemical_schedule_starts_after_light_and_light_stops_before_flush():
    result = _hydrogen_result()
    contract = build_final_design_contract(result)
    graph = ExecutableProcessGraph.model_validate(contract["process_graph"])
    params = {**contract["parameters"], "wavelength_nm": 450.0}
    hazards = ["high_intensity_light"]
    steps = _operating_procedure(
        result=result,
        process_graph=graph,
        parameters=params,
        streams=contract["streams"],
        components=[CanonicalStreamComponent.model_validate(c) for c in contract["stream_components"]],
        hazards=hazards,
        requirements=_safety_requirements(hazards, params),
    )
    by_id = {step.step_id: step for step in steps}
    order = {step.step_id: i for i, step in enumerate(steps)}
    assert order["START-FEEDS"] < order["PHOTO-LIGHT-ON"] < order["STEADY-STATE"]
    assert order["COLLECT-PRODUCT"] < order["PHOTO-LIGHT-OFF"] < order["NORMAL-SHUTDOWN"]
    assert by_id["PHOTO-LIGHT-ON"].parameter_bindings["wavelength_nm"] == 450.0
    assert "before flushing" in by_id["PHOTO-LIGHT-OFF"].instruction
