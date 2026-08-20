from flora_translate.intake_agent import IntakeAgent, QUESTION_BANK, intake_context_block
from flora_translate.schemas import DesignInputPackage, IntakeAnswer


def test_design_input_package_roundtrip():
    package = DesignInputPackage(
        raw_protocol="Substrate to product in MeCN, 0.1 M, 25 C, 4 h, 80% yield.",
        objective="first safe screening design",
        historical_data="none",
        inventory_constraints={"reactors": []},
        hypotheses=["kinetics may be slow"],
        operating_limits={"max_pressure_bar": 8},
        ready_for_design=True,
    )

    restored = DesignInputPackage.model_validate_json(package.model_dump_json())

    assert restored.schema_version == "flowpilot_intake_v1.0"
    assert restored.objective == "first safe screening design"
    assert restored.hypotheses == ["kinetics may be slow"]


def test_intake_agent_generates_stable_fixed_question_ids_without_llm():
    agent = IntakeAgent()
    protocol = "A to B in DMSO, 0.5 M, 40 C, 15 h, blue LED, O2."

    first = agent.analyze(protocol, use_llm=False)
    second = agent.analyze(protocol, use_llm=False)

    assert first.missing_question_ids == second.missing_question_ids
    assert set(first.missing_question_ids) <= set(QUESTION_BANK)
    assert first.missing_question_ids == [
        "Q-OBJ-001",
        "Q-CHEM-001",
        "Q-HIST-001",
        "Q-INV-001",
        "Q-CONSTR-001",
        "Q-HYP-001",
    ]
    assert not first.ready_for_design


def test_intake_agent_explicit_deterministic_mode_does_not_call_llm(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called by default intake analysis")

    import flora_translate.intake_agent as intake_agent

    monkeypatch.setattr(intake_agent, "call_model_text", fail_if_called)

    package = IntakeAgent().analyze("A to B in DMSO, 0.5 M, 40 C, 15 h.", use_llm=False)

    assert package.extracted_batch_fields["concentration_M"] == 0.5
    assert "Q-OBJ-001" in package.missing_question_ids


def test_intake_agent_blocks_until_required_sections_answered_or_unavailable():
    agent = IntakeAgent()
    answers = [
        IntakeAnswer(question_id="Q-OBJ-001", answer="next longer screening design"),
        IntakeAnswer(
            question_id="Q-CHEM-001",
            answer="Oxidation of A to B.",
        ),
        IntakeAnswer(question_id="Q-HIST-001", answer="Entry 1: tau 35 min, product 10%"),
        IntakeAnswer(question_id="Q-INV-001", status="unavailable"),
        IntakeAnswer(question_id="Q-CONSTR-001", status="unavailable"),
        IntakeAnswer(question_id="Q-HYP-001", answer="Longer residence time may be needed."),
    ]

    package = agent.analyze(
        "A to B in DMSO, 0.5 M, 40 C, 15 h.",
        answers=answers,
        use_llm=False,
    )

    assert package.ready_for_design
    assert package.missing_question_ids == []
    assert package.objective == "next longer screening design"
    assert package.inventory_constraints is None
    assert package.hypotheses == ["Longer residence time may be needed."]


def test_intake_requires_chemist_identity_when_protocol_family_is_ambiguous():
    agent = IntakeAgent()
    package = agent.analyze(
        "Compound A was treated with reagent B and afforded compound C.",
        use_llm=False,
    )

    assert "Q-CHEM-001" in package.missing_question_ids
    assert not package.chemistry_identity_confirmation


def test_protocol_stated_identity_is_frozen_without_extra_llm_authority():
    agent = IntakeAgent()
    package = agent.analyze(
        "Hydrogenolysis/debenzylation of protected amine A afforded amine B.",
        use_llm=False,
    )

    assert "Q-CHEM-001" not in package.missing_question_ids
    assert package.chemistry_identity_confirmation == {
        "transformation_family": "hydrogenolysis",
        "confirmed": True,
        "source": "protocol_fact",
    }


def test_intake_context_block_preserves_authority_labels():
    package = DesignInputPackage(
        raw_protocol="A to B.",
        objective="screen",
        historical_data="Measured: 10% product at 35 min.",
        inventory_constraints={"BPR_available": [6]},
        hypotheses=["photon limitation"],
        operating_limits="max pressure 8 bar",
        ready_for_design=True,
    )

    block = intake_context_block(package)

    assert "measured evidence > hard constraints > protocol facts" in block
    assert "Measured: 10% product" in block
    assert "photon limitation" in block
