"""Streamlit wizard for FlowPilot standardized intake."""

from __future__ import annotations

import json

import streamlit as st

from components.inventory_selector import render_inventory_selector
from flora_translate.intake_agent import IntakeAgent
from flora_translate.inventory_profiles import InventoryProfile
from flora_translate.schemas import DesignInputPackage, IntakeAnswer


def render_intake_wizard(key_prefix: str = "flowpilot_intake") -> DesignInputPackage | None:
    """Render the blocking intake wizard and return the current package."""

    agent = IntakeAgent()
    package_key = f"{key_prefix}_package"

    st.subheader("Standardized Intake")
    st.caption("Complete the reproducible intake package before running design.")

    inventory_profile = render_inventory_selector(f"{key_prefix}_inventory")

    c1, c2 = st.columns([4, 1])
    with c1:
        use_llm = st.checkbox(
            "Use LLM field extraction",
            value=True,
            key=f"{key_prefix}_use_llm_v3",
            help=(
                "The LLM extracts stated protocol fields; fixed question IDs "
                "and readiness logic remain deterministic."
            ),
        )
        st.caption("Question IDs and blocking readiness are deterministic and reproducible.")
    with c2:
        if st.button("Reset intake", key=f"{key_prefix}_reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                if str(key).startswith(key_prefix):
                    del st.session_state[key]
            st.rerun()

    protocol = st.text_area(
        "Initial batch protocol",
        height=180,
        key=f"{key_prefix}_protocol",
        placeholder=(
            "Paste the batch protocol here. The intake agent will extract stated "
            "facts and ask standardized follow-up questions."
        ),
    )

    if st.button(
        "Analyze Intake",
        type="primary",
        disabled=not bool(protocol.strip()),
        key=f"{key_prefix}_analyze",
        use_container_width=True,
    ):
        with st.spinner("Analyzing intake..."):
            package = agent.analyze(
                protocol,
                existing_package=st.session_state.get(package_key),
                use_llm=use_llm,
            )
            if inventory_profile is not None:
                package = _apply_inventory_profile(
                    agent,
                    protocol,
                    package,
                    inventory_profile,
                )
            st.session_state[package_key] = package.model_dump()
        st.rerun()

    package = _load_package(st.session_state.get(package_key))
    if package is None:
        st.info("Start by analyzing the batch protocol.")
        return None

    if inventory_profile is not None and _profile_changed(package, inventory_profile):
        package = _apply_inventory_profile(
            agent,
            protocol,
            package,
            inventory_profile,
        )
        st.session_state[package_key] = package.model_dump()

    pending = agent.pending_questions(package)
    if pending:
        st.markdown("#### Missing standardized inputs")
        new_answers: list[IntakeAnswer] = []
        for question in pending:
            with st.expander(
                f"{question.question_id} - {question.section}",
                expanded=True,
            ):
                st.markdown(question.question)
                if question.why_needed:
                    st.caption(question.why_needed)
                if question.expected_format:
                    st.caption(f"Expected format: {question.expected_format}")

                allow_unavailable = question.question_id not in {
                    "Q-BATCH-001", "Q-OBJ-001", "Q-CHEM-001"
                }
                unavailable = False
                if allow_unavailable:
                    unavailable = st.checkbox(
                        "Mark unavailable",
                        key=f"{key_prefix}_{question.question_id}_unavailable",
                    )
                answer_text = st.text_area(
                    "Answer",
                    height=90,
                    disabled=unavailable,
                    key=f"{key_prefix}_{question.question_id}_answer",
                )
                if unavailable:
                    new_answers.append(
                        IntakeAnswer(question_id=question.question_id, status="unavailable")
                    )
                elif answer_text.strip():
                    new_answers.append(
                        IntakeAnswer(
                            question_id=question.question_id,
                            answer=answer_text.strip(),
                            status="answered",
                        )
                    )

        if st.button(
            "Save Intake Answers",
            disabled=not bool(new_answers),
            key=f"{key_prefix}_save_answers",
            use_container_width=True,
        ):
            st.session_state[package_key] = agent.analyze(
                protocol,
                existing_package=package,
                answers=new_answers,
                use_llm=False,
            ).model_dump()
            st.rerun()

    package = _load_package(st.session_state.get(package_key))
    if package is None:
        return None

    if package.ready_for_design:
        st.success("Intake package is ready for design.")
    else:
        st.warning(
            "Design is blocked until these question IDs are answered or marked unavailable: "
            + ", ".join(package.missing_question_ids)
        )

    with st.expander("Frozen DesignInputPackage", expanded=False):
        st.json(package.model_dump())

    return package


def _load_package(value) -> DesignInputPackage | None:
    if value is None:
        return None
    if isinstance(value, DesignInputPackage):
        return value
    return DesignInputPackage.model_validate(value)


def _apply_inventory_profile(
    agent: IntakeAgent,
    protocol: str,
    package: DesignInputPackage,
    profile: InventoryProfile,
) -> DesignInputPackage:
    answers = [
        IntakeAnswer(
            question_id="Q-INV-001",
            answer=profile.lab_inventory.model_dump(),
            status="answered",
            source="inventory_profile",
        ),
        IntakeAnswer(
            question_id="Q-CONSTR-001",
            answer=profile.design_operating_limits(),
            status="answered",
            source="inventory_profile",
        ),
    ]
    updated = agent.analyze(
        protocol,
        existing_package=package,
        answers=answers,
        use_llm=False,
    )
    updated.inventory_profile_snapshot = profile.model_dump()
    return updated


def _profile_changed(
    package: DesignInputPackage,
    profile: InventoryProfile,
) -> bool:
    current = package.inventory_profile_snapshot or {}
    return json.dumps(current, sort_keys=True, default=str) != json.dumps(
        profile.model_dump(), sort_keys=True, default=str
    )
