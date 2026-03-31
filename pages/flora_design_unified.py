"""FLORA Design — unified batch-to-flow translation and process design page."""

from pathlib import Path
import streamlit as st


def render():
    st.title("FLORA Design")
    st.markdown(
        "Describe your chemistry — FLORA will design a validated flow process "
        "grounded in literature. Works from a batch protocol or a free-text goal."
    )

    # ── Mode selector ──────────────────────────────────────────────────────
    mode = st.radio(
        "What do you have?",
        ["I have a batch protocol to convert",
         "I have a chemistry goal (no batch data)"],
        horizontal=True,
        key="design_mode",
    )

    if mode.startswith("I have a batch"):
        _render_translate()
    else:
        _render_design()


# ─────────────────────────────────────────────────────────────────────────────
# BATCH-TO-FLOW mode
# ─────────────────────────────────────────────────────────────────────────────

def _render_translate():
    st.markdown("---")
    st.subheader("Batch Protocol")

    input_mode = st.radio("Input format", ["Free text", "Structured form"],
                          horizontal=True, key="t_input_mode")
    batch_input = None

    if input_mode == "Free text":
        txt = st.text_area(
            "Paste your batch protocol",
            height=160,
            key="t_free_text",
            placeholder=(
                "e.g. fac-Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative "
                "radical addition of N-Boc-proline (1.0 equiv) to methyl vinyl "
                "ketone (2.0 equiv), K2HPO4 (1.5 equiv), DMF, 0.1 M, RT, "
                "450 nm blue LED, N2, 24 h, 72% yield."
            ),
        )
        if txt:
            batch_input = txt

    else:
        c1, c2 = st.columns(2)
        with c1:
            desc   = st.text_input("Reaction description", key="t_desc")
            cat    = st.text_input("Photocatalyst", key="t_cat",
                                   placeholder="e.g. Ir(ppy)3")
            base   = st.text_input("Base", key="t_base",
                                   placeholder="e.g. K2HPO4")
            solv   = st.text_input("Solvent", key="t_solv",
                                   placeholder="e.g. DMF")
            atm    = st.selectbox("Atmosphere", ["N2", "Ar", "air"], key="t_atm")
        with c2:
            temp   = st.number_input("Temperature (°C)", value=25.0, key="t_temp")
            time_h = st.number_input("Reaction time (h)", value=0.0, step=0.5, key="t_time")
            conc   = st.number_input("Concentration (M)", value=0.0, step=0.01, key="t_conc")
            wl     = st.number_input("Wavelength (nm)", value=450, step=10, key="t_wl")
            yld    = st.number_input("Yield (%)", value=0.0, max_value=100.0, key="t_yield")
        if desc:
            batch_input = {
                "reaction_description": desc,
                "photocatalyst": cat or None,
                "base": base or None,
                "solvent": solv or None,
                "temperature_C": temp,
                "reaction_time_h": time_h or None,
                "concentration_M": conc or None,
                "wavelength_nm": wl or None,
                "yield_pct": yld or None,
                "atmosphere": atm,
            }

    if st.button("Translate to Flow", type="primary",
                 disabled=batch_input is None,
                 use_container_width=True, key="t_run"):
        with st.spinner("Running FLORA pipeline…"):
            try:
                from flora_translate.main import translate
                result = translate(batch_input)
                st.session_state["flora_result"] = result
                st.session_state["flora_result_type"] = "translate"
            except Exception as e:
                from components.error_card import render_error
                render_error(e, "FLORA-Translate")
                return

    if st.session_state.get("flora_result_type") == "translate":
        _render_result(st.session_state["flora_result"])


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN-FROM-GOAL mode
# ─────────────────────────────────────────────────────────────────────────────

def _render_design():
    st.markdown("---")
    st.subheader("Chemistry Goal")

    goal = st.text_area(
        "Describe what you want to do",
        height=140,
        key="d_goal",
        placeholder=(
            "e.g. Design a flow process for Ir(ppy)3-catalyzed photoredox "
            "radical addition of alpha-amino acids to electron-poor alkenes "
            "in MeCN at room temperature, ~0.5 mL/min scale."
        ),
    )

    if st.button("Design Flow Process", type="primary",
                 disabled=not (goal or "").strip(),
                 use_container_width=True, key="d_run"):
        with st.spinner("Designing your flow process…"):
            try:
                from flora_design.main import design
                result = design(goal)
                st.session_state["flora_design_result"] = result
                st.session_state["flora_result_type"] = "design"
            except Exception as e:
                from components.error_card import render_error
                render_error(e, "FLORA-Design")
                return

    if st.session_state.get("flora_result_type") == "design":
        _render_design_result(st.session_state["flora_design_result"])


# ─────────────────────────────────────────────────────────────────────────────
# Shared result renderer (translate output)
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(result: dict):
    proposal = result.get("proposal", {})
    conf = result.get("confidence", "LOW")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    tabs = st.tabs([
        "Summary",
        "Process Diagram",
        "Chemistry Plan",
        "Stream Assignments",
        "Conditions",
        "Engineering",
        "Raw JSON",
    ])

    with tabs[0]:
        st.markdown(result.get("explanation", ""))
        notes = proposal.get("chemistry_notes", "")
        if notes:
            st.divider()
            st.info(notes)

    with tabs[1]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(result.get("svg_path", ""), result.get("png_path", ""))
        topo = result.get("process_topology", {})
        if topo:
            st.divider()
            for i, op in enumerate(topo.get("unit_operations", []), 1):
                if op.get("op_type") == "led_module":
                    continue
                with st.expander(f"{i}. {op.get('label', '?')}"):
                    p = op.get("parameters", {})
                    for k, v in p.items():
                        if v is not None and k not in ("light_required",):
                            st.markdown(f"**{k.replace('_', ' ')}:** {v}")
                    if op.get("rationale"):
                        st.caption(op["rationale"])
            if topo.get("pid_description"):
                st.code(topo["pid_description"])

    with tabs[2]:
        from pages.translate import _render_chemistry_plan
        _render_chemistry_plan(result.get("chemistry_plan", {}))

    with tabs[3]:
        from pages.translate import _render_streams
        _render_streams(proposal)

    with tabs[4]:
        _render_conditions(proposal)

    with tabs[5]:
        from components.council_report import render_council_report
        class _C:
            def __init__(self, r):
                self.council_rounds = r.get("council_rounds", 0)
                self.safety_report  = r.get("safety_report", {})
                self.council_messages = r.get("council_messages", [])
        render_council_report(_C(result))

    with tabs[6]:
        st.json(result)

    from components.feedback import render_feedback_widget
    render_feedback_widget(result, context="flora_design_translate")


def _render_design_result(result):
    """Render output from FLORA-Design (from-goal mode)."""
    topo = result.topology
    conf = topo.topology_confidence
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    feats = result.chem_features
    st.info(
        f"**{feats.reaction_class.replace('_', ' ').title()}** · "
        f"Catalyst: {feats.photocatalyst or '?'} · "
        f"λ = {feats.wavelength_nm or '?'} nm"
    )

    tabs = st.tabs(["Summary", "Process Diagram", "Engineering", "Raw JSON"])

    with tabs[0]:
        st.markdown(result.explanation)
        c1, c2, c3 = st.columns(3)
        c1.metric("Residence time", f"{topo.residence_time_min:.1f} min")
        c2.metric("Flow rate", f"{topo.total_flow_rate_mL_min:.2f} mL/min")
        c3.metric("Reactor volume", f"{topo.reactor_volume_mL:.1f} mL")
        st.divider()
        st.code(topo.pid_description)
        for i, op in enumerate(topo.unit_operations, 1):
            with st.expander(f"{i}. {op.label}"):
                st.json(op.parameters)
                st.caption(op.rationale)

    with tabs[1]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(result.svg_path, result.png_path)

    with tabs[2]:
        dc = result.design_candidate
        if dc:
            from components.council_report import render_council_report
            render_council_report(dc)

    with tabs[3]:
        st.json(result.model_dump(exclude_none=True))

    fb_result = {"proposal": {
        "residence_time_min": topo.residence_time_min,
        "flow_rate_mL_min": topo.total_flow_rate_mL_min,
        "confidence": topo.topology_confidence,
    }}
    from components.feedback import render_feedback_widget
    render_feedback_widget(fb_result, context="flora_design_goal")


def _render_conditions(proposal: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Residence time", f"{proposal.get('residence_time_min', 0):.1f} min")
        st.metric("Flow rate", f"{proposal.get('flow_rate_mL_min', 0):.2f} mL/min")
        st.metric("Temperature", f"{proposal.get('temperature_C', 25):.0f} °C")
        st.metric("Concentration", f"{proposal.get('concentration_M', 0):.3f} M")
    with c2:
        st.metric("Reactor", proposal.get("reactor_type", "N/A"))
        st.metric("Tubing", proposal.get("tubing_material", "N/A"))
        st.metric("Tubing ID", f"{proposal.get('tubing_ID_mm', 0):.1f} mm")
        st.metric("Volume", f"{proposal.get('reactor_volume_mL', 0):.1f} mL")
    with c3:
        st.metric("BPR", f"{proposal.get('BPR_bar', 0):.0f} bar")
        st.metric("Wavelength", f"{proposal.get('wavelength_nm', 'N/A')} nm")
        st.metric("Deoxygenation", proposal.get("deoxygenation_method", "N/A"))

    reasoning = proposal.get("reasoning_per_field", {})
    if reasoning:
        st.divider()
        st.markdown("**Reasoning per field**")
        for field, reason in reasoning.items():
            st.markdown(f"- **{field}:** {reason}")
