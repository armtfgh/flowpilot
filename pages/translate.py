"""FLORA — Batch-to-Flow translation page."""

import json
from pathlib import Path

import streamlit as st


def render():
    st.title("Batch-to-Flow")
    st.markdown(
        "Paste a batch synthesis protocol and FLORA will propose "
        "a literature-grounded flow chemistry equivalent with full "
        "process design, engineering validation, and flow diagram."
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_mode = st.radio("Input mode", ["Free Text", "Structured Form"], horizontal=True)
    batch_input = None

    if input_mode == "Free Text":
        batch_text = st.text_area(
            "Batch protocol",
            height=180,
            placeholder=(
                "Example: fac-Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative "
                "radical addition of N-Boc-proline (1.0 equiv) to methyl vinyl "
                "ketone (2.0 equiv), K2HPO4 (1.5 equiv), DMF, 0.1 M, RT, "
                "450 nm blue LED, N2 atmosphere, 24 hours, 72% yield."
            ),
        )
        if batch_text:
            batch_input = batch_text
    else:
        c1, c2 = st.columns(2)
        with c1:
            reaction_desc = st.text_input("Reaction description")
            photocatalyst = st.text_input("Photocatalyst", placeholder="e.g. Ir(ppy)3")
            cat_loading = st.number_input("Catalyst loading (mol%)", min_value=0.0, value=0.0, step=0.1)
            base = st.text_input("Base", placeholder="e.g. K2HPO4")
            solvent = st.text_input("Solvent", placeholder="e.g. DMF")
            atmosphere = st.selectbox("Atmosphere", ["N2", "Ar", "air", "O2"])
        with c2:
            temperature = st.number_input("Temperature (°C)", value=25.0, step=1.0)
            reaction_time = st.number_input("Reaction time (h)", min_value=0.0, value=0.0, step=0.5)
            concentration = st.number_input("Concentration (M)", min_value=0.0, value=0.0, step=0.01)
            scale = st.number_input("Scale (mmol)", min_value=0.0, value=0.0, step=0.1)
            yield_pct = st.number_input("Yield (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            wavelength = st.number_input("Wavelength (nm)", min_value=0, value=450, step=10)
            light_source = st.text_input("Light source", placeholder="e.g. 450 nm blue LED")

        if reaction_desc:
            batch_input = {
                "reaction_description": reaction_desc,
                "photocatalyst": photocatalyst or None,
                "catalyst_loading_mol_pct": cat_loading or None,
                "base": base or None,
                "solvent": solvent or None,
                "temperature_C": temperature,
                "reaction_time_h": reaction_time or None,
                "concentration_M": concentration or None,
                "scale_mmol": scale or None,
                "yield_pct": yield_pct or None,
                "wavelength_nm": wavelength or None,
                "light_source": light_source or None,
                "atmosphere": atmosphere,
            }

    # ── Run ────────────────────────────────────────────────────────────────────
    if st.button("Translate to Flow", type="primary", disabled=batch_input is None,
                 use_container_width=True):
        with st.spinner("Running FLORA-Translate pipeline..."):
            try:
                from flora_translate.main import translate
                result = translate(batch_input)
                st.session_state["translate_result"] = result
            except Exception as e:
                from components.error_card import render_error
                render_error(e, "FLORA-Translate")
                return

    # ── Display ────────────────────────────────────────────────────────────────
    if "translate_result" not in st.session_state:
        _show_example()
        return

    result = st.session_state["translate_result"]
    proposal = result.get("proposal", {})

    # Confidence badge
    conf = result.get("confidence", "LOW")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Summary",
        "Process Diagram",
        "Chemistry Plan",
        "Stream Assignments",
        "Flow Conditions",
        "Engineering Report",
        "Raw JSON",
    ])

    # ── TAB 0: Summary ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(result.get("explanation", ""))
        chem_notes = proposal.get("chemistry_notes", "")
        if chem_notes:
            st.divider()
            st.markdown(f"**Chemistry Notes:** {chem_notes}")

    # ── TAB 1: Process Diagram ─────────────────────────────────────────────────
    with tabs[1]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(
            result.get("svg_path", ""), result.get("png_path", "")
        )
        # Topology details
        topo = result.get("process_topology", {})
        if topo:
            st.divider()
            st.subheader("Unit Operations")
            for i, op in enumerate(topo.get("unit_operations", []), 1):
                if op.get("op_type") == "led_module":
                    continue
                with st.expander(f"{i}. {op.get('label', '?')}"):
                    p = op.get("parameters", {})
                    contents = p.get("contents", [])
                    if contents:
                        for c in contents:
                            st.markdown(f"- {c}")
                    if p.get("solvent"):
                        st.markdown(f"**Solvent:** {p['solvent']}")
                    if p.get("flow_rate_mL_min"):
                        st.markdown(f"**Flow rate:** {p['flow_rate_mL_min']} mL/min")
                    for k, v in p.items():
                        if k not in ("contents", "solvent", "flow_rate_mL_min", "stream", "light_required"):
                            if v is not None:
                                st.markdown(f"**{k.replace('_', ' ')}:** {v}")
                    if op.get("rationale"):
                        st.caption(op["rationale"])
            if topo.get("pid_description"):
                st.code(topo["pid_description"], language=None)

    # ── TAB 2: Chemistry Plan ──────────────────────────────────────────────────
    with tabs[2]:
        _render_chemistry_plan(result.get("chemistry_plan", {}))

    # ── TAB 3: Stream Assignments ──────────────────────────────────────────────
    with tabs[3]:
        _render_streams(proposal)

    # ── TAB 4: Flow Conditions ─────────────────────────────────────────────────
    with tabs[4]:
        _render_conditions(proposal)

    # ── TAB 5: Engineering Report ──────────────────────────────────────────────
    with tabs[5]:
        from components.council_report import render_council_report

        class _Candidate:
            def __init__(self, r):
                self.council_rounds = r.get("council_rounds", 0)
                self.safety_report = r.get("safety_report", {})
                self.council_messages = r.get("council_messages", [])

        render_council_report(_Candidate(result))

    # ── TAB 6: Raw JSON ───────────────────────────────────────────────────────
    with tabs[6]:
        st.json(result)

    # ── Feedback widget ────────────────────────────────────────────────────────
    from components.feedback import render_feedback_widget
    render_feedback_widget(result, context="translate")


# ── Helper renderers ──────────────────────────────────────────────────────────

def _render_chemistry_plan(plan: dict):
    if not plan:
        st.info("No chemistry plan available.")
        return

    st.markdown(f"### {plan.get('reaction_name', 'Unknown Reaction')}")
    st.markdown(
        f"**Class:** {plan.get('reaction_class', 'N/A')} | "
        f"**Mechanism:** {plan.get('mechanism_type', 'N/A')} | "
        f"**Bond formed:** {plan.get('bond_formed', 'N/A')}"
    )

    # Reagents
    reagents = plan.get("reagents", [])
    if reagents:
        st.markdown("#### Species Inventory")
        for r in reagents:
            role = r.get("role", "?")
            name = r.get("name", "?")
            amt = r.get("equiv_or_loading", "")
            notes = f" — *{r['notes']}*" if r.get("notes") else ""
            st.markdown(f"- **{name}** ({role}, {amt}){notes}")

    # Mechanism
    steps = plan.get("mechanism_steps", [])
    if steps:
        st.markdown("#### Reaction Mechanism")
        for step in steps:
            prefix = "**[hv]** " if step.get("is_photon_dependent") else ""
            rls = " *(rate-limiting)*" if step.get("is_rate_limiting") else ""
            st.markdown(f"{step.get('step_number', '?')}. {prefix}{step.get('description', '')}{rls}")
        if plan.get("key_intermediate"):
            st.info(f"Key intermediate: **{plan['key_intermediate']}**")

    # Sensitivities
    st.markdown("#### Sensitivities")
    c1, c2, c3 = st.columns(3)
    c1.metric("O2 sensitive", "Yes" if plan.get("oxygen_sensitive") else "No")
    c2.metric("Moisture sensitive", "Yes" if plan.get("moisture_sensitive") else "No")
    c3.metric("Temp sensitive", "Yes" if plan.get("temperature_sensitive") else "No")

    if plan.get("deoxygenation_required"):
        st.warning(f"Deoxygenation required: {plan.get('deoxygenation_reasoning', '')}")

    # Stream logic
    slogic = plan.get("stream_logic", [])
    if slogic:
        st.markdown("#### Stream Separation Logic")
        for sl in slogic:
            st.markdown(f"**Stream {sl.get('stream_label', '?')}:** {', '.join(sl.get('reagents', []))}")
            if sl.get("reasoning"):
                st.caption(sl["reasoning"])

    incompat = plan.get("incompatible_pairs", [])
    if incompat:
        st.error(f"Incompatible pairs: {incompat}")

    if plan.get("recommended_wavelength_nm"):
        st.markdown(
            f"**Wavelength:** {plan['recommended_wavelength_nm']} nm "
            f"— {plan.get('wavelength_reasoning', '')}"
        )

    keywords = plan.get("retrieval_keywords", [])
    if keywords:
        st.caption(f"Retrieval keywords: {', '.join(keywords)}")


def _render_streams(proposal: dict, design_calc: dict | None = None):
    streams = proposal.get("streams", [])
    if not streams:
        st.info("No stream assignments available.")
        return

    # ── Engineering context from design calculator ─────────────────────────
    if design_calc:
        n_lim = design_calc.get("n_molar_flow_mmol_min")
        q_proposal = proposal.get("flow_rate_mL_min")

        # Classify streams: reactor feeds go into the main reactor,
        # quench streams are injected downstream.
        _QUENCH_KW = ("quench", "neutraliz", "workup", "post-reactor")

        def _is_quench(s):
            role = (s.get("pump_role") or "").lower()
            return any(kw in role for kw in _QUENCH_KW)

        def _is_gas(s):
            role = (s.get("pump_role") or "").lower().strip()
            return role in {"n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂",
                            "h2", "h₂", "hydrogen", "ar", "argon", "helium", "air", "mfc"}

        feed_rates = [
            s.get("flow_rate_mL_min") for s in streams
            if s.get("flow_rate_mL_min") and not _is_quench(s) and not _is_gas(s)
        ]
        quench_rates = [
            s.get("flow_rate_mL_min") for s in streams
            if s.get("flow_rate_mL_min") and _is_quench(s) and not _is_gas(s)
        ]
        q_reactor = round(sum(feed_rates), 4) if feed_rates else q_proposal
        q_quench_total = round(sum(quench_rates), 4)
        q_outlet = round((q_reactor or 0.0) + q_quench_total, 4) if q_quench_total else q_reactor

        n_metric_cols = 3 + (1 if q_quench_total else 0)
        ctx_cols = st.columns(n_metric_cols)
        col_i = 0
        if n_lim:
            ctx_cols[col_i].metric(
                "ṅ_limiting", f"{n_lim:.4f} mmol/min",
                help="Molar flow of limiting reagent",
            )
            col_i += 1
        if q_reactor:
            ctx_cols[col_i].metric(
                "Q_reactor_inlet", f"{q_reactor} mL/min",
                help="Σ feed pumps entering the main reactor (excludes quench streams)",
            )
            col_i += 1
        if q_quench_total:
            ctx_cols[col_i].metric(
                "Q_outlet (after quench)", f"{q_outlet} mL/min",
                help="Reactor outlet + quench stream(s)",
            )
            col_i += 1
        C_rxr = design_calc.get("C_reactor_M")
        if C_rxr and col_i < n_metric_cols:
            ctx_cols[col_i].metric(
                "C_reactor", f"{C_rxr:.3f} M",
                help="[limiting reagent] inside reactor after stream mixing",
            )
        st.divider()

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown("#### Pump / Stream Assignments")
    chief_derived = any(s.get("reasoning") for s in streams)
    if chief_derived:
        st.caption("Flowrates derived by the Chief Engineer from ṅ_limiting and feed concentrations. "
                   "Expand each pump below for the derivation formula.")

    for s in streams:
        label = s.get("stream_label", "?")
        role = s.get("pump_role", "")
        contents = s.get("contents", [])
        solvent = s.get("solvent", "")
        conc = s.get("concentration_M")
        rate = s.get("flow_rate_mL_min")
        reasoning = s.get("reasoning", "")

        # Compact header with flowrate badge
        rate_badge = f" — **{rate} mL/min**" if rate else ""
        with st.expander(f"Pump {label}: {role}{rate_badge}", expanded=True):
            if contents:
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown("**Contents:**")
                    for item in contents:
                        st.markdown(f"- {item}")
                with cols[1]:
                    detail_rows = []
                    if solvent:
                        detail_rows.append(("Solvent", solvent))
                    if conc:
                        detail_rows.append(("Feed conc.", f"{conc} M"))
                    if rate:
                        detail_rows.append(("Flow rate", f"{rate} mL/min"))
                    for k, v in detail_rows:
                        st.markdown(f"**{k}:** {v}")
            else:
                details = []
                if solvent:
                    details.append(f"Solvent: {solvent}")
                if conc:
                    details.append(f"{conc} M")
                if rate:
                    details.append(f"**{rate} mL/min**")
                if details:
                    st.markdown(" | ".join(details))

            if reasoning:
                st.markdown("**Chief derivation:**")
                st.code(reasoning, language=None)

    mixing = proposal.get("mixing_order_reasoning", "")
    if mixing:
        st.divider()
        st.markdown(f"**Mixing ({proposal.get('mixer_type', 'T-mixer')}):** {mixing}")

    pre = proposal.get("pre_reactor_steps", [])
    post = proposal.get("post_reactor_steps", [])
    if pre:
        st.divider()
        st.markdown("**Pre-reactor steps:**")
        for s in pre:
            st.markdown(f"- {s}")
    if post:
        st.markdown("**Post-reactor steps:**")
        for s in post:
            st.markdown(f"- {s}")


def _render_conditions(proposal: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Residence Time", f"{proposal.get('residence_time_min', 0):.1f} min")
        st.metric("Flow Rate", f"{proposal.get('flow_rate_mL_min', 0):.2f} mL/min")
        st.metric("Temperature", f"{proposal.get('temperature_C', 25):.0f} °C")
        st.metric("Concentration", f"{proposal.get('concentration_M', 0):.3f} M")
    with c2:
        st.metric("Reactor Type", proposal.get("reactor_type", "N/A"))
        st.metric("Tubing Material", proposal.get("tubing_material", "N/A"))
        st.metric("Tubing ID", f"{proposal.get('tubing_ID_mm', 0):.1f} mm")
        st.metric("Reactor Volume", f"{proposal.get('reactor_volume_mL', 0):.1f} mL")
    with c3:
        st.metric("BPR", f"{proposal.get('BPR_bar', 0):.0f} bar")
        st.metric("Light Setup", proposal.get("light_setup", "N/A"))
        st.metric("Wavelength", f"{proposal.get('wavelength_nm', 'N/A')} nm")
        st.metric("Deoxygenation", proposal.get("deoxygenation_method", "N/A"))

    reasoning = proposal.get("reasoning_per_field", {})
    if reasoning:
        st.divider()
        st.markdown("#### Reasoning per Field")
        for field, reason in reasoning.items():
            st.markdown(f"- **{field}:** {reason}")


def _show_example():
    with st.expander("See an example", expanded=False):
        st.markdown("""
**Example input:**

> fac-Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative radical addition
> of N-Boc-proline (1.0 equiv) to methyl vinyl ketone (2.0 equiv),
> K2HPO4 (1.5 equiv), DMF, 0.1 M, RT, 450 nm blue LED, N2, 24h, 72% yield.

**Expected output:**
- FEP coil reactor (1.0 mm ID, ~5 mL), inline N2 deoxygenation
- Pump A: proline + Ir(ppy)3 in DMF | Pump B: MVK + K2HPO4 in DMF
- Kessil 450 nm LED, BPR 5 bar
- Residence time ~8-12 min, flow rate ~0.5 mL/min
""")
