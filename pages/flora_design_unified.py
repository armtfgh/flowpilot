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
# BATCH-TO-FLOW mode  — conversational chat interface
# ─────────────────────────────────────────────────────────────────────────────

def _init_chat_state():
    """Initialise session-state keys for the translate chat."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_agent" not in st.session_state:
        from flora_translate.conversation_agent import ConversationAgent
        st.session_state.chat_agent = ConversationAgent()
    if "active_result" not in st.session_state:
        st.session_state.active_result = None


def _render_translate():
    _init_chat_state()

    st.markdown("---")

    # ── Control bar ────────────────────────────────────────────────────────
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.subheader("Batch → Flow Chat")
    with col_reset:
        if st.button("🔄 Reset", key="chat_reset", help="Start a new conversation"):
            st.session_state.chat_messages = []
            st.session_state.active_result = None
            st.session_state.chat_agent.reset()
            st.rerun()

    # ── Render chat history ────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show a compact result badge in the chat — no tabs inside chat bubble
            if msg.get("result"):
                _render_result_compact(msg["result"], i)
            if msg.get("questions"):
                st.info("**Questions for you:**\n" +
                        "\n".join(f"{j+1}. {q}" for j, q in enumerate(msg["questions"])))

    # ── Example prompts (only when chat is empty) ──────────────────────────
    if not st.session_state.chat_messages:
        st.markdown(
            """
            <div style="color:#888; font-size:0.88em; margin-bottom:8px;">
            <b>Try one of these to get started:</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        examples = [
            "fac-Ir(ppy)₃ (1 mol%) photoredox, N-Boc-proline (1 equiv) + MVK (2 equiv), K₂HPO₄, DMF, 0.1 M, RT, 450 nm, N₂, 24 h, 72% yield",
            "Pd-catalyzed Suzuki coupling, ArBr + PhB(OH)₂, K₂CO₃, EtOH/H₂O 4:1, 80°C, 2h, 89% yield",
            "NaBH₄ reduction of ketone to alcohol, MeOH, 0°C, 30 min, quant. yield",
        ]
        for ex in examples:
            if st.button(ex[:80] + "…" if len(ex) > 80 else ex,
                         key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state["_prefill_input"] = ex
                st.rerun()

    # ── Chat input ─────────────────────────────────────────────────────────
    prefill = st.session_state.pop("_prefill_input", "")
    prompt  = st.chat_input(
        "Describe your batch protocol, ask a question, or request a revision…",
        key="chat_input",
    ) or (prefill if prefill else None)

    if prompt:
        # Show user message immediately
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with agent
        with st.chat_message("assistant"):
            agent = st.session_state.chat_agent

            # Determine if this will trigger a translation run
            has_result = agent.current_result is not None
            is_likely_translation = not has_result or any(
                kw in prompt.lower()
                for kw in ["translate", "convert", "new reaction", "fresh", "start over"]
            )

            spinner_msg = (
                "Running FLORA pipeline — this takes ~30-60 seconds…"
                if is_likely_translation or any(
                    kw in prompt.lower()
                    for kw in ["add", "remove", "change", "revise", "modify",
                                "instead", "try", "use", "switch"]
                )
                else "Thinking…"
            )

            with st.spinner(spinner_msg):
                try:
                    response = agent.process(prompt)
                except Exception as e:
                    response_msg = f"Something went wrong: {e}"
                    st.error(response_msg)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response_msg}
                    )
                    st.stop()

            # Display assistant response
            st.markdown(response.message)

            if response.result:
                st.session_state["active_result"] = response.result
                _render_result_compact(response.result, "new")

            if response.questions:
                st.info("**I have a few questions to improve the design:**\n" +
                        "\n".join(f"{j+1}. {q}" for j, q in enumerate(response.questions)))

            if response.error:
                from components.error_card import render_error
                render_error(Exception(response.error), "FLORA-Translate")

        # Persist to history
        st.session_state.chat_messages.append({
            "role":      "assistant",
            "content":   response.message,
            "result":    response.result,
            "questions": response.questions,
        })

    # ── Full result view — OUTSIDE chat bubbles (no nested tabs issue) ─────
    active = st.session_state.get("active_result")
    if active:
        st.markdown("---")
        _render_result(active, key_prefix="active")


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
# Compact result badge (shown INSIDE chat bubble — no tabs, no nested widgets)
# ─────────────────────────────────────────────────────────────────────────────

def _render_result_compact(result: dict, key_suffix):
    """Show a small summary card inside a chat message — no tabs, no downloads."""
    proposal = result.get("proposal", {})
    conf     = result.get("confidence", "?")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    rt  = proposal.get("residence_time_min", "?")
    rxt = proposal.get("reactor_type", "?")
    fr  = proposal.get("flow_rate_mL_min", "?")
    st.markdown(
        f"**Confidence:** :{conf_color}[{conf}] &nbsp;|&nbsp; "
        f"**Reactor:** {rxt} &nbsp;|&nbsp; "
        f"**τ =** {rt} min &nbsp;|&nbsp; "
        f"**Q =** {fr} mL/min"
    )
    st.caption("↓ Full design with process diagram, chemistry plan, and conditions shown below")


# ─────────────────────────────────────────────────────────────────────────────
# Shared result renderer (translate output) — shown OUTSIDE chat bubbles
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(result: dict, key_prefix: str = ""):
    proposal = result.get("proposal", {})
    conf = result.get("confidence", "LOW")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    tabs = st.tabs([
        "Summary",
        "Design Space",
        "Engineering Design",
        "Process Diagram",
        "Chemistry Plan & Recipe",
        "Stream Assignments",
        "Council Deliberation",
        "Council Report",
        "Raw JSON",
    ])

    with tabs[0]:
        st.markdown(result.get("explanation", ""))
        notes = proposal.get("chemistry_notes", "")
        if notes:
            st.divider()
            st.info(notes)

    with tabs[1]:  # Design Space
        st.markdown("### Design Space — Pre-Council Candidates")
        st.caption(
            "Grid search run BEFORE the council. Each point is a (τ, d, Q) combination "
            "that satisfies geometric constraints. The highlighted candidate was used as "
            "the council starting point."
        )
        design_space = result.get("design_space", [])
        if design_space:
            from components.design_space_viz import render_design_space
            render_design_space(design_space, key_prefix=key_prefix)
        else:
            st.info("Design space search was not run for this result (older format).")

    with tabs[2]:
        # 9-step engineering design calculations
        design_calc = result.get("design_calculations")
        if design_calc:
            from components.design_steps import render_design_steps
            render_design_steps(design_calc, key_prefix=f"{key_prefix}_ds")
        else:
            st.info("No design calculations available for this result.")
        # Per-reactor breakdown from topology
        topo_for_eng = result.get("process_topology", {})
        reactor_ops = [
            op for op in topo_for_eng.get("unit_operations", [])
            if op.get("op_type") in ("coil_reactor", "reactor", "heated_coil",
                                      "photoreactor", "chip_reactor",
                                      "packed_bed", "packed_bed_reactor")
        ]
        if len(reactor_ops) > 1:
            st.divider()
            st.markdown("### Per-Reactor Breakdown")
            for idx, rop in enumerate(reactor_ops, 1):
                p = rop.get("parameters", {})
                with st.expander(f"Reactor {idx}: {rop.get('label', '?')}", expanded=True):
                    cols = st.columns(4)
                    cols[0].metric("τ", f"{p.get('residence_time_min', '?')} min")
                    cols[1].metric("Volume", f"{p.get('volume_mL', '?')} mL")
                    cols[2].metric("ID", f"{p.get('ID_mm', '?')} mm")
                    cols[3].metric("Temperature", f"{p.get('temperature_C', '?')} °C")
                    c2 = st.columns(3)
                    c2[0].metric("Material", p.get("material", "?"))
                    length = p.get("length_m")
                    c2[1].metric("Length", f"{length} m" if length else "?")
                    wl = p.get("wavelength_nm")
                    c2[2].metric("Wavelength", f"{wl} nm" if wl else "N/A")

    with tabs[3]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(result.get("svg_path", ""), result.get("png_path", ""), key_prefix=key_prefix)
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

    with tabs[4]:
        from pages.translate import _render_chemistry_plan
        _render_chemistry_plan(result.get("chemistry_plan", {}))
        st.divider()
        _render_recipe(result)

    with tabs[5]:
        from pages.translate import _render_streams
        _render_streams(proposal)

    with tabs[6]:
        _render_council_deliberation(result)

    with tabs[7]:
        from components.council_report import render_council_report
        class _C:
            def __init__(self, r):
                self.council_rounds = r.get("council_rounds", 0)
                self.safety_report  = r.get("safety_report", {})
                self.council_messages = r.get("council_messages", [])
        render_council_report(_C(result))

    with tabs[8]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Council Deliberation — narrative summary of multi-agent conversation
# ─────────────────────────────────────────────────────────────────────────────

def _render_council_deliberation(result: dict):
    """Render the ENGINE council deliberation with full chain-of-thought."""
    delib_log = result.get("deliberation_log")
    proposal = result.get("proposal", {})
    rounds_count = result.get("council_rounds", 0)

    if not delib_log:
        # Fallback to legacy format
        _render_legacy_deliberation(result)
        return

    rounds = delib_log.get("rounds", [])
    sanity_checks = delib_log.get("sanity_checks", [])
    total_rounds = delib_log.get("total_rounds", len(rounds))
    consensus = delib_log.get("consensus_reached", False)

    st.markdown(f"### Multi-Agent Deliberation — {total_rounds} Round{'s' if total_rounds != 1 else ''}")

    # ── Header stats ──────────────────────────────────────────────────
    all_deliberations = [d for rnd in rounds for d in rnd]
    n_accept = sum(1 for d in all_deliberations if d.get("status") == "ACCEPT")
    n_warn   = sum(1 for d in all_deliberations if d.get("status") == "WARNING")
    n_revise = sum(1 for d in all_deliberations if d.get("status") == "REVISE")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds", total_rounds)
    c2.metric("Accepted", n_accept)
    c3.metric("Warnings", n_warn)
    c4.metric("Revisions", n_revise)

    _STATUS_ICONS = {"ACCEPT": "✅", "WARNING": "⚠️", "REVISE": "🔄"}
    _AGENT_AVATARS = {
        "Dr. Kinetics": "⏱️",
        "Dr. Fluidics": "🌊",
        "Dr. Safety": "🛡️",
        "Dr. Chemistry": "🧪",
        "Dr. Process": "🏗️",
        "Chief Engineer": "👷",
    }

    # ── Round-by-round rendering ──────────────────────────────────────
    for r_idx, round_delibs in enumerate(rounds):
        round_num = r_idx + 1
        st.divider()
        st.markdown(f"#### Round {round_num}" +
                    (" — Independent Analysis" if round_num == 1 else " — Cross-Agent Debate"))

        for d in round_delibs:
            agent_name = d.get("agent_display_name", d.get("agent", "?"))
            status = d.get("status", "?")
            had_error = d.get("had_error", False)
            icon = "💥" if had_error else _STATUS_ICONS.get(status, "•")
            avatar = _AGENT_AVATARS.get(agent_name, "🤖")
            error_suffix = " [ERROR — blocks convergence]" if had_error else ""

            cot = d.get("chain_of_thought", "")
            is_green_skip = cot.startswith("Domain ") and "green zone per triage" in cot
            with st.expander(
                f"{avatar} {agent_name} — {icon} {status}{error_suffix}",
                expanded=(had_error or (bool(cot) and not is_green_skip)),
            ):
                # Chain of thought
                if cot:
                    st.markdown(cot)

                # Values Referenced (from DesignCalculator)
                calcs = d.get("values_referenced", d.get("calculations", []))
                if calcs:
                    st.markdown("**Values Referenced:**")
                    for calc in calcs:
                        st.code(calc, language=None)

                # Findings
                findings = d.get("findings", [])
                if findings:
                    st.markdown("**Findings:**")
                    for f in findings:
                        st.markdown(f"- {f}")

                # Proposals (structured FieldProposal objects)
                proposals = d.get("proposals", [])
                if proposals:
                    st.markdown("**Proposals:**")
                    for p in proposals:
                        if isinstance(p, dict) and p.get("field"):
                            st.info(f"💡 `{p['field']}` → **{p['value']}** — {p.get('reason', '')}")
                        elif isinstance(p, dict) and p.get("reason"):
                            st.info(f"💡 {p['reason']}")
                        elif isinstance(p, str):
                            st.info(f"💡 {p}")

                # Concerns
                concerns = d.get("concerns", [])
                if concerns:
                    st.markdown("**Concerns:**")
                    for c in concerns:
                        st.warning(c)

                # References to other agents
                refs = d.get("references_to_agents", [])
                if refs:
                    st.caption("References: " + ", ".join(refs))

                # Rules cited
                rules = d.get("rules_cited", [])
                if rules:
                    st.caption("Handbook rules cited: " + ", ".join(rules[:5]))

        # ── Sanity check for this round ───────────────────────────────
        if r_idx < len(sanity_checks):
            sc = sanity_checks[r_idx]
            st.divider()
            with st.expander(f"👷 Chief Engineer — Sanity Check (Round {round_num})",
                             expanded=bool(sc.get("conflicts_found"))):
                sc_cot = sc.get("chain_of_thought", "")
                if sc_cot:
                    st.markdown(sc_cot)

                conflicts = sc.get("conflicts_found", [])
                if conflicts:
                    st.markdown("**Conflicts found:**")
                    for c in conflicts:
                        st.error(c)

                resolutions = sc.get("resolutions", [])
                if resolutions:
                    st.markdown("**Resolutions:**")
                    for r in resolutions:
                        st.success(r)

                changes = sc.get("final_changes", {})
                if changes:
                    st.markdown("**Applied changes:**")
                    for field, val in changes.items():
                        st.markdown(f"- `{field}` → **{val}**")

    # ── Final outcome ─────────────────────────────────────────────────
    st.divider()
    # Build final design parameters string — always show τ, Q, V_R plus any
    # field that was actually changed during deliberation
    _FIELD_LABELS = {
        "residence_time_min": ("τ", "min"),
        "flow_rate_mL_min": ("Q", "mL/min"),
        "reactor_volume_mL": ("V_R", "mL"),
        "tubing_ID_mm": ("d", "mm"),
        "tubing_material": ("material", ""),
        "temperature_C": ("T", "°C"),
        "concentration_M": ("C", "M"),
        "BPR_bar": ("BPR", "bar"),
        "wavelength_nm": ("λ", "nm"),
        "deoxygenation_method": ("degas", ""),
        "mixer_type": ("mixer", ""),
    }
    always_show = ["residence_time_min", "flow_rate_mL_min", "reactor_volume_mL"]
    changed_fields = list(delib_log.get("all_changes_applied", {}).keys())
    show_fields = list(dict.fromkeys(always_show + changed_fields))  # preserve order, dedupe

    param_parts = []
    for field in show_fields:
        if field not in _FIELD_LABELS:
            continue
        label, unit = _FIELD_LABELS[field]
        val = proposal.get(field, "?")
        suffix = f" {unit}" if unit else ""
        marker = " ★" if field in changed_fields else ""  # star = modified by council
        param_parts.append(f"{label} = {val}{suffix}{marker}")
    param_str = "  ·  ".join(param_parts)

    if consensus:
        msg = (
            f"Consensus reached after {total_rounds} round{'s' if total_rounds != 1 else ''}  "
            f"·  {param_str}"
        )
        if changed_fields:
            msg += f"  ·  (★ = modified by council: {', '.join(changed_fields)})"
        st.success(msg)
    else:
        n_errors = sum(
            1 for rnd in rounds for d in rnd if d.get("had_error")
        )
        warning_msg = f"Max rounds ({total_rounds}) reached"
        if n_errors:
            warning_msg += f" — {n_errors} agent error(s) prevented convergence"
        warning_msg += f"  ·  {param_str}"
        st.warning(warning_msg)


def _render_legacy_deliberation(result: dict):
    """Fallback renderer for results without the new deliberation log."""
    msgs_raw = result.get("council_messages", [])
    rounds = result.get("council_rounds", 0)
    proposal = result.get("proposal", {})

    if not msgs_raw:
        st.info("No council deliberation data available.")
        return

    msgs = []
    for m in msgs_raw:
        if isinstance(m, dict):
            msgs.append(m)
        elif hasattr(m, "model_dump"):
            msgs.append(m.model_dump())

    st.markdown(f"### Council Review — {rounds} Round{'s' if rounds != 1 else ''}")

    _ICONS = {"REJECT": "❌", "WARNING": "⚠️", "ACCEPT": "✅"}
    _NAMES = {
        "DesignCalculator": "Physics Engine",
        "KineticsAgent": "Dr. Kinetics", "KineticsSpecialist": "Dr. Kinetics",
        "FluidicsAgent": "Dr. Fluidics", "FluidicsSpecialist": "Dr. Fluidics",
        "SafetyCriticAgent": "Dr. Safety", "SafetySpecialist": "Dr. Safety",
        "ChemistryValidator": "Dr. Chemistry", "ChemistrySpecialist": "Dr. Chemistry",
        "ProcessArchitectAgent": "Dr. Process", "IntegrationSpecialist": "Dr. Process",
    }
    for m in msgs:
        agent = m.get("agent", "?")
        status = m.get("status", "?")
        concern = m.get("concern", "")
        value = m.get("value", "")
        icon = _ICONS.get(status, "•")
        name = _NAMES.get(agent, agent)
        text = concern or value or "OK"
        st.markdown(f"{icon} **{name}**: {text[:200]}")

    st.divider()
    validated = proposal.get("engine_validated", False)
    if validated:
        st.success(f"Design validated after {rounds} rounds")
    else:
        st.warning("Design did not fully converge.")


# ─────────────────────────────────────────────────────────────────────────────
# Recipe — step-by-step experimental instructions for the bench chemist
# ─────────────────────────────────────────────────────────────────────────────

_GAS_NAMES = {
    "o2": "oxygen (O₂)", "o₂": "oxygen (O₂)", "oxygen": "oxygen (O₂)",
    "h2": "hydrogen (H₂)", "h₂": "hydrogen (H₂)", "hydrogen": "hydrogen (H₂)",
    "co2": "carbon dioxide (CO₂)", "co₂": "carbon dioxide (CO₂)",
    "co": "carbon monoxide (CO)", "n2": "nitrogen (N₂)", "n₂": "nitrogen (N₂)",
    "nitrogen": "nitrogen (N₂)", "ar": "argon (Ar)", "argon": "argon (Ar)",
    "air": "compressed air",
}

def _is_gas_stream(stream: dict) -> bool:
    """Return True if this stream carries a gas (not a liquid solution)."""
    _GAS_KW = {"o2", "o₂", "oxygen", "h2", "h₂", "hydrogen", "co2", "co₂",
               "syngas", "ethylene", "acetylene", "carbon monoxide", "carbonylation",
               "mfc", "gas", "n2 gas", "argon gas"}
    contents = stream.get("contents", [])
    pump_role = (stream.get("pump_role") or "").lower()
    label = (stream.get("stream_label") or "").lower()
    all_text = " ".join(str(c) for c in contents).lower() + " " + pump_role + " " + label
    return any(kw in all_text for kw in _GAS_KW)

def _identify_gas(stream: dict) -> str:
    """Return the human-readable gas name from a stream."""
    contents = stream.get("contents", [])
    pump_role = (stream.get("pump_role") or "").lower()
    all_text = " ".join(str(c) for c in contents).lower() + " " + pump_role
    for kw, name in _GAS_NAMES.items():
        if kw in all_text:
            return name
    return "gas"

def _render_recipe(result: dict):
    """Generate and display a step-by-step experimental recipe."""
    proposal = result.get("proposal", {})
    chem_plan = result.get("chemistry_plan", {})
    streams_raw = proposal.get("streams", [])

    if not proposal:
        return

    # Normalise streams to dicts
    streams = []
    for s in streams_raw:
        if isinstance(s, dict):
            streams.append(s)
        elif hasattr(s, "model_dump"):
            streams.append(s.model_dump())

    # Separate liquid and gas streams
    liquid_streams = [s for s in streams if not _is_gas_stream(s)]
    gas_streams    = [s for s in streams if _is_gas_stream(s)]

    st.markdown("### Experimental Recipe")
    st.markdown("*Step-by-step instructions for the bench chemist.*")

    step_num = [0]

    def step(text):
        step_num[0] += 1
        st.markdown(f"**{step_num[0]}.** {text}")

    # ── Safety note ─────────────────────────────────────────────────────
    safety_flags = proposal.get("safety_flags", [])
    if safety_flags or gas_streams:
        warnings_text = []
        if gas_streams:
            gas_names = [_identify_gas(s) for s in gas_streams]
            warnings_text.append(f"This process uses pressurised gas: **{', '.join(gas_names)}**. "
                                  "Follow your institution's gas handling safety procedures.")
        for flag in safety_flags[:3]:
            warnings_text.append(flag)
        if warnings_text:
            st.warning("⚠️ **Safety notes:** " + "  \n".join(warnings_text))

    # ── Section A: Solution Preparation ─────────────────────────────────
    st.markdown("#### A. Preparation")

    if liquid_streams:
        st.markdown("**Liquid solutions:**")
        for s in liquid_streams:
            label   = s.get("stream_label", "?")
            contents = s.get("contents", [])
            solvent  = s.get("solvent", "")
            conc     = s.get("concentration_M")

            # Clean content names (strip loading in brackets)
            names = [str(c).split("(")[0].strip() for c in contents] if contents else ["reagents"]
            content_str = ", ".join(names)
            conc_str = f" to give a **{conc} M** solution" if conc else ""

            step(
                f"**Stream {label} — Liquid solution:** Weigh out {content_str} "
                f"and dissolve in **{solvent or 'the appropriate solvent'}**{conc_str}. "
                "Transfer to a clean, dry, inert-atmosphere-compatible flask or syringe. "
                "Cap and label."
            )
    elif not gas_streams:
        step("Prepare reagent solutions at the specified concentrations in the appropriate solvents.")

    if gas_streams:
        st.markdown("**Gas feeds:**")
        for s in gas_streams:
            label    = s.get("stream_label", "?")
            gas_name = _identify_gas(s)
            fr       = s.get("flow_rate_mL_min")
            fr_str   = f" at **{fr:.1f} mL/min (≈ {fr * 16.67:.0f} sccm)**" if fr else ""

            step(
                f"**Stream {label} — Gas feed ({gas_name}):** Connect the "
                f"**{gas_name}** cylinder to a **mass flow controller (MFC)**"
                f"{fr_str}. Use gas-rated stainless steel or PTFE fittings "
                "and a check valve to prevent back-flow. "
                "Do **not** use a syringe pump for gas delivery."
            )

    # Deoxygenation
    deoxy = proposal.get("deoxygenation_method")
    if not deoxy and chem_plan.get("deoxygenation_required"):
        deoxy = "N₂ sparging"
    if deoxy and liquid_streams:
        step(
            f"**Deoxygenate all liquid solutions:** Sparge each with **{deoxy}** "
            "for 15 minutes using a stainless steel needle. "
            "Keep capped under inert atmosphere until use."
        )

    # ── Section B: System Setup ─────────────────────────────────────────
    st.markdown("#### B. Flow System Assembly")

    tubing_mat  = proposal.get("tubing_material", "FEP")
    tubing_id   = proposal.get("tubing_ID_mm", 1.0)
    reactor_vol = proposal.get("reactor_volume_mL", 0)
    reactor_type = proposal.get("reactor_type", "coil")

    step(
        f"Assemble the **{reactor_type} reactor**: cut **{tubing_mat}** tubing "
        f"(ID = **{tubing_id} mm**) to give a reactor volume of **{reactor_vol:.1f} mL**. "
        "Coil neatly and secure."
    )

    wl = proposal.get("wavelength_nm")
    if wl:
        step(
            f"Mount the **{wl:.0f} nm LED** light source around the reactor coil. "
            "Ensure uniform, full-length irradiation. Shield from ambient light."
        )

    temp = proposal.get("temperature_C", 25)
    if temp and temp != 25:
        step(
            f"Pre-heat or pre-cool the **temperature-controlled bath** to "
            f"**{temp:.0f} °C** and allow 10 minutes to equilibrate. "
            "Submerge the reactor coil completely."
        )

    bpr = proposal.get("BPR_bar", 0)
    if bpr and bpr > 0:
        step(
            f"Install the **back-pressure regulator (BPR)** set to **{bpr:.0f} bar** "
            "at the reactor outlet. "
            + ("This is required to maintain gas solubility throughout the reactor. " if gas_streams else
               "This prevents solvent boiling at the operating temperature. ")
        )

    mixer = proposal.get("mixer_type", "T-mixer")
    n_liquid = len(liquid_streams)
    n_gas    = len(gas_streams)
    total_inlets = n_liquid + n_gas
    step(
        f"Connect all **{total_inlets} inlet line{'s' if total_inlets != 1 else ''}** "
        f"({n_liquid} liquid pump{'s' if n_liquid != 1 else ''}"
        + (f", {n_gas} gas MFC{'s' if n_gas != 1 else ''}" if n_gas else "")
        + f") to the **{mixer}** at the reactor inlet. "
        "Finger-tighten all fittings, then confirm with wrench. Check for leaks."
    )

    if liquid_streams:
        n_syringes = len(liquid_streams)
        step(
            f"Load **{n_syringes} syringe{'s' if n_syringes != 1 else ''}** with "
            "the prepared liquid solutions. "
            "Remove all air bubbles — invert and tap. Mount on pump(s)."
        )

    # ── Section C: Running the Process ──────────────────────────────────
    st.markdown("#### C. Running the Process")

    flow_rate = proposal.get("flow_rate_mL_min", 0)
    rt        = proposal.get("residence_time_min", 0)

    if liquid_streams:
        for s in liquid_streams:
            fr = s.get("flow_rate_mL_min")
            label = s.get("stream_label", "?")
            if fr:
                step(f"Set **Pump {label}** (liquid) to **{fr:.3f} mL/min**.")
            elif flow_rate and len(liquid_streams) > 0:
                fr_each = flow_rate / len(liquid_streams)
                step(f"Set **Pump {label}** (liquid) to **{fr_each:.3f} mL/min**.")

    if gas_streams:
        for s in gas_streams:
            fr = s.get("flow_rate_mL_min")
            label = s.get("stream_label", "?")
            gas_name = _identify_gas(s)
            if fr:
                sccm = fr * 16.67
                step(
                    f"Set **MFC {label}** ({gas_name}) to "
                    f"**{fr:.2f} mL/min ({sccm:.0f} sccm)**."
                )
            else:
                step(
                    f"Set **MFC {label}** ({gas_name}) to the target flow rate "
                    "per your stoichiometry calculation."
                )

    step(
        "**Prime the system:** Start all liquid pumps at low flow (0.1 mL/min) "
        "to fill the tubing. Once solution exits at the outlet, increase to target flow."
    )

    if gas_streams:
        step(
            "Open the gas supply: slowly increase MFC flow to the target rate. "
            "Observe the outlet for stable slug/segmented flow formation."
        )

    step(
        f"Allow **{3 * rt:.0f} min** ({3}× residence time of {rt:.0f} min) "
        "for steady state. Discard the initial effluent."
        if rt > 0 else
        "Allow the system to reach steady state before collecting product."
    )

    step(
        "Collect product effluent into a pre-weighed vial. "
        "Record exact collection time, volume, and weight for yield calculation."
    )

    # ── Section D: Shutdown & Workup ────────────────────────────────────
    st.markdown("#### D. Shutdown & Workup")

    if gas_streams:
        step("Close the gas supply valves. Wait 30 seconds for pressure to equalise.")

    step("Stop all liquid pumps.")

    step(
        f"Flush the system with **{reactor_vol * 3:.1f} mL** of pure solvent "
        f"({3}× reactor volume) to recover residual product and clean the lines."
    )

    post_steps = proposal.get("post_reactor_steps", [])
    for ps in post_steps:
        step(f"{ps}")

    quench_reagent = chem_plan.get("quench_reagent", "")
    if chem_plan.get("quench_required") and quench_reagent:
        step(f"Quench the combined product fractions with **{quench_reagent}**.")

    step(
        "Work up the product: standard aqueous extraction, dry over anhydrous MgSO₄ "
        "or Na₂SO₄, filter, and concentrate under reduced pressure. "
        "Purify by column chromatography or recrystallisation as appropriate."
    )
