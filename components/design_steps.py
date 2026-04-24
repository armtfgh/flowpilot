"""Streamlit component: renders the 9-step engineering design calculation.

Shows each step as an expandable card with:
  - Status indicator (✓ / ⚠ / ✗ / ↻ / ≈)
  - Summary line
  - LaTeX equations with actual numbers
  - Computed values as metrics
  - Warnings and adjustments
  - Assumptions made
"""

import streamlit as st


_STATUS_ICON = {
    "PASS": "✅",
    "WARNING": "⚠️",
    "FAIL": "❌",
    "ADJUSTED": "🔄",
    "ESTIMATED": "≈",
}

_STATUS_COLOR = {
    "PASS": "green",
    "WARNING": "orange",
    "FAIL": "red",
    "ADJUSTED": "blue",
    "ESTIMATED": "gray",
}


def render_design_steps(calc_dict: dict, key_prefix: str = "ds"):
    """Render the full 9-step engineering design from a dict (serialised DesignCalculations)."""
    steps = calc_dict.get("steps", [])
    if not steps:
        st.info("No design calculations available.")
        return

    # ── Header metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("τ (residence time)", f"{calc_dict.get('residence_time_min', 0):.1f} min")
    c2.metric("Q (flow rate)", f"{calc_dict.get('flow_rate_mL_min', 0):.3f} mL/min")
    c3.metric("V_R (reactor vol)", f"{calc_dict.get('reactor_volume_mL', 0):.2f} mL")
    c4.metric("Re", f"{calc_dict.get('reynolds_number', 0):.1f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("d (tubing ID)", f"{calc_dict.get('tubing_ID_mm', 0):.2f} mm")
    c6.metric("L (tube length)", f"{calc_dict.get('tubing_length_m', 0):.2f} m")
    c7.metric("ΔP", f"{calc_dict.get('pressure_drop_bar', 0):.4f} bar")
    bpr = calc_dict.get("bpr_pressure_bar", 0)
    c8.metric("BPR", f"{bpr:.1f} bar" if bpr else "Not required")

    # Row 3 — flow chemistry specific
    c9, c10, c11, c12 = st.columns(4)
    Pe = calc_dict.get("Pe")
    Pe_ok = calc_dict.get("Pe_adequate", True)
    c9.metric("Péclet (Pe)", f"{Pe:.0f}" if Pe else "N/A",
              delta="plug flow ✓" if Pe_ok else "⚠ axial dispersion",
              delta_color="normal" if Pe_ok else "inverse",
              help="Pe = 192·τ·D_mol/d² — must be ≥100 for plug-flow approximation")
    n_lim = calc_dict.get("n_molar_flow_mmol_min")
    c10.metric("ṅ_limiting", f"{n_lim:.4f} mmol/min" if n_lim else "N/A",
               help="Molar flow of limiting reagent = P_batch/(Y×60)")
    P_flow = calc_dict.get("P_flow_mmol_h")
    c11.metric("P_flow", f"{P_flow:.1f} mmol/h" if P_flow else "N/A",
               help="Forward productivity check: ṅ_lim × Y × 60")
    startup = calc_dict.get("startup_waste_mL")
    c12.metric("Startup Waste", f"{startup} mL" if startup else "N/A",
               help="Volume wasted during startup = 3×τ×Q")

    # Consistency badge
    if calc_dict.get("consistent", True):
        st.success("All consistency checks passed — τ = V_R/Q, L = 4V/(πd²), Re = ρvd/μ ✓")
    else:
        notes = calc_dict.get("consistency_notes", [])
        st.error("Consistency issues:\n" + "\n".join(f"- {n}" for n in notes))

    st.divider()

    # ── Step-by-step cards ──────────────────────────────────────────────
    for step in steps:
        _render_step(step, key_prefix)


def _render_step(step: dict, key_prefix: str):
    """Render a single step as an expandable card."""
    num = step.get("step", 0)
    name = step.get("name", "")
    status = step.get("status", "PASS")
    summary = step.get("summary", "")
    icon = _STATUS_ICON.get(status, "?")
    color = _STATUS_COLOR.get(status, "gray")

    with st.expander(
        f"{icon}  Step {num}: {name} — :{color}[{status}]",
        expanded=(status in ("FAIL", "ADJUSTED", "WARNING")),
    ):
        st.markdown(f"**{summary}**")

        # Equations (LaTeX)
        equations = step.get("equations", [])
        if equations:
            st.markdown("**Equations:**")
            for eq in equations:
                try:
                    st.latex(eq)
                except Exception:
                    st.code(eq)

        # Key values
        values = step.get("values", {})
        if values:
            # Pick the most useful values to display as metrics
            displayable = {
                k: v for k, v in values.items()
                if isinstance(v, (int, float)) and v is not None
            }
            if displayable:
                cols = st.columns(min(len(displayable), 4))
                for i, (k, v) in enumerate(displayable.items()):
                    label = k.replace("_", " ")
                    if isinstance(v, float):
                        if abs(v) < 0.01 and v != 0:
                            display = f"{v:.3e}"
                        elif abs(v) > 1e4:
                            display = f"{v:.0f}"
                        else:
                            display = f"{v:.4g}"
                    else:
                        display = str(v)
                    cols[i % len(cols)].metric(label, display)

            # Non-numeric values
            text_vals = {
                k: v for k, v in values.items()
                if isinstance(v, str) and v
            }
            if text_vals:
                for k, v in text_vals.items():
                    st.markdown(f"**{k.replace('_', ' ')}:** {v}")

        # Warnings
        for w in step.get("warnings", []):
            st.warning(w)

        # Adjustments
        for a in step.get("adjustments", []):
            st.info(f"↻ {a}")

        # Assumptions
        assumptions = step.get("assumptions", [])
        if assumptions:
            st.caption("Assumptions: " + " · ".join(assumptions))
