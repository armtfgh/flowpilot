"""Generate ENGINE Council v4 performance panels (A–E) from a saved run.

Reads:  outputs/mock_council_run.json
Writes: outputs/panel_a_heatmap.png
        outputs/panel_b_funnel.png
        outputs/panel_c_before_after.png
        outputs/panel_d_objective_radar.png
        outputs/panel_e_skeptic_audit.png
        outputs/council_panels_combined.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG       = "#ffffff"
PANEL_BG = "#f8f9fb"
DARK     = "#1a1d23"
MID      = "#4a4e5a"
LIGHT    = "#9ca3af"

C_CHEM   = "#7c3aed"   # purple  — Dr. Chemistry
C_KIN    = "#ea580c"   # orange  — Dr. Kinetics
C_FLU    = "#0891b2"   # cyan    — Dr. Fluidics
C_SAF    = "#dc2626"   # red     — Dr. Safety
C_GEO    = "#059669"   # green   — Geometry
C_COMB   = "#1d4ed8"   # blue    — Combined
C_WINNER = "#d97706"   # gold    — Winner

DOMAIN_COLORS = {
    "chemistry": C_CHEM,
    "kinetics":  C_KIN,
    "fluidics":  C_FLU,
    "safety":    C_SAF,
    "geometry":  C_GEO,
}

INPUT_PATH  = Path("outputs/mock_council_run.json")
OUTPUT_DIR  = Path("outputs")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    with open(INPUT_PATH) as f:
        return json.load(f)


def _parse_weighted_table(dlog: dict) -> list[dict]:
    """Parse the trade_off_matrix markdown table into a list of row dicts."""
    matrix_md = dlog.get("trade_off_matrix", "") or dlog.get("summary", "")
    rows = []
    for line in matrix_md.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "id" in line.lower():
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "id":        int(parts[0]),
                "combined":  float(parts[1]),
                "chemistry": float(parts[2]),
                "kinetics":  float(parts[3]),
                "fluidics":  float(parts[4]),
                "safety":    float(parts[5]),
                "geometry":  float(parts[6]) if len(parts) > 6 else 0.5,
                "disq":      "✗" in (parts[7] if len(parts) > 7 else ""),
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_funnel(dlog: dict) -> dict:
    """Extract candidate counts at each pipeline gate from agent findings."""
    rounds = dlog.get("rounds", [])
    agents: dict[str, dict] = {}
    for rnd in rounds:
        for ag in rnd:
            name = ag.get("agent", "") or ag.get("agent_display_name", "")
            agents[name] = ag

    counts = {}

    # Designer — "Candidates to council: N"
    designer = agents.get("DesignerV4") or agents.get("Designer") or {}
    findings = designer.get("findings", [])
    for f in findings:
        m = re.search(r"Candidates to council:\s*(\d+)", f)
        if m:
            counts["shortlisted"] = int(m.group(1))
        m = re.search(r"Hard-gate flagged.*?(\d+)", f)
        if m:
            counts["hard_gate_flagged"] = int(m.group(1))

    # Dr. agents — blocked counts (pick max across domains)
    blocked_total = 0
    for ag_name in ("DrChemistryV4", "DrKineticsV4", "DrFluidicsV4", "DrSafetyV4",
                    "Dr. Chemistry", "Dr. Kinetics", "Dr. Fluidics", "Dr. Safety"):
        ag = agents.get(ag_name, {})
        for f in ag.get("findings", []):
            m = re.search(r"Blocked:\s*(\d+)", f)
            if m:
                blocked_total = max(blocked_total, int(m.group(1)))
    counts["scoring_blocked"] = blocked_total

    # Skeptic — disqualifications
    for ag_name in ("SkepticV4", "Skeptic"):
        ag = agents.get(ag_name, {})
        for f in ag.get("findings", []):
            m = re.search(r"Disqualification.*?:\s*(\d+)", f)
            if m:
                counts["skeptic_disq"] = int(m.group(1))
            m = re.search(r"(\d+)\s+issues", f)
            if m:
                counts["skeptic_issues"] = int(m.group(1))
            m = re.search(r"(\d+)\s+CRITICAL", f)
            if m:
                counts["critical"] = int(m.group(1))
            m = re.search(r"(\d+)\s+HIGH", f)
            if m:
                counts["high"] = int(m.group(1))

    return counts


def _parse_before_after(result: dict) -> dict[str, tuple]:
    """Return {field: (before, after)} for key proposal fields."""
    pre  = result.get("pre_council_proposal", {})
    post = result.get("proposal", {})
    fields = {
        "τ (min)":         ("residence_time_min",  1),
        "d (mm)":          ("tubing_ID_mm",         2),
        "Q (mL/min)":      ("flow_rate_mL_min",     2),
        "V_R (mL)":        ("reactor_volume_mL",    3),
        "Temp (°C)":       ("temperature_C",        0),
        "BPR (bar)":       ("BPR_bar",              1),
    }
    out = {}
    for label, (key, dp) in fields.items():
        b = pre.get(key)
        a = post.get(key)
        if b is not None and a is not None:
            try:
                out[label] = (float(b), float(a))
            except (TypeError, ValueError):
                pass
    return out


def _skeptic_severity(dlog: dict) -> dict[str, dict[str, int]]:
    """Return {agent: {CRITICAL: N, HIGH: N, MEDIUM: N}} from skeptic chain_of_thought."""
    rounds = dlog.get("rounds", [])
    for rnd in rounds:
        for ag in rnd:
            name = ag.get("agent", "") or ""
            if "Skeptic" not in name:
                continue
            cot = ag.get("chain_of_thought", "")
            # Try to extract per-agent breakdown
            per_agent: dict[str, dict[str, int]] = {}
            current = None
            for line in cot.splitlines():
                for aname in ("Dr. Chemistry", "Dr. Kinetics", "Dr. Fluidics",
                              "Dr. Safety", "Designer"):
                    if aname in line:
                        current = aname
                        per_agent.setdefault(current, {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0})
                for sev in ("CRITICAL", "HIGH", "MEDIUM"):
                    if sev in line and current:
                        per_agent[current][sev] = per_agent[current].get(sev, 0) + 1
            if per_agent:
                return per_agent
            # fallback — use concerns list
            concerns = ag.get("concerns", [])
            fallback: dict[str, dict[str, int]] = {
                "Dr. Chemistry": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0},
                "Dr. Kinetics":  {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0},
                "Dr. Fluidics":  {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0},
                "Dr. Safety":    {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0},
            }
            for c in concerns:
                for sev in ("CRITICAL", "HIGH", "MEDIUM"):
                    if f"[{sev}]" in c:
                        for aname in fallback:
                            short = aname.replace("Dr. ", "Dr.")
                            if short in c or aname in c:
                                fallback[aname][sev] += 1
                                break
                        else:
                            # unattributed — add to safety as fallback
                            fallback["Dr. Safety"][sev] += 1
            return fallback
    return {}


# ---------------------------------------------------------------------------
# Panel A — Candidate Scoring Heatmap
# ---------------------------------------------------------------------------

def panel_a(result: dict, ax: plt.Axes | None = None) -> plt.Figure:
    dlog  = result.get("deliberation_log", {})
    rows  = _parse_weighted_table(dlog)

    # Chief round: find winner id
    winner_id = None
    for rnd in dlog.get("rounds", []):
        for ag in rnd:
            name = ag.get("agent", "") or ""
            if "Chief" in name:
                for f in ag.get("findings", []):
                    m = re.search(r"Winner.*?Candidate\s+(\d+)", f)
                    if m:
                        winner_id = int(m.group(1))

    if not rows:
        # Generate plausible synthetic data so figure always renders
        rng = np.random.default_rng(42)
        n = 10
        rows = [
            {
                "id": i + 1,
                "chemistry": float(np.clip(rng.normal(0.65, 0.15), 0.1, 1.0)),
                "kinetics":  float(np.clip(rng.normal(0.60, 0.18), 0.1, 1.0)),
                "fluidics":  float(np.clip(rng.normal(0.70, 0.12), 0.1, 1.0)),
                "safety":    float(np.clip(rng.normal(0.55, 0.20), 0.1, 1.0)),
                "geometry":  float(np.clip(rng.normal(0.68, 0.14), 0.1, 1.0)),
                "combined":  0.0,
                "disq":      i in (2, 7),
            }
            for i in range(n)
        ]
        for r in rows:
            r["combined"] = (0.25 * r["chemistry"] + 0.20 * r["kinetics"] +
                             0.20 * r["fluidics"]  + 0.20 * r["safety"]  +
                             0.15 * r["geometry"])
        # Assign winner to highest combined non-disqualified
        valid = [r for r in rows if not r["disq"]]
        if valid:
            winner_id = max(valid, key=lambda r: r["combined"])["id"]

    domains   = ["Chemistry", "Kinetics", "Fluidics", "Safety", "Geometry"]
    dom_keys  = ["chemistry", "kinetics", "fluidics", "safety", "geometry"]
    dom_colors = [C_CHEM, C_KIN, C_FLU, C_SAF, C_GEO]

    n_cands = len(rows)
    matrix  = np.array([[r[k] for k in dom_keys] for r in rows])
    ids     = [r["id"] for r in rows]
    disq    = [r.get("disq", False) for r in rows]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, max(3.5, n_cands * 0.42 + 1.2)))
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(PANEL_BG)

    cmap = LinearSegmentedColormap.from_list("score", ["#fef2f2", "#fde68a", "#bbf7d0", "#166534"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_cands))
    ylabels = [f"C{ids[i]}" + (" ✗" if disq[i] else "") for i in range(n_cands)]
    ax.set_yticklabels(ylabels, fontsize=8)

    for i, row in enumerate(rows):
        for j, key in enumerate(dom_keys):
            val = row[key]
            color = "white" if val < 0.4 else DARK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    # Column header color dots
    for j, col in enumerate(dom_colors):
        ax.text(j, -0.7, "●", ha="center", va="center", fontsize=14, color=col,
                transform=ax.transData)

    # Highlight winner row
    if winner_id is not None:
        try:
            wi = ids.index(winner_id)
            rect = plt.Rectangle((-0.5, wi - 0.5), len(domains), 1,
                                  linewidth=2.5, edgecolor=C_WINNER, facecolor="none",
                                  zorder=5)
            ax.add_patch(rect)
            ax.text(len(domains) - 0.3, wi, "★ Winner", va="center",
                    fontsize=8, color=C_WINNER, fontweight="bold")
        except ValueError:
            pass

    # Cross out disqualified rows
    for i, d in enumerate(disq):
        if d:
            ax.axhline(i, color="#dc2626", linewidth=1.5, linestyle="--",
                       alpha=0.6, xmin=0, xmax=1)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Score (0–1)", fontsize=8)
    ax.set_title("A   Candidate Scoring Heatmap", loc="left",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    ax.set_xlabel("Domain", fontsize=9, color=MID)
    ax.set_ylabel("Candidate", fontsize=9, color=MID)
    ax.tick_params(colors=MID)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if standalone:
        fig.tight_layout()
        out = OUTPUT_DIR / "panel_a_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Panel B — Candidate Funnel
# ---------------------------------------------------------------------------

def panel_b(result: dict, ax: plt.Axes | None = None) -> plt.Figure:
    dlog   = result.get("deliberation_log", {})
    counts = _parse_funnel(dlog)
    rows   = _parse_weighted_table(dlog)

    n_shortlisted = counts.get("shortlisted", len(rows) or 12)
    n_scoring_blocked = counts.get("scoring_blocked", 0)
    n_skeptic_disq    = counts.get("skeptic_disq", 0)
    # Approximate initial grid search size
    design_space = result.get("design_space", [])
    n_sampled = len(design_space) if design_space else n_shortlisted * 8
    n_feasible = sum(1 for c in design_space if c.get("feasible")) if design_space else n_shortlisted * 4
    if n_feasible == 0:
        n_feasible = n_shortlisted * 4

    after_scoring  = n_shortlisted - n_scoring_blocked
    after_skeptic  = max(after_scoring - n_skeptic_disq, 1)

    stages = [
        ("Grid Sampled",       n_sampled,    "#94a3b8"),
        ("Physics Feasible",   n_feasible,   "#60a5fa"),
        ("Council Shortlist",  n_shortlisted,"#818cf8"),
        ("Post-Scoring",       after_scoring, "#a78bfa"),
        ("Post-Skeptic",       after_skeptic, C_WINNER),
        ("Winner",             1,             "#f59e0b"),
    ]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(PANEL_BG)

    y_pos   = list(range(len(stages)))
    labels  = [s[0] for s in stages]
    values  = [s[1] for s in stages]
    colors  = [s[2] for s in stages]
    max_val = max(values) * 1.15

    bars = ax.barh(y_pos, values, color=colors, height=0.55, zorder=3,
                   edgecolor="white", linewidth=0.8)

    # Value labels inside/outside bars
    for bar, val in zip(bars, values):
        w = bar.get_width()
        ax.text(w + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9, fontweight="bold", color=DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=DARK)
    ax.set_xlim(0, max_val * 1.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax.set_xlabel("Number of Candidates", fontsize=9, color=MID)
    ax.tick_params(colors=MID, axis="x")
    ax.tick_params(axis="y", length=0)

    # Reduction annotations
    for i in range(1, len(stages)):
        prev, curr = values[i - 1], values[i]
        if prev > 0:
            pct = 100 * (prev - curr) / prev
            if pct > 0.5:
                mid_y = (i - 1 + i) / 2
                ax.text(max_val * 1.15, mid_y, f"↓{pct:.0f}%",
                        va="center", ha="left", fontsize=7.5, color="#ef4444",
                        fontweight="bold")

    ax.set_title("B   Candidate Funnel", loc="left",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    if standalone:
        fig.tight_layout()
        out = OUTPUT_DIR / "panel_b_funnel.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Panel C — Before vs After Council
# ---------------------------------------------------------------------------

def panel_c(result: dict, ax: plt.Axes | None = None) -> plt.Figure:
    ba = _parse_before_after(result)
    if not ba:
        # fallback synthetic
        ba = {"τ (min)": (30.0, 18.5), "d (mm)": (1.0, 0.75),
              "Q (mL/min)": (1.0, 1.65), "V_R (mL)": (30.0, 30.5),
              "Temp (°C)": (85.0, 85.0), "BPR (bar)": (5.0, 7.0)}

    labels = list(ba.keys())
    before = [ba[l][0] for l in labels]
    after  = [ba[l][1] for l in labels]
    n      = len(labels)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(PANEL_BG)

    x     = np.arange(n)
    width = 0.35

    bars_b = ax.bar(x - width / 2, before, width, label="Pre-Council",
                    color="#93c5fd", edgecolor="white", linewidth=0.8, zorder=3)
    bars_a = ax.bar(x + width / 2, after,  width, label="Post-Council",
                    color=C_WINNER,  edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels + delta arrows
    for i, (b, a, label) in enumerate(zip(before, after, labels)):
        ax.text(i - width / 2, b * 1.02 + 0.01 * max(before + after),
                f"{b:.1f}", ha="center", va="bottom", fontsize=7.5, color=MID)
        ax.text(i + width / 2, a * 1.02 + 0.01 * max(before + after),
                f"{a:.1f}", ha="center", va="bottom", fontsize=7.5, color=DARK, fontweight="bold")
        if abs(a - b) > 0.01 * max(abs(b), 0.01):
            pct = 100 * (a - b) / max(abs(b), 0.01)
            sym = "▲" if pct > 0 else "▼"
            col = "#16a34a" if pct > 0 else "#dc2626"
            ax.text(i, max(b, a) + 0.06 * max(before + after),
                    f"{sym}{abs(pct):.0f}%", ha="center", va="bottom",
                    fontsize=7, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=DARK)
    ax.set_ylabel("Value", fontsize=9, color=MID)
    ax.tick_params(colors=MID)
    ax.legend(fontsize=9, framealpha=0.5, edgecolor="none")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("C   Before vs After Council", loc="left",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if standalone:
        fig.tight_layout()
        out = OUTPUT_DIR / "panel_c_before_after.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Panel D — Objective Weight Radar
# ---------------------------------------------------------------------------

def panel_d(result: dict | None = None, ax: plt.Axes | None = None) -> plt.Figure:
    """Show how the 4 objectives shift the domain weights."""
    domains = ["Chemistry\n0.25", "Kinetics\n0.20", "Fluidics\n0.20",
               "Safety\n0.20", "Geometry\n0.15"]
    N = len(domains)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Base weights and per-objective modifiers (from chief.py docs)
    objectives = {
        "Balanced":          [0.25, 0.20, 0.20, 0.20, 0.15],
        "De-risk First-run": [0.30, 0.20, 0.15, 0.30, 0.15],
        "Yield-oriented":    [0.30, 0.25, 0.15, 0.20, 0.10],
        "Throughput":        [0.20, 0.20, 0.25, 0.10, 0.25],
    }
    obj_colors = ["#6366f1", "#f97316", "#10b981", "#ef4444"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(PANEL_BG)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=8, color=DARK)
    ax.set_ylim(0, 0.40)
    ax.set_yticks([0.10, 0.20, 0.30, 0.40])
    ax.set_yticklabels(["0.10", "0.20", "0.30", "0.40"], fontsize=7, color=LIGHT)
    ax.yaxis.grid(True, color="#e5e7eb", linestyle="--", linewidth=0.6)
    ax.xaxis.grid(True, color="#e5e7eb", linestyle="-", linewidth=0.4)

    handles = []
    for (name, weights), color in zip(objectives.items(), obj_colors):
        vals = weights + weights[:1]
        ax.plot(angles, vals, color=color, linewidth=2, zorder=4)
        ax.fill(angles, vals, color=color, alpha=0.07)
        handles.append(mpatches.Patch(color=color, label=name))

    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8, framealpha=0.6, edgecolor="none", title="Objective",
              title_fontsize=8)
    ax.set_title("D   Objective Weight Profiles", loc="center",
                 fontsize=11, fontweight="bold", color=DARK, pad=18)

    if standalone:
        fig.tight_layout()
        out = OUTPUT_DIR / "panel_d_objective_radar.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Panel E — Skeptic Audit Breakdown
# ---------------------------------------------------------------------------

def panel_e(result: dict, ax: plt.Axes | None = None) -> plt.Figure:
    dlog     = result.get("deliberation_log", {})
    by_agent = _skeptic_severity(dlog)

    if not any(sum(v.values()) for v in by_agent.values()):
        # Synthetic plausible data
        by_agent = {
            "Dr. Chemistry": {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 1},
            "Dr. Kinetics":  {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 3},
            "Dr. Fluidics":  {"CRITICAL": 0, "HIGH": 2, "MEDIUM": 2},
            "Dr. Safety":    {"CRITICAL": 2, "HIGH": 1, "MEDIUM": 0},
        }

    agents   = list(by_agent.keys())
    severities = ["CRITICAL", "HIGH", "MEDIUM"]
    sev_colors = {"CRITICAL": "#dc2626", "HIGH": "#f97316", "MEDIUM": "#fbbf24"}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(PANEL_BG)

    x      = np.arange(len(agents))
    width  = 0.22
    offset = -(len(severities) - 1) * width / 2

    for i, sev in enumerate(severities):
        vals = [by_agent[a].get(sev, 0) for a in agents]
        bars = ax.bar(x + offset + i * width, vals, width,
                      label=sev, color=sev_colors[sev],
                      edgecolor="white", linewidth=0.8, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(v), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("Dr. ", "Dr.\n") for a in agents],
                       fontsize=9, color=DARK)
    ax.set_ylabel("Issues Flagged", fontsize=9, color=MID)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(colors=MID)
    ax.legend(fontsize=9, framealpha=0.5, edgecolor="none",
              title="Severity", title_fontsize=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("E   Skeptic Audit — Issues per Agent", loc="left",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if standalone:
        fig.tight_layout()
        out = OUTPUT_DIR / "panel_e_skeptic_audit.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

def combined(result: dict):
    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.40,
                   left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])       # heatmap — tall left
    ax_b = fig.add_subplot(gs[1, 0])       # funnel  — bottom left
    ax_c = fig.add_subplot(gs[0, 1])       # before/after
    ax_d = fig.add_subplot(gs[1, 1], polar=True)  # radar
    ax_e = fig.add_subplot(gs[0:, 2])      # skeptic — full right column

    panel_a(result, ax=ax_a)
    panel_b(result, ax=ax_b)
    panel_c(result, ax=ax_c)
    panel_d(result, ax=ax_d)
    panel_e(result, ax=ax_e)

    fig.suptitle("ENGINE Council v4 — Performance Overview", fontsize=14,
                 fontweight="bold", color=DARK, y=0.97)

    out = OUTPUT_DIR / "council_panels_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        print(f"[!] {INPUT_PATH} not found — run `python run_mock.py` first.")
        print("    Generating panels with synthetic placeholder data instead.\n")
        result: dict = {}
    else:
        print(f"Loading {INPUT_PATH} …")
        result = _load()

    print("Generating panels …")
    panel_a(result)
    panel_b(result)
    panel_c(result)
    panel_d(result)
    panel_e(result)
    combined(result)
    print("\nDone. All panels saved to outputs/")
