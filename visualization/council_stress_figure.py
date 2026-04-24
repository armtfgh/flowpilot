"""
Council stress-test results figure.

Five panels:
  A  Pipeline schematic   — flawed input → 5 stages → corrected output
  B  Comparison table     — flawed vs council on key metrics
  C  Score heatmap        — 12 candidates × 4 domain agents
  D  Agent diagnosis      — what each expert caught / fixed
  E  Design space scatter — τ vs d, candidates coloured by score, winner starred
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── load data ──────────────────────────────────────────────────────────────────
DATA = Path("outputs/council_stress_test.json")
with open(DATA) as f:
    d = json.load(f)

fp   = d["flawed_proposal"]
cp   = d["council_proposal"]
fc   = d["flawed_calculations"]
cc   = d["council_calculations"]
dlog = d["deliberation_log"]

# Extract per-candidate scores from summary markdown table
summary = dlog.get("summary", "")

# Parse score table from summary
import re
score_lines = []
in_score_table = False
for line in summary.split("\n"):
    if "| id | combined | chemistry |" in line:
        in_score_table = True
        continue
    if in_score_table:
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 6:
            try:
                cid      = int(parts[0].replace("**","").replace("★","").strip())
                combined = float(parts[1].replace("**",""))
                chem     = float(parts[2].replace("**",""))
                kin      = float(parts[3].replace("**",""))
                fluid    = float(parts[4].replace("**",""))
                safety   = float(parts[5].replace("**",""))
                score_lines.append((cid, combined, chem, kin, fluid, safety))
            except (ValueError, IndexError):
                pass

# Also parse candidate table for τ, d, Re etc.
cand_data = {}
in_cand_table = False
for line in summary.split("\n"):
    if "| id | τ min | d mm |" in line:
        in_cand_table = True
        continue
    if in_cand_table:
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 11:
            try:
                cid = int(parts[0])
                cand_data[cid] = {
                    "tau":  float(parts[1]),
                    "d":    float(parts[2]),
                    "Q":    float(parts[3]),
                    "V_R":  float(parts[4]),
                    "L":    float(parts[5]),
                    "Re":   float(parts[6]),
                    "dP":   float(parts[7]),
                    "r_mix":float(parts[8]),
                    "X":    float(parts[9]),
                }
            except (ValueError, IndexError):
                pass

scores_by_id = {row[0]: row[1:] for row in score_lines}  # id → (combined,chem,kin,fluid,safety)
candidate_ids = sorted(cand_data.keys())

# ── colour palette ─────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
RED     = "#f85149"
ORANGE  = "#d29922"
GREEN   = "#3fb950"
BLUE    = "#58a6ff"
PURPLE  = "#bc8cff"
GOLD    = "#ffa657"

AGENT_COLORS = {
    "Chemistry": "#58a6ff",
    "Kinetics":  "#3fb950",
    "Fluidics":  "#ffa657",
    "Safety":    "#f85149",
    "Combined":  "#bc8cff",
}

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = GridSpec(
    4, 2,
    figure=fig,
    left=0.04, right=0.97,
    top=0.96, bottom=0.04,
    hspace=0.38, wspace=0.08,
    height_ratios=[0.85, 1.1, 1.1, 1.15],
)

ax_pipe  = fig.add_subplot(gs[0, :])   # full-width pipeline
ax_table = fig.add_subplot(gs[1, 0])   # comparison table
ax_heat  = fig.add_subplot(gs[1, 1])   # score heatmap
ax_diag  = fig.add_subplot(gs[2, 0])   # agent diagnosis
ax_scat  = fig.add_subplot(gs[2, 1])   # scatter τ vs d
ax_bar   = fig.add_subplot(gs[3, :])   # parameter delta bars

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Pipeline schematic
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_pipe
ax.set_xlim(0, 22)
ax.set_ylim(0, 4)
ax.axis("off")
ax.set_title("ENGINE Council v4 — Deliberation Pipeline",
             color=TEXT, fontsize=13, fontweight="bold", pad=6, loc="left")

def _rounded_box(ax, x, y, w, h, label, sublabel="", color=BLUE, text_color=TEXT,
                 fontsize=9.5, sublabel_fontsize=7.5, radius=0.3):
    fancy = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=1.2, edgecolor=color, facecolor=color + "28",
        zorder=3,
    )
    ax.add_patch(fancy)
    ax.text(x, y + (0.12 if sublabel else 0), label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha="center", va="center",
                color=MUTED, fontsize=sublabel_fontsize, zorder=4)

def _arrow(ax, x0, x1, y=2.0, color=MUTED):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                zorder=5)

# ── Flawed input box ──
flawed_lines = [
    "FLAWED INPUT",
    "τ = 4 min  [X]",
    "d = 2.0 mm [X]",
    "BPR = 0 bar [X]",
    "Q = 2.0 mL/min [X]",
]
box_input = FancyBboxPatch(
    (0.2, 0.55), 2.5, 2.9,
    boxstyle="round,pad=0.0,rounding_size=0.25",
    linewidth=1.5, edgecolor=RED, facecolor=RED + "18", zorder=3,
)
ax.add_patch(box_input)
ax.text(1.45, 3.15, "FLAWED INPUT", ha="center", va="center",
        color=RED, fontsize=9, fontweight="bold", zorder=4)
for i, line in enumerate(flawed_lines[1:], 1):
    ax.text(1.45, 3.15 - i * 0.52, line, ha="center", va="center",
            color=TEXT if "[X]" not in line else "#ffaaaa",
            fontsize=8, zorder=4, fontfamily="monospace")

# ── pipeline stages ──
stages = [
    (4.0,  "Stage 0", "Problem\nFraming",      BLUE),
    (6.35, "Stage 1", "Candidate\nMatrix ×12", BLUE),
    (8.7,  "Stage 2", "Domain\nScoring",       PURPLE),
    (11.05,"Stage 3", "Skeptic\nAudit",        ORANGE),
    (13.4, "Stage 4", "Chief\nSelection",      GREEN),
]
for x, stage, label, col in stages:
    _rounded_box(ax, x, 2.0, 1.95, 1.6, stage, label, color=col, fontsize=9)

_arrow(ax, 2.75, 3.05)
for i in range(len(stages) - 1):
    _arrow(ax, stages[i][0] + 0.975, stages[i+1][0] - 0.975)

# ── agent icons under Stage 2 ──
agent_info = [
    (7.65, "Dr.\nChem", BLUE),
    (8.35, "Dr.\nKin",  GREEN),
    (9.05, "Dr.\nFluid",GOLD),
    (9.75, "Dr.\nSafety",RED),
]
for ax_x, lbl, col in agent_info:
    ax.plot(ax_x, 0.75, "o", ms=14, color=col + "44", mec=col, mew=1.2, zorder=4)
    ax.text(ax_x, 0.75, lbl, ha="center", va="center",
            color=col, fontsize=5.8, fontweight="bold", zorder=5)

ax.annotate("", xy=(8.7, 1.0), xytext=(8.7, 1.2),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))

# ── Council output box ──
box_out = FancyBboxPatch(
    (14.5, 0.55), 2.6, 2.9,
    boxstyle="round,pad=0.0,rounding_size=0.25",
    linewidth=1.5, edgecolor=GREEN, facecolor=GREEN + "18", zorder=3,
)
ax.add_patch(box_out)
ax.text(15.8, 3.15, "COUNCIL OUTPUT ★", ha="center", va="center",
        color=GREEN, fontsize=9, fontweight="bold", zorder=4)
out_lines = [
    f"τ = {cp['residence_time_min']} min  [OK]",
    f"d = {cp['tubing_ID_mm']} mm  [OK]",
    f"BPR = {cp['BPR_bar']} bar  [OK]",
    f"Q = {cp['flow_rate_mL_min']:.3f} mL/min  [OK]",
]
for i, line in enumerate(out_lines):
    ax.text(15.8, 2.65 - i * 0.52, line, ha="center", va="center",
            color="#aaffcc", fontsize=8, zorder=4, fontfamily="monospace")

_arrow(ax, 14.4, 14.5)

# ── score badge ──
badge = FancyBboxPatch(
    (17.3, 1.1), 3.2, 1.8,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    linewidth=1, edgecolor=PURPLE, facecolor=PURPLE + "22", zorder=3,
)
ax.add_patch(badge)
ax.text(18.9, 2.55, "Winner: Candidate 11", ha="center",
        color=PURPLE, fontsize=8.5, fontweight="bold", zorder=4)
ax.text(18.9, 2.1, "Combined score: 0.854", ha="center",
        color=TEXT, fontsize=8, zorder=4)
ax.text(18.9, 1.65, "Chem 0.94 | Kin 0.85", ha="center",
        color=MUTED, fontsize=7.5, zorder=4)
ax.text(18.9, 1.25, "Fluid 0.72 | Safety 1.00", ha="center",
        color=MUTED, fontsize=7.5, zorder=4)

_arrow(ax, 17.1, 17.3)

ax.text(0.15, 0.18, "A", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Comparison table
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_table
ax.axis("off")
ax.set_title("B  Flawed Input vs Council Output", color=TEXT,
             fontsize=10.5, fontweight="bold", pad=6, loc="left")

rows = [
    # (label, flawed_val, council_val, domain, lower_is_better_or_none)
    ("τ  residence time",     f"{fp['residence_time_min']:.0f} min",  f"{cp['residence_time_min']:.0f} min",
     "Kinetics",  False),
    ("d  tubing diameter",    f"{fp['tubing_ID_mm']:.1f} mm",        f"{cp['tubing_ID_mm']:.2f} mm",
     "Chemistry", False),  # smaller is better here
    ("Q  flow rate",          f"{fp['flow_rate_mL_min']:.2f} mL/min",f"{cp['flow_rate_mL_min']:.3f} mL/min",
     "Fluidics",  False),
    ("BPR  back-pressure",    f"{fp['BPR_bar']:.1f} bar",            f"{cp['BPR_bar']:.2f} bar",
     "Safety",    False),
    ("Re  Reynolds number",   f"{fc['reynolds_number']:.0f}",        f"{cc['reynolds_number']:.0f}",
     "Fluidics",  False),
    ("Beer-Lambert  A",       "A ≈ 1.50  [X]",                         "A = 0.375  [OK]",
     "Chemistry", False),
    ("Conversion  X",         "X ≈ 0.63  [X]",                        "X = 0.86  [OK]",
     "Kinetics",  False),
    ("BPR required?",         "Yes [X] (not set)",                      "Yes [OK] (1.38 bar set)",
     "Safety",    None),
    ("S/V ratio",             f"{fc['surface_to_volume']:.0f} m⁻¹",  f"{cc['surface_to_volume']:.0f} m⁻¹",
     "Fluidics",  False),
]

domain_col = {
    "Chemistry": BLUE,
    "Kinetics":  GREEN,
    "Fluidics":  GOLD,
    "Safety":    RED,
}

col_labels = ["Parameter", "Flawed", "Council", "Domain"]
col_x      = [0.01, 0.38, 0.60, 0.87]
col_align  = ["left", "center", "center", "center"]

ax.text(col_x[0], 0.97, col_labels[0], transform=ax.transAxes,
        color=MUTED, fontsize=8.5, fontweight="bold", va="top")
ax.text(col_x[1], 0.97, col_labels[1], transform=ax.transAxes,
        color=RED, fontsize=8.5, fontweight="bold", va="top", ha="center")
ax.text(col_x[2] + 0.1, 0.97, col_labels[2], transform=ax.transAxes,
        color=GREEN, fontsize=8.5, fontweight="bold", va="top", ha="center")
ax.text(col_x[3], 0.97, col_labels[3], transform=ax.transAxes,
        color=MUTED, fontsize=8.5, fontweight="bold", va="top", ha="center")

# header underline
ax.plot([0, 1], [0.93, 0.93], color=BORDER, lw=0.8, transform=ax.transAxes)

row_h = 0.088
y0    = 0.905
for i, (label, fv, cv, domain, _) in enumerate(rows):
    y = y0 - i * row_h
    if i % 2 == 0:
        bg = FancyBboxPatch(
            (0.0, y - 0.038), 1.0, row_h - 0.004,
            boxstyle="square,pad=0",
            facecolor="#ffffff08", edgecolor="none",
            transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(bg)

    # domain pill
    dcol = domain_col.get(domain, MUTED)
    pill = FancyBboxPatch(
        (0.83, y - 0.025), 0.16, 0.048,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=dcol + "30", edgecolor=dcol,
        linewidth=0.7,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(pill)
    ax.text(0.91, y + 0.002, domain[:5], transform=ax.transAxes,
            color=dcol, fontsize=6.8, va="center", ha="center", fontweight="bold")

    ax.text(col_x[0], y + 0.003, label, transform=ax.transAxes,
            color=TEXT, fontsize=8.0, va="center")
    ax.text(col_x[1], y + 0.003, fv, transform=ax.transAxes,
            color="#ffaaaa", fontsize=7.8, va="center", ha="center")
    ax.text(col_x[2] + 0.1, y + 0.003, cv, transform=ax.transAxes,
            color="#aaffcc", fontsize=7.8, va="center", ha="center")

ax.text(0.01, 0.01, "B", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Score heatmap
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_heat
ax.set_title("C  Domain Score Heatmap — All 12 Candidates", color=TEXT,
             fontsize=10.5, fontweight="bold", pad=6, loc="left")

domains_order = ["Chemistry", "Kinetics", "Fluidics", "Safety", "Combined"]
score_keys    = [1, 2, 3, 4, 0]  # index into (combined,chem,kin,fluid,safety)

n_cands  = len(candidate_ids)
n_domain = len(domains_order)

heat_data = np.zeros((n_domain, n_cands))
for j, cid in enumerate(candidate_ids):
    row = scores_by_id.get(cid, (0.5, 0.5, 0.5, 0.5, 0.5))
    combined, chem, kin, fluid, safety = row
    vals = [chem, kin, fluid, safety, combined]
    for i in range(n_domain):
        heat_data[i, j] = vals[i]

from matplotlib.colors import LinearSegmentedColormap
_cmap = LinearSegmentedColormap.from_list(
    "council", ["#f85149", "#d29922", "#3fb950"], N=256
)

im = ax.imshow(heat_data, cmap=_cmap, vmin=0, vmax=1.0,
               aspect="auto", interpolation="nearest")

ax.set_xticks(range(n_cands))
ax.set_xticklabels([f"C{cid}" for cid in candidate_ids],
                   color=TEXT, fontsize=7.5)
ax.set_yticks(range(n_domain))
ax.set_yticklabels(domains_order, color=TEXT, fontsize=8.5)
ax.tick_params(colors=MUTED, length=3)

for i in range(n_domain):
    for j in range(n_cands):
        v = heat_data[i, j]
        txt_col = "#000000" if v > 0.55 else TEXT
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=6.5, color=txt_col, fontweight="bold")

# highlight winner (candidate 11 → index = candidate_ids.index(11))
if 11 in candidate_ids:
    win_j = candidate_ids.index(11)
    for i in range(n_domain):
        rect = plt.Rectangle(
            (win_j - 0.5, i - 0.5), 1, 1,
            linewidth=2.0, edgecolor=GOLD, facecolor="none", zorder=5,
        )
        ax.add_patch(rect)
    ax.text(win_j, n_domain - 0.05, "★", ha="center", va="bottom",
            color=GOLD, fontsize=11, fontweight="bold",
            transform=ax.get_xaxis_transform(), zorder=6)

# divider line before Combined row
ax.axhline(3.5, color=BORDER, lw=1.2, ls="--")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors=MUTED, labelsize=7)
cbar.outline.set_edgecolor(BORDER)

ax.text(0.01, -0.08, "C", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Agent diagnosis
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_diag
ax.axis("off")
ax.set_title("D  Agent Diagnoses & Fixes", color=TEXT,
             fontsize=10.5, fontweight="bold", pad=6, loc="left")

diagnoses = [
    ("Dr. Chemistry",
     BLUE,
     "Flawed: d = 2.0 mm → A = 1.50 (inner-filter [X])",
     "Fixed:  d = 0.75 mm → A = 0.375 (moderate [OK])",
     "Rule: Beer-Lambert A > 0.8 → opaque core; d ≤ 0.75 mm\n"
     "for ε ≈ 50 M⁻¹cm⁻¹ at C = 0.1 M"),

    ("Dr. Kinetics",
     GREEN,
     "Flawed: τ = 4 min → X = 0.63, IF = 30× ([X])",
     "Fixed:  τ = 8 min → X = 0.86, IF = 15× ([OK])",
     "Rule: Photoredox IF 4–8×; X ≥ 0.85 required.\n"
     "Candidate 11 (τ=8 min) is the only one passing."),

    ("Dr. Fluidics",
     GOLD,
     "Flawed: Q = 2.0 mL/min → r_mix = 0.29 (borderline)",
     "Fixed:  Q = 1.05 mL/min, Re = 63, ΔP = 0.16 bar ([OK])",
     "Rule: r_mix < 0.20 target; reduce d, not increase.\n"
     "d = 0.75 mm improves S/V and keeps ΔP < 0.2 bar."),

    ("Dr. Safety",
     RED,
     "Flawed: BPR = 0 bar; MeCN bp = 82°C, T = 85°C ([X])",
     "Fixed:  BPR = 1.38 bar; P_vap + ΔP + margin ([OK])",
     "Rule: BPR ≥ P_vap(T) + ΔP + 0.5 bar.\n"
     "All candidates flagged BPR REVISE until set correctly."),
]

y0    = 0.95
row_h = 0.235
for i, (name, col, before, after, rule) in enumerate(diagnoses):
    y = y0 - i * row_h

    # agent label pill
    pill = FancyBboxPatch(
        (0.0, y - row_h + 0.015), 0.18, row_h - 0.025,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=col + "28", edgecolor=col, linewidth=1.0,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(pill)
    ax.text(0.09, y - row_h/2 + 0.01, name, transform=ax.transAxes,
            color=col, fontsize=8, fontweight="bold", va="center", ha="center")

    ax.text(0.21, y - 0.01, before, transform=ax.transAxes,
            color="#ffaaaa", fontsize=7.5, va="top")
    ax.text(0.21, y - 0.055, after, transform=ax.transAxes,
            color="#aaffcc", fontsize=7.5, va="top")
    ax.text(0.21, y - 0.105, rule, transform=ax.transAxes,
            color=MUTED, fontsize=6.8, va="top", linespacing=1.4)

    if i < len(diagnoses) - 1:
        ax.plot([0, 1], [y - row_h + 0.01, y - row_h + 0.01],
                color=BORDER, lw=0.6, transform=ax.transAxes)

ax.text(0.01, 0.01, "D", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL E — Design space scatter (τ vs d, coloured by combined score)
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_scat
ax.set_facecolor(PANEL)
ax.set_title("E  Design Space: τ vs d  (colour = combined score)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=6, loc="left")

_cmap2 = LinearSegmentedColormap.from_list(
    "council2", ["#f85149", "#d29922", "#3fb950"], N=256
)

taus   = [cand_data[cid]["tau"]  for cid in candidate_ids]
ds     = [cand_data[cid]["d"]    for cid in candidate_ids]
combos = [scores_by_id.get(cid, (0.5,))[0] for cid in candidate_ids]
Xs     = [cand_data[cid]["X"]    for cid in candidate_ids]

sc = ax.scatter(taus, ds, c=combos, cmap=_cmap2, vmin=0, vmax=1,
                s=160, zorder=4, edgecolors=BORDER, linewidths=0.7)

# label each point
for cid, tau, d_val, score in zip(candidate_ids, taus, ds, combos):
    txt_col = "#cccccc"
    ax.text(tau, d_val + 0.025, f"C{cid}", ha="center", va="bottom",
            fontsize=6.5, color=txt_col, zorder=5)

# highlight winner
if 11 in candidate_ids:
    wi = candidate_ids.index(11)
    ax.scatter([taus[wi]], [ds[wi]], s=340, marker="*", color=GOLD,
               zorder=6, edgecolors="#000000", linewidths=0.5)
    ax.annotate(
        f" ★ C11 (score={combos[wi]:.3f})",
        xy=(taus[wi], ds[wi]), xytext=(taus[wi] + 0.4, ds[wi] + 0.06),
        color=GOLD, fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8),
        zorder=7,
    )

# flawed input marker
ax.scatter([fp["residence_time_min"]], [fp["tubing_ID_mm"]],
           marker="X", s=280, color=RED, zorder=7, edgecolors="#000",
           linewidths=0.6)
ax.annotate(
    " [X] Flawed\n input",
    xy=(fp["residence_time_min"], fp["tubing_ID_mm"]),
    xytext=(fp["residence_time_min"] + 0.3, fp["tubing_ID_mm"] - 0.12),
    color=RED, fontsize=8, fontweight="bold",
    arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
    zorder=8,
)

# X = 0.85 iso-contour (approximately)
tau_line = np.linspace(1.5, 9.5, 200)
# X = 1 - exp(-tau/tau_k); tau_k ~ 15 min → X=0.85 at tau≈33 min (off scale)
# Use IF-based: tau_k from batch_h=8h → IF=7.5× → tau_k=64min
# X=0.85 → tau = -ln(0.15)*tau_k ≈ 1.897 * tau_k
# But these kinetics agents used tau_k~15min → tau*=28min, also off scale.
# Just draw the "sufficient conversion" boundary from score color.

ax.set_xlabel("Residence time τ  (min)", color=MUTED, fontsize=9)
ax.set_ylabel("Tubing ID  d  (mm)",       color=MUTED, fontsize=9)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_xlim(1.0, 9.5)
ax.set_ylim(0.35, 1.15)
for sp in ax.spines.values():
    sp.set_color(BORDER)

cbar2 = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar2.set_label("Combined score", color=MUTED, fontsize=8)
cbar2.ax.tick_params(colors=MUTED, labelsize=7)
cbar2.outline.set_edgecolor(BORDER)

ax.text(0.01, 0.96, "E", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL F — Horizontal "before → after" dot-and-arrow chart, one row per metric
# Each metric is normalised independently to [0, 1] so all rows are readable.
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_bar
ax.set_facecolor(PANEL)
ax.set_title(
    "F  Parameter Changes: Flawed → Council  "
    "(each metric scaled independently; arrow shows direction of fix)",
    color=TEXT, fontsize=10.5, fontweight="bold", pad=6, loc="left",
)

# (row_label, flawed, council, unit_str, higher_is_better_for_this_metric, domain)
param_defs = [
    ("Residence time τ",  fp["residence_time_min"],   cp["residence_time_min"],
     "min",    True,  "Kinetics"),
    ("Tubing diameter d", fp["tubing_ID_mm"],          cp["tubing_ID_mm"],
     "mm",     False, "Chemistry"),
    ("BPR setting",       fp["BPR_bar"],               cp["BPR_bar"],
     "bar",    True,  "Safety"),
    ("Flow rate Q",       fp["flow_rate_mL_min"],      cp["flow_rate_mL_min"],
     "mL/min", False, "Fluidics"),
    ("Reynolds number",   fc["reynolds_number"],       cc["reynolds_number"],
     "",       False, "Fluidics"),
    ("Conversion X",      0.63,                        0.86,
     "",       True,  "Kinetics"),
    ("Surface/Volume S/V",fc["surface_to_volume"],     cc["surface_to_volume"],
     "m⁻¹",   True,  "Fluidics"),
    ("Beer-Lambert A",    1.50,                        0.375,
     "",       False, "Chemistry"),
]

n_rows = len(param_defs)
# one y position per metric (top-to-bottom)
ys = np.arange(n_rows - 1, -1, -1, dtype=float)

# x-axis: 0 = "worst" (flawed), 1 = "best" (council), per metric
x_bad    = np.zeros(n_rows)
x_good   = np.ones(n_rows)

ax.set_xlim(-0.12, 1.28)
ax.set_ylim(-0.7, n_rows - 0.3)
ax.axis("off")

for i, (label, fv, cv, unit, hib, dom) in enumerate(param_defs):
    y  = ys[i]
    dc = domain_col.get(dom, MUTED)

    # normalised positions
    lo, hi = min(fv, cv), max(fv, cv)
    if hi == lo:
        x_f, x_c = 0.0, 1.0
    else:
        x_f = (fv - lo) / (hi - lo)
        x_c = (cv - lo) / (hi - lo)
    # flip so "correct" side is always x=1
    if not hib:
        x_f, x_c = 1 - x_f, 1 - x_c

    # background track
    ax.plot([0, 1], [y, y], color=BORDER, lw=2.5, solid_capstyle="round", zorder=1)

    # arrow flawed → council
    ax.annotate(
        "",
        xy=(x_c, y), xytext=(x_f, y),
        arrowprops=dict(
            arrowstyle="-|>", color=dc, lw=1.8,
            mutation_scale=14,
        ),
        zorder=3,
    )

    # flawed dot
    ax.scatter([x_f], [y], s=110, color=RED,  zorder=4, edgecolors="#000", lw=0.5)
    # council dot
    ax.scatter([x_c], [y], s=140, color=GREEN, marker="*", zorder=5,
               edgecolors="#000", lw=0.4)

    # flawed value label (left or right depending on position)
    fv_fmt  = f"{fv:.1f} {unit}" if isinstance(fv, float) else f"{fv} {unit}"
    cv_fmt  = f"{cv:.3g} {unit}" if isinstance(cv, float) else f"{cv} {unit}"
    # flawed label
    offset_f = -0.045 if x_f < 0.5 else +0.045
    ha_f     = "right" if x_f < 0.5 else "left"
    ax.text(x_f + offset_f, y + 0.28, fv_fmt.strip(), ha=ha_f, va="center",
            color="#ffaaaa", fontsize=7.5, fontweight="bold")
    # council label
    offset_c = +0.045 if x_c > x_f else -0.045
    ha_c     = "left"  if x_c > x_f else "right"
    ax.text(x_c + offset_c, y - 0.32, cv_fmt.strip(), ha=ha_c, va="center",
            color="#aaffcc", fontsize=7.5, fontweight="bold")

    # metric label (left margin)
    ax.text(-0.01, y, label, ha="right", va="center",
            color=TEXT, fontsize=8.2)

    # domain pill (right margin)
    pill = FancyBboxPatch(
        (1.03, y - 0.22), 0.22, 0.44,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=dc + "30", edgecolor=dc, linewidth=0.7,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(pill)
    ax.text(1.14, y, dom[:5], ha="center", va="center",
            color=dc, fontsize=7, fontweight="bold")

# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o",  color="none", mfc=RED,   ms=9,  label="Flawed input",    mec="#000"),
    Line2D([0], [0], marker="*",  color="none", mfc=GREEN, ms=11, label="Council output",  mec="#000"),
]
ax.legend(handles=legend_elements, loc="lower right",
          fontsize=8.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
          framealpha=0.9)

ax.text(0.0, 1.01, "F", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ── global title ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.993,
         "ENGINE Council v4 — Stress Test: Deliberate Flaw Correction",
         ha="center", va="top", color=TEXT, fontsize=15, fontweight="bold")
fig.text(0.5, 0.977,
         "Ir(ppy)₃ photoredox C–H functionalisation · MeCN · 85 °C · "
         "Flawed input: 4 deliberate engineering errors injected",
         ha="center", va="top", color=MUTED, fontsize=9.5)

# ── save ──────────────────────────────────────────────────────────────────────
OUT_PNG = Path("outputs/council_stress_figure.png")
OUT_SVG = Path("outputs/council_stress_figure.svg")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT_PNG}")
print(f"Saved → {OUT_SVG}")
plt.close(fig)
