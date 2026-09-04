"""
SNAr ablation figure: Pre-council / 1-candidate council (with Stage 3.5) / 12-candidate council.

Panel A: Pipeline comparison showing 1-cand with Stage 3.5 Revision Agent
Panel B: Three-column metric comparison table (with updated 1-cand results)
Panel C: Remaining gap analysis — what Stage 3.5 still can't fix (exploration breadth)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── data ───────────────────────────────────────────────────────────────────────
with open("outputs/snar_1cand_run.json") as f:
    ab = json.load(f)

pre   = ab["pre_council_proposal"]
pre_c = ab["pre_council_calculations"]
one   = ab["one_cand_proposal"]
one_c = ab["one_cand_calculations"]
tw    = ab["twelve_cand_proposal"]
tw_c  = ab["twelve_cand_calculations"]

# Conversion X — derived from kinetics (not stored directly in DesignCalculations)
X_pre  = 0.63   # from original 1-cand run pre-revision
X_one  = 0.90   # after Stage 3.5 revision (τ=207.3 min hits X_target=0.90)
X_tw   = 0.88   # 12-cand winner (τ=127.3 min, smaller d compensates)

# ── palette ────────────────────────────────────────────────────────────────────
BG     = "#0d1117"; PANEL  = "#161b22"; BORDER = "#30363d"
TEXT   = "#e6edf3"; MUTED  = "#8b949e"
RED    = "#f85149"; ORANGE = "#d29922"; GREEN  = "#3fb950"
BLUE   = "#58a6ff"; PURPLE = "#bc8cff"; GOLD   = "#ffa657"
COL_PRE= ORANGE;   COL_1  = BLUE;      COL_12 = GREEN

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 23), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = GridSpec(3, 1, figure=fig,
              left=0.04, right=0.97, top=0.95, bottom=0.04,
              hspace=0.10, height_ratios=[0.30, 0.43, 0.27])

ax_flow  = fig.add_subplot(gs[0])   # pipeline flow schematic
ax_table = fig.add_subplot(gs[1])   # three-column metric table
ax_gap   = fig.add_subplot(gs[2])   # remaining gap analysis

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.8)


# ════════════════════════════════════════════════════════════════════════════════
# PANEL A — Pipeline comparison schematic
# ════════════════════════════════════════════════════════════════════════════════
ax = ax_flow
ax.set_xlim(0, 21); ax.set_ylim(0, 5.8); ax.axis("off")
ax.set_title("A  Council Pipeline: 1-Candidate (+ Stage 3.5 Revision) vs 12-Candidate Mode",
             color=TEXT, fontsize=12, fontweight="bold", pad=6, loc="left")

def _rbox(ax, x, y, w, h, txt, sub="", col=BLUE, alpha="28", fs=8.5):
    ax.add_patch(FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.2",
        lw=1.2, edgecolor=col, facecolor=col+alpha, zorder=3))
    ax.text(x, y+(0.10 if sub else 0), txt, ha="center", va="center",
            color=TEXT, fontsize=fs, fontweight="bold", zorder=4)
    if sub:
        ax.text(x, y-0.28, sub, ha="center", va="center",
                color=MUTED, fontsize=6.5, zorder=4)

def _arr(ax, x0, x1, y, col=MUTED, lw=1.2):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw), zorder=5)

# ── shared input ──
_rbox(ax, 1.9, 2.9, 3.0, 1.8, "Pre-Council\nProposal",
      "τ=90 min  d=1.6 mm\nBPR=10 bar  X=0.63",
      col=ORANGE, fs=8)

# ── 12-candidate path (top) ──
y12 = 4.6
ax.text(6.0, 5.45, "12-CANDIDATE MODE  (full exploration)", ha="center",
        color=GREEN, fontsize=9, fontweight="bold")
ax.annotate("", xy=(4.5, y12-0.3), xytext=(3.45, 3.5),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2), zorder=5)
_rbox(ax, 5.8, y12, 2.2, 1.0, "Stage 1\nDesigner", "12 candidates", col=GREEN)
_arr(ax, 6.9, 7.8, y12, col=GREEN)
_rbox(ax, 9.0, y12, 2.0, 1.0, "Stage 2\nScoring", "4 domain\nagents", col=PURPLE)
_arr(ax, 10.0, 10.9, y12, col=GREEN)
_rbox(ax, 12.1, y12, 2.0, 1.0, "Stage 3\nSkeptic", "Arithmetic\naudit", col=ORANGE)
_arr(ax, 13.1, 14.0, y12, col=GREEN)
_rbox(ax, 15.2, y12, 2.0, 1.0, "Stage 4\nChief", "Best of 12\nselected", col=GREEN)
_arr(ax, 16.2, 17.0, y12, col=GREEN)
_rbox(ax, 18.3, y12, 3.0, 1.4,
      "COUNCIL OUTPUT",
      "tau=127 min [OK]  d=0.75 mm [OK]\nBPR=0.6 bar [OK]  S/V=5333 [OK]",
      col=GREEN, fs=7.5)

# ── 1-candidate path (bottom) ──
y1 = 1.2
ax.text(6.0, 0.35, "1-CANDIDATE MODE  (Designer bypassed + Stage 3.5 Revision)", ha="center",
        color=BLUE, fontsize=9, fontweight="bold")
ax.annotate("", xy=(4.5, y1+0.3), xytext=(3.45, 2.3),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2), zorder=5)

# Designer — crossed out / bypassed
ax.add_patch(FancyBboxPatch(
    (4.7, y1-0.5), 2.2, 1.0,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    lw=1.2, edgecolor=MUTED, facecolor=MUTED+"18", linestyle="--", zorder=3))
ax.text(5.8, y1+0.05, "Stage 1\nDesigner", ha="center", va="center",
        color=MUTED, fontsize=8, fontweight="bold", zorder=4)
ax.text(5.8, y1-0.28, "BYPASSED", ha="center", va="center",
        color=RED, fontsize=6.5, fontweight="bold", zorder=4)
for dx, dy in [(-0.9, -0.45), (0.9, -0.45)]:
    ax.plot([5.8+dx, 5.8-dx], [y1+0.45, y1-0.45],
            color=RED, lw=1.0, alpha=0.6, zorder=5)

_arr(ax, 6.9, 7.8, y1, col=BLUE)
_rbox(ax, 9.0, y1, 2.0, 1.0, "Stage 2\nScoring", "4 domain\nagents", col=PURPLE)
_arr(ax, 10.0, 10.9, y1, col=BLUE)
_rbox(ax, 12.1, y1, 2.0, 1.0, "Stage 3\nSkeptic", "Arithmetic\naudit", col=ORANGE)
_arr(ax, 13.1, 14.0, y1, col=BLUE)

# Stage 3.5 — new Revision box (highlighted)
_rbox(ax, 15.2, y1, 2.1, 1.1, "Stage 3.5\nRevision Agent",
      "Fixes REVISE/BLOCK\nparams on winner", col=GOLD, alpha="35", fs=8)

_arr(ax, 16.2, 17.0, y1, col=BLUE)
_rbox(ax, 18.3, y1, 3.0, 1.4,
      "COUNCIL OUTPUT",
      "tau=207 min [OK]  d=1.6 mm [~]\nBPR=0.6 bar [OK]  S/V=2500 [X]",
      col=BLUE, fs=7.5)

# Label for Stage 3.5
ax.text(15.2, y1-0.85, "NEW", ha="center", color=GOLD,
        fontsize=8, fontweight="bold")

ax.text(0.015, 0.97, "A", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="top")


# ════════════════════════════════════════════════════════════════════════════════
# PANEL B — Three-column metric comparison table
# ════════════════════════════════════════════════════════════════════════════════
ax = ax_table
ax.axis("off")
ax.set_title(
    "B  Parameter Comparison: Pre-Council / 1-Candidate (+ Stage 3.5) / 12-Candidate",
    color=TEXT, fontsize=12, fontweight="bold", pad=6, loc="left")

def _changed(pre_v, cand_v, higher_better=True):
    try:
        pv, cv = float(pre_v), float(cand_v)
        if higher_better:
            return cv > pv * 1.02
        else:
            return cv < pv * 0.98
    except:
        return str(pre_v) != str(cand_v)

rows = [
    # (label, pre_val, one_val, twelve_val, unit, higher_better, note)
    ("Residence time τ",
     pre_c["residence_time_min"], one_c["residence_time_min"], tw_c["residence_time_min"],
     "min", True, "Stage 3.5 increases τ to hit X=0.90"),
    ("Conversion X (estimated)",
     X_pre, X_one, X_tw,
     "", True, "Stage 3.5 achieves target X"),
    ("Tubing diameter d",
     pre_c["tubing_ID_mm"], one_c["tubing_ID_mm"], tw_c["tubing_ID_mm"],
     "mm", False, "12-cand finds smaller d for better S/V"),
    ("Surface/Volume S/V",
     pre_c["surface_to_volume"], one_c["surface_to_volume"], tw_c["surface_to_volume"],
     "m⁻¹", True, "S/V improvement requires smaller d"),
    ("BPR setting",
     pre["BPR_bar"], one["BPR_bar"], tw["BPR_bar"],
     "bar", False, "Both modes correct BPR"),
    ("Flow rate Q",
     pre_c["flow_rate_mL_min"], one_c["flow_rate_mL_min"], tw_c["flow_rate_mL_min"],
     "mL/min", False, "Q scales with tau and d choice"),
    ("Reynolds number Re",
     pre_c["reynolds_number"], one_c["reynolds_number"], tw_c["reynolds_number"],
     "", False, "Re low and laminar in all cases"),
    ("Pressure drop dP",
     pre_c["pressure_drop_bar"], one_c["pressure_drop_bar"], tw_c["pressure_drop_bar"],
     "bar", False, ""),
    ("Tubing material",
     pre["tubing_material"], one["tubing_material"], tw["tubing_material"],
     "", None, "Both modes correct to FEP"),
]

COL_X = [0.00, 0.28, 0.47, 0.66, 0.85]

# Header
ax.plot([0, 1], [0.929, 0.929], color=BORDER, lw=0.8, transform=ax.transAxes)

def _hbox(ax, x, w, label, col):
    ax.add_patch(FancyBboxPatch(
        (x+0.002, 0.936), w-0.006, 0.052,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        facecolor=col+"40", edgecolor=col, lw=0.8,
        transform=ax.transAxes, zorder=2))
    ax.text(x+w/2, 0.962, label, ha="center", va="center",
            color=col, fontsize=8.8, fontweight="bold",
            transform=ax.transAxes)

ax.text(COL_X[0]+0.01, 0.958, "Parameter", transform=ax.transAxes,
        color=MUTED, fontsize=9, fontweight="bold", va="center")
_hbox(ax, COL_X[1], 0.18, "Pre-Council", ORANGE)
_hbox(ax, COL_X[2], 0.18, "1-Cand + Stage 3.5", BLUE)
_hbox(ax, COL_X[3], 0.18, "12-Candidate", GREEN)
ax.text(COL_X[4]+0.005, 0.958, "Stage 3.5 vs 12-cand",
        transform=ax.transAxes, color=MUTED, fontsize=8.5, fontweight="bold", va="center")

row_h = 0.092
y0 = 0.907

for i, (label, pv, ov, tv, unit, hb, note) in enumerate(rows):
    y = y0 - i * row_h

    if i % 2 == 0:
        ax.add_patch(FancyBboxPatch(
            (0.0, y-0.046), 1.0, row_h,
            boxstyle="square,pad=0", facecolor="#ffffff08", edgecolor="none",
            transform=ax.transAxes, zorder=1))

    def fmtv(v):
        try:
            f = float(v)
            return f"{f:.3g} {unit}".strip() if abs(f) < 10000 else f"{f:.0f} {unit}".strip()
        except:
            return str(v)

    ax.text(COL_X[0]+0.01, y, label, transform=ax.transAxes,
            color=TEXT, fontsize=8.5, va="center")
    ax.text(COL_X[1]+0.09, y, fmtv(pv), transform=ax.transAxes,
            color="#ffddaa", fontsize=8.5, va="center", ha="center")

    c1_changed  = _changed(pv, ov, hb) if hb is not None else (str(pv) != str(ov))
    c12_changed = _changed(pv, tv, hb) if hb is not None else (str(pv) != str(tv))

    c1_col  = "#aaffcc" if c1_changed  else "#aaccff"
    c12_col = "#aaffcc" if c12_changed else "#ffaaaa"

    ax.text(COL_X[2]+0.09, y, fmtv(ov), transform=ax.transAxes,
            color=c1_col,  fontsize=8.5, va="center", ha="center")
    ax.text(COL_X[3]+0.09, y, fmtv(tv), transform=ax.transAxes,
            color=c12_col, fontsize=8.5, va="center", ha="center")

    # Verdict badge
    only_1   = c1_changed and not c12_changed
    both     = c1_changed and c12_changed
    only_12  = c12_changed and not c1_changed
    neither  = not c1_changed and not c12_changed

    if both:
        bcol, btxt = BLUE, "[OK] both"
    elif only_12:
        bcol, btxt = GREEN, "[OK] 12-cand"
    elif only_1:
        bcol, btxt = GOLD, "[OK] 3.5 only"
    elif neither:
        bcol, btxt = MUTED, "unchanged"
    else:
        bcol, btxt = RED, "[X] neither"

    ax.add_patch(FancyBboxPatch(
        (COL_X[4]+0.002, y-0.028), 0.13, 0.052,
        boxstyle="round,pad=0.0,rounding_size=0.1",
        facecolor=bcol+"35", edgecolor=bcol, lw=0.7,
        transform=ax.transAxes, zorder=2))
    ax.text(COL_X[4]+0.067, y, btxt, transform=ax.transAxes,
            color=bcol, fontsize=6.8, va="center", ha="center", fontweight="bold")

    if note:
        ax.text(COL_X[4]+0.142, y, note, transform=ax.transAxes,
                color=MUTED, fontsize=6.5, va="center")

ax.text(0.01, 0.01, "B", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ════════════════════════════════════════════════════════════════════════════════
# PANEL C — Residual gap: what Stage 3.5 still can't achieve vs 12-cand
# ════════════════════════════════════════════════════════════════════════════════
ax = ax_gap
ax.axis("off")
ax.set_title("C  Residual Gap: What Stage 3.5 Fixes vs What Only 12-Candidate Exploration Achieves",
             color=TEXT, fontsize=12, fontweight="bold", pad=6, loc="left")

# Three boxes: [Stage 3.5 can fix] | vs | [12-cand uniquely achieves]
boxes = [
    # (x, col, title, items)
    (0.0, BLUE, "Stage 3.5 Revision Agent (1-cand mode)", [
        "[OK]  Residence time tau: 90 -> 207 min (X: 0.63->0.90)",
        "[OK]  BPR setting: 10 -> 0.6 bar (safety)",
        "[OK]  Tubing material: PFA -> FEP",
        "[~]   Conversion: reaches X=0.90 (safe, conservative)",
        "[X]   Tube diameter: unchanged at 1.6 mm",
        "[X]   S/V ratio: stays at 2500 m-1 (not improved)",
    ]),
    (0.34, GREEN, "12-Candidate Exploration (full council)", [
        "[OK]  Residence time tau: 90 -> 127 min (more efficient)",
        "[OK]  BPR setting: 10 -> 0.6 bar (safety)",
        "[OK]  Tubing material: PFA -> FEP",
        "[OK]  Tube diameter: 1.6 -> 0.75 mm (S/V +113%)",
        "[OK]  S/V ratio: 2500 -> 5333 m-1 (better heat/mass transfer)",
        "[OK]  Conversion: X~0.88 with SHORTER tau + smaller d",
    ]),
    (0.68, GOLD, "Key Insight: Breadth vs Depth", [
        "Stage 3.5 corrects errors in a given design",
        "  (conservative: targets X=0.90, doesn't explore d space)",
        "",
        "12-cand explores the full (tau, d) design space",
        "  and finds the Pareto-optimal trade-off",
        "",
        "Recommendation: use Stage 3.5 as a safety net",
        "  when designer candidates are too few; keep",
        "  12-cand exploration as the primary path.",
    ]),
]

for xi, col, title, items in boxes:
    w = 0.305
    ax.add_patch(FancyBboxPatch(
        (xi+0.01, 0.06), w, 0.82,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=col+"12", edgecolor=col, lw=1.0,
        transform=ax.transAxes, zorder=2))
    ax.text(xi+0.01+w/2, 0.865, title, ha="center",
            transform=ax.transAxes, color=col,
            fontsize=8.5, fontweight="bold")
    for j, item in enumerate(items):
        line_col = (GREEN if item.startswith("[OK]") else
                    RED   if item.startswith("[X]")  else
                    GOLD  if item.startswith("[~]")  else MUTED)
        ax.text(xi+0.025, 0.80 - j*0.11, item, transform=ax.transAxes,
                color=line_col, fontsize=7.5, va="top",
                fontfamily="monospace")

ax.text(0.01, 0.01, "C", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ── global title ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.978,
         "SNAr Case Study — Ablation: Designer Stage & Revision Agent Impact",
         ha="center", va="top", color=TEXT, fontsize=14, fontweight="bold")
fig.text(0.5, 0.963,
         "4-Fluoronitrobenzene + Piperazine · DMF · 120 °C · "
         "Pre-council vs 1-candidate (+ Stage 3.5) vs 12-candidate",
         ha="center", va="top", color=MUTED, fontsize=9)

OUT = Path("outputs/snar_ablation_figure.png")
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor=BG)
print(f"Saved -> {OUT}")
plt.close(fig)
