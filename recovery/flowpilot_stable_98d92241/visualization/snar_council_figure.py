"""
SNAr council case-study figure — same 6-panel layout as council_stress_figure.py.

Panels:
  A  Pipeline schematic   — pre-council input through 5 stages to council output
  B  Comparison table     — pre-council vs council on key metrics
  C  Score heatmap        — 12 candidates × 4 domain agents
  D  Agent diagnosis      — what each expert caught / fixed
  E  Design space scatter — tau vs d, colour = combined score, winner starred
  F  Parameter track      — normalised before/after arrow tracks
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path

# ── data ───────────────────────────────────────────────────────────────────────
DATA = Path("outputs/snar_council_run.json")
with open(DATA) as f:
    d = json.load(f)

fp   = d["pre_council_proposal"]
cp   = d["proposal"]
fc   = d["pre_council_calculations"]
cc   = d["post_council_calculations"]
dlog = d["deliberation_log"]

# ── parse score + candidate tables from summary ────────────────────────────────
summary = dlog.get("summary", "")
score_lines, cand_data = [], {}
in_score, in_cand = False, False
for line in summary.split("\n"):
    if "| id | combined | chemistry |" in line:
        in_score = True; continue
    if "| id | τ min | d mm |" in line:
        in_cand = True; continue
    if in_score:
        if line.startswith("|---"): continue
        if not line.startswith("|"): in_score = False; continue
        p = [x.strip() for x in line.split("|")[1:-1]]
        if len(p) >= 6:
            try:
                cid = int(p[0].replace("**","").replace("★","").strip())
                score_lines.append((cid, float(p[1].replace("**","")),
                                    float(p[2].replace("**","")),
                                    float(p[3].replace("**","")),
                                    float(p[4].replace("**","")),
                                    float(p[5].replace("**",""))))
            except (ValueError, IndexError): pass
    if in_cand:
        if line.startswith("|---"): continue
        if not line.startswith("|"): in_cand = False; continue
        p = [x.strip() for x in line.split("|")[1:-1]]
        if len(p) >= 10:
            try:
                cid = int(p[0])
                cand_data[cid] = {
                    "tau": float(p[1]), "d": float(p[2]), "Q": float(p[3]),
                    "Re": float(p[6]), "r_mix": float(p[8]), "X": float(p[9]),
                }
            except (ValueError, IndexError): pass

scores_by_id  = {r[0]: r[1:] for r in score_lines}
candidate_ids = sorted(cand_data.keys())

# ── palette ────────────────────────────────────────────────────────────────────
BG     = "#0d1117"; PANEL  = "#161b22"; BORDER = "#30363d"
TEXT   = "#e6edf3"; MUTED  = "#8b949e"
RED    = "#f85149"; ORANGE = "#d29922"; GREEN  = "#3fb950"
BLUE   = "#58a6ff"; PURPLE = "#bc8cff"; GOLD   = "#ffa657"

domain_col = {"Chemistry": BLUE, "Kinetics": GREEN, "Fluidics": GOLD, "Safety": RED}

# ── layout ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.patch.set_facecolor(BG)
gs = GridSpec(4, 2, figure=fig,
              left=0.04, right=0.97, top=0.96, bottom=0.04,
              hspace=0.38, wspace=0.08,
              height_ratios=[0.85, 1.1, 1.1, 1.15])

ax_pipe  = fig.add_subplot(gs[0, :])
ax_table = fig.add_subplot(gs[1, 0])
ax_heat  = fig.add_subplot(gs[1, 1])
ax_diag  = fig.add_subplot(gs[2, 0])
ax_scat  = fig.add_subplot(gs[2, 1])
ax_bar   = fig.add_subplot(gs[3, :])

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.8)


# ══════════════════════════════════════════════════════════════════════════════
# A — Pipeline schematic
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_pipe
ax.set_xlim(0, 22); ax.set_ylim(0, 4); ax.axis("off")
ax.set_title("ENGINE Council v4 — Deliberation Pipeline",
             color=TEXT, fontsize=13, fontweight="bold", pad=6, loc="left")

def _box(ax, x, y, w, h, label, sub="", col=BLUE, fs=9):
    ax.add_patch(FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size=0.3",
        linewidth=1.2, edgecolor=col, facecolor=col+"28", zorder=3))
    ax.text(x, y+(0.12 if sub else 0), label, ha="center", va="center",
            color=TEXT, fontsize=fs, fontweight="bold", zorder=4)
    if sub:
        ax.text(x, y-0.28, sub, ha="center", va="center",
                color=MUTED, fontsize=7.5, zorder=4)

def _arr(ax, x0, x1, y=2.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2), zorder=5)

# pre-council box
ax.add_patch(FancyBboxPatch((0.2, 0.55), 2.5, 2.9,
    boxstyle="round,pad=0.0,rounding_size=0.25",
    linewidth=1.5, edgecolor=ORANGE, facecolor=ORANGE+"18", zorder=3))
ax.text(1.45, 3.15, "PRE-COUNCIL", ha="center", va="center",
        color=ORANGE, fontsize=9, fontweight="bold", zorder=4)
pre_lines = [
    f"τ = {fp['residence_time_min']:.0f} min",
    f"d = {fp['tubing_ID_mm']} mm",
    f"BPR = {fp['BPR_bar']:.0f} bar",
    f"T = {fp['temperature_C']:.0f} °C",
]
for i, line in enumerate(pre_lines):
    ax.text(1.45, 2.65 - i*0.52, line, ha="center", va="center",
            color="#ffddaa", fontsize=8, zorder=4, fontfamily="monospace")

# stages
stages = [
    (4.0,  "Stage 0", "Problem\nFraming",      BLUE),
    (6.35, "Stage 1", "Candidate\nMatrix ×12", BLUE),
    (8.7,  "Stage 2", "Domain\nScoring",       PURPLE),
    (11.05,"Stage 3", "Skeptic\nAudit",        ORANGE),
    (13.4, "Stage 4", "Chief\nSelection",      GREEN),
]
for x, s, lbl, col in stages:
    _box(ax, x, 2.0, 1.95, 1.6, s, lbl, col)
_arr(ax, 2.75, 3.05)
for i in range(len(stages)-1):
    _arr(ax, stages[i][0]+0.975, stages[i+1][0]-0.975)

# agent icons
for ax_x, lbl, col in [(7.65,"Dr.\nChem",BLUE),(8.35,"Dr.\nKin",GREEN),
                        (9.05,"Dr.\nFluid",GOLD),(9.75,"Dr.\nSafety",RED)]:
    ax.plot(ax_x, 0.75, "o", ms=14, color=col+"44", mec=col, mew=1.2, zorder=4)
    ax.text(ax_x, 0.75, lbl, ha="center", va="center",
            color=col, fontsize=5.8, fontweight="bold", zorder=5)
ax.annotate("", xy=(8.7, 1.0), xytext=(8.7, 1.2),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))

# council output box
ax.add_patch(FancyBboxPatch((14.5, 0.55), 2.6, 2.9,
    boxstyle="round,pad=0.0,rounding_size=0.25",
    linewidth=1.5, edgecolor=GREEN, facecolor=GREEN+"18", zorder=3))
ax.text(15.8, 3.15, "COUNCIL OUTPUT [OK]", ha="center", va="center",
        color=GREEN, fontsize=9, fontweight="bold", zorder=4)
for i, line in enumerate([
    f"τ = {cp['residence_time_min']:.1f} min  [OK]",
    f"d = {cp['tubing_ID_mm']} mm  [OK]",
    f"BPR = {cp['BPR_bar']} bar  [OK]",
    f"T = {cp['temperature_C']:.0f} °C  [OK]",
]):
    ax.text(15.8, 2.65-i*0.52, line, ha="center", va="center",
            color="#aaffcc", fontsize=8, zorder=4, fontfamily="monospace")
_arr(ax, 14.4, 14.5)

# winner badge
ax.add_patch(FancyBboxPatch((17.3, 1.1), 3.2, 1.8,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    linewidth=1, edgecolor=PURPLE, facecolor=PURPLE+"22", zorder=3))
ax.text(18.9, 2.55, "Winner: Candidate 4", ha="center",
        color=PURPLE, fontsize=8.5, fontweight="bold", zorder=4)
ax.text(18.9, 2.1,  "Combined score: 0.860", ha="center",
        color=TEXT, fontsize=8, zorder=4)
ax.text(18.9, 1.65, "Chem 0.88 | Kin 0.70", ha="center",
        color=MUTED, fontsize=7.5, zorder=4)
ax.text(18.9, 1.25, "Fluid 0.95 | Safety 0.95", ha="center",
        color=MUTED, fontsize=7.5, zorder=4)
_arr(ax, 17.1, 17.3)

ax.text(0.015, 0.18, "A", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# B — Comparison table
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_table
ax.axis("off")
ax.set_title("B  Pre-Council vs Council Output", color=TEXT,
             fontsize=10.5, fontweight="bold", pad=6, loc="left")

rows = [
    ("Residence time τ",  f"{fp['residence_time_min']:.0f} min",
     f"{cp['residence_time_min']:.1f} min",   "Kinetics"),
    ("Tubing diameter d", f"{fp['tubing_ID_mm']} mm",
     f"{cp['tubing_ID_mm']} mm",              "Chemistry"),
    ("Temperature",       f"{fp['temperature_C']:.0f} °C",
     f"{cp['temperature_C']:.0f} °C",         "Chemistry"),
    ("BPR setting",       f"{fp['BPR_bar']:.1f} bar",
     f"{cp['BPR_bar']} bar",                  "Safety"),
    ("Flow rate Q",       f"{fp['flow_rate_mL_min']:.4f} mL/min",
     f"{cp['flow_rate_mL_min']:.4f} mL/min",  "Fluidics"),
    ("Reynolds number",   f"{fc['reynolds_number']:.1f}",
     f"{cc['reynolds_number']:.2f}",           "Fluidics"),
    ("Conversion X",      "X ~ 0.63 (90 min)",
     "X = 0.76 (127 min)",                    "Kinetics"),
    ("BPR required?",     "False (pre-set 10 bar)",
     "False (correctly 0.6 bar)",             "Safety"),
    ("S/V ratio",         f"{fc['surface_to_volume']:.0f} m⁻¹",
     f"{cc['surface_to_volume']:.0f} m⁻¹",    "Fluidics"),
]

ax.plot([0,1],[0.93,0.93], color=BORDER, lw=0.8, transform=ax.transAxes)
ax.text(0.01, 0.97, "Parameter",    transform=ax.transAxes, color=MUTED,
        fontsize=8.5, fontweight="bold", va="top")
ax.text(0.38, 0.97, "Pre-Council",  transform=ax.transAxes, color=ORANGE,
        fontsize=8.5, fontweight="bold", va="top", ha="center")
ax.text(0.70, 0.97, "Council",      transform=ax.transAxes, color=GREEN,
        fontsize=8.5, fontweight="bold", va="top", ha="center")
ax.text(0.87, 0.97, "Domain",       transform=ax.transAxes, color=MUTED,
        fontsize=8.5, fontweight="bold", va="top", ha="center")

row_h, y0 = 0.088, 0.905
for i, (label, pv, cv, dom) in enumerate(rows):
    y = y0 - i*row_h
    if i%2 == 0:
        ax.add_patch(FancyBboxPatch((0.0, y-0.038), 1.0, row_h-0.004,
            boxstyle="square,pad=0", facecolor="#ffffff08", edgecolor="none",
            transform=ax.transAxes, zorder=1))
    dcol = domain_col.get(dom, MUTED)
    ax.add_patch(FancyBboxPatch((0.83, y-0.025), 0.16, 0.048,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=dcol+"30", edgecolor=dcol, linewidth=0.7,
        transform=ax.transAxes, zorder=2))
    ax.text(0.91, y+0.002, dom[:5], transform=ax.transAxes,
            color=dcol, fontsize=6.8, va="center", ha="center", fontweight="bold")
    ax.text(0.01, y+0.003, label, transform=ax.transAxes,
            color=TEXT, fontsize=8.0, va="center")
    ax.text(0.38, y+0.003, pv, transform=ax.transAxes,
            color="#ffddaa", fontsize=7.8, va="center", ha="center")
    ax.text(0.70, y+0.003, cv, transform=ax.transAxes,
            color="#aaffcc", fontsize=7.8, va="center", ha="center")

ax.text(0.01, 0.01, "B", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# C — Score heatmap
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_heat
ax.set_title("C  Domain Score Heatmap — All 12 Candidates", color=TEXT,
             fontsize=10.5, fontweight="bold", pad=6, loc="left")

domains_order = ["Chemistry", "Kinetics", "Fluidics", "Safety", "Combined"]
n_c, n_d = len(candidate_ids), len(domains_order)
heat_data = np.zeros((n_d, n_c))
for j, cid in enumerate(candidate_ids):
    row = scores_by_id.get(cid, (0.5,)*5)
    combined, chem, kin, fluid, safety = row
    for i, v in enumerate([chem, kin, fluid, safety, combined]):
        heat_data[i, j] = v

_cmap = LinearSegmentedColormap.from_list("c", ["#f85149","#d29922","#3fb950"], N=256)
im = ax.imshow(heat_data, cmap=_cmap, vmin=0, vmax=1,
               aspect="auto", interpolation="nearest")
ax.set_xticks(range(n_c))
ax.set_xticklabels([f"C{c}" for c in candidate_ids], color=TEXT, fontsize=7.5)
ax.set_yticks(range(n_d))
ax.set_yticklabels(domains_order, color=TEXT, fontsize=8.5)
ax.tick_params(colors=MUTED, length=3)

for i in range(n_d):
    for j in range(n_c):
        v = heat_data[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=6.5, color="#000" if v > 0.55 else TEXT, fontweight="bold")

# highlight winner (C4)
if 4 in candidate_ids:
    wj = candidate_ids.index(4)
    for i in range(n_d):
        ax.add_patch(plt.Rectangle((wj-0.5, i-0.5), 1, 1,
            lw=2.0, edgecolor=GOLD, facecolor="none", zorder=5))
    ax.text(wj, n_d-0.05, "★", ha="center", va="bottom", color=GOLD,
            fontsize=11, fontweight="bold",
            transform=ax.get_xaxis_transform(), zorder=6)

ax.axhline(3.5, color=BORDER, lw=1.2, ls="--")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors=MUTED, labelsize=7)
cbar.outline.set_edgecolor(BORDER)
ax.text(0.01, -0.08, "C", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════════════════════
# D — Agent diagnosis
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_diag
ax.axis("off")
ax.set_title("D  Agent Diagnoses & Fixes", color=TEXT,
             fontsize=10.5, fontweight="bold", pad=6, loc="left")

diagnoses = [
    ("Dr. Chemistry", BLUE,
     "Pre:  Concentration 0.3 M (too high); d=1.6 mm chosen",
     "Fix:  d = 0.75 mm, conc. 0.15 M — optimal SNAr window",
     "Rule: SNAr sweet spot 0.1–0.2 M. No photonics needed;\n"
     "FEP compatible with DMF at 150°C (50°C safety margin)."),
    ("Dr. Kinetics", GREEN,
     "Pre:  τ = 90 min → X = 0.63, barely in range",
     "Fix:  τ = 127 min → X = 0.76 (closer to target)",
     "Rule: SNAr at 150°C: τ_kinetics ≈ 48 min, τ/τ_k = 2.65×.\n"
     "Laminar RTD: centreflow sees τ/2 = 64 min, still above τ_k."),
    ("Dr. Fluidics", GOLD,
     "Pre:  d=1.6 mm → r_mix=0.237 (borderline), Re=8",
     "Fix:  d=0.75 mm → r_mix=0.018 (excellent), Re=2",
     "Rule: r_mix < 0.10 target; d reduction improves mixing.\n"
     "Q = 0.056 mL/min → syringe pump; ΔP = 0.018 bar."),
    ("Dr. Safety", RED,
     "Pre:  BPR = 10 bar (over-specified; DMF bp 153°C, T=150°C)",
     "Fix:  BPR = 0.6 bar (P_vap + ΔP + 0.5 bar margin)",
     "Rule: DMF at 150°C: P_vap ~ 0.1 bar → min BPR 0.6 bar.\n"
     "Da_thermal ~ 0.03 (isothermal). DMF = SVHC, closed system."),
]

y0, row_h = 0.95, 0.235
for i, (name, col, before, after, rule) in enumerate(diagnoses):
    y = y0 - i*row_h
    ax.add_patch(FancyBboxPatch(
        (0.0, y-row_h+0.015), 0.18, row_h-0.025,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=col+"28", edgecolor=col, linewidth=1.0,
        transform=ax.transAxes, zorder=2))
    ax.text(0.09, y-row_h/2+0.01, name, transform=ax.transAxes,
            color=col, fontsize=8, fontweight="bold", va="center", ha="center")
    ax.text(0.21, y-0.01,   before, transform=ax.transAxes,
            color="#ffddaa", fontsize=7.5, va="top")
    ax.text(0.21, y-0.055,  after,  transform=ax.transAxes,
            color="#aaffcc", fontsize=7.5, va="top")
    ax.text(0.21, y-0.105,  rule,   transform=ax.transAxes,
            color=MUTED, fontsize=6.8, va="top", linespacing=1.4)
    if i < len(diagnoses)-1:
        ax.plot([0,1], [y-row_h+0.01]*2, color=BORDER, lw=0.6,
                transform=ax.transAxes)

ax.text(0.01, 0.01, "D", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# E — Design space scatter
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_scat
ax.set_facecolor(PANEL)
ax.set_title("E  Design Space: τ vs d  (colour = combined score)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=6, loc="left")

_cmap2 = LinearSegmentedColormap.from_list("c2",
    ["#f85149","#d29922","#3fb950"], N=256)

taus   = [cand_data[c]["tau"]  for c in candidate_ids]
ds     = [cand_data[c]["d"]    for c in candidate_ids]
combos = [scores_by_id.get(c,(0.5,))[0] for c in candidate_ids]

sc = ax.scatter(taus, ds, c=combos, cmap=_cmap2, vmin=0, vmax=1,
                s=160, zorder=4, edgecolors=BORDER, linewidths=0.7)
for cid, tau, dv, sc_ in zip(candidate_ids, taus, ds, combos):
    ax.text(tau, dv+0.028, f"C{cid}", ha="center", va="bottom",
            fontsize=6.5, color="#cccccc", zorder=5)

# winner (C4)
if 4 in candidate_ids:
    wi = candidate_ids.index(4)
    ax.scatter([taus[wi]], [ds[wi]], s=340, marker="*", color=GOLD,
               zorder=6, edgecolors="#000", linewidths=0.5)
    ax.annotate(f" [OK] C4 (score={combos[wi]:.3f})",
                xy=(taus[wi], ds[wi]), xytext=(taus[wi]-25, ds[wi]+0.07),
                color=GOLD, fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8), zorder=7)

# pre-council marker
ax.scatter([fp["residence_time_min"]], [fp["tubing_ID_mm"]],
           marker="X", s=280, color=ORANGE, zorder=7,
           edgecolors="#000", linewidths=0.6)
ax.annotate(" Pre-council\n input",
            xy=(fp["residence_time_min"], fp["tubing_ID_mm"]),
            xytext=(fp["residence_time_min"]-25, fp["tubing_ID_mm"]-0.15),
            color=ORANGE, fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8), zorder=8)

ax.set_xlabel("Residence time τ  (min)", color=MUTED, fontsize=9)
ax.set_ylabel("Tubing ID  d  (mm)",      color=MUTED, fontsize=9)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_xlim(35, 145)
ax.set_ylim(0.55, 1.75)
for sp in ax.spines.values():
    sp.set_color(BORDER)
cbar2 = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar2.set_label("Combined score", color=MUTED, fontsize=8)
cbar2.ax.tick_params(colors=MUTED, labelsize=7)
cbar2.outline.set_edgecolor(BORDER)
ax.text(0.01, 0.96, "E", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════════════════════
# F — Parameter track chart
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_bar
ax.set_facecolor(PANEL)
ax.set_title(
    "F  Parameter Changes: Pre-Council → Council  "
    "(each metric scaled independently; arrow shows direction of fix)",
    color=TEXT, fontsize=10.5, fontweight="bold", pad=6, loc="left")

# (label, pre, post, unit, higher_is_better, domain)
param_defs = [
    ("Residence time τ",  fp["residence_time_min"],   cp["residence_time_min"],
     "min",    True,  "Kinetics"),
    ("Tubing diameter d", fp["tubing_ID_mm"],          cp["tubing_ID_mm"],
     "mm",     False, "Chemistry"),
    ("BPR setting",       fp["BPR_bar"],               cp["BPR_bar"],
     "bar",    False, "Safety"),   # lower is better (right-size, not over-spec)
    ("Flow rate Q",       fp["flow_rate_mL_min"],      cp["flow_rate_mL_min"],
     "mL/min", False, "Fluidics"),
    ("Reynolds number",   fc["reynolds_number"],       cc["reynolds_number"],
     "",       False, "Fluidics"),
    ("Conversion X",      0.63,                        0.76,
     "",       True,  "Kinetics"),
    ("Surface/Volume S/V",fc["surface_to_volume"],     cc["surface_to_volume"],
     "m⁻¹",   True,  "Fluidics"),
    ("Mixing ratio r_mix",fc["damkohler_mass"],         cc["damkohler_mass"],
     "",       False, "Fluidics"),
]

n_rows = len(param_defs)
ys = np.arange(n_rows-1, -1, -1, dtype=float)
ax.set_xlim(-0.12, 1.28); ax.set_ylim(-0.7, n_rows-0.3); ax.axis("off")

for i, (label, fv, cv, unit, hib, dom) in enumerate(param_defs):
    y  = ys[i]
    dc = domain_col.get(dom, MUTED)
    lo, hi = min(fv, cv), max(fv, cv)
    if hi == lo:
        x_f, x_c = 0.0, 1.0
    else:
        x_f = (fv-lo)/(hi-lo)
        x_c = (cv-lo)/(hi-lo)
    if not hib:
        x_f, x_c = 1-x_f, 1-x_c

    ax.plot([0,1],[y,y], color=BORDER, lw=2.5, solid_capstyle="round", zorder=1)
    ax.annotate("", xy=(x_c, y), xytext=(x_f, y),
                arrowprops=dict(arrowstyle="-|>", color=dc, lw=1.8,
                                mutation_scale=14), zorder=3)
    ax.scatter([x_f], [y], s=110, color=ORANGE, zorder=4,
               edgecolors="#000", lw=0.5)
    ax.scatter([x_c], [y], s=140, color=GREEN, marker="*", zorder=5,
               edgecolors="#000", lw=0.4)

    fv_s = f"{fv:.3g} {unit}".strip()
    cv_s = f"{cv:.3g} {unit}".strip()
    ha_f = "right" if x_f < 0.5 else "left"
    off_f = -0.045 if x_f < 0.5 else 0.045
    ax.text(x_f+off_f, y+0.28, fv_s, ha=ha_f, va="center",
            color="#ffddaa", fontsize=7.5, fontweight="bold")
    ha_c = "left" if x_c > x_f else "right"
    off_c = 0.045 if x_c > x_f else -0.045
    ax.text(x_c+off_c, y-0.32, cv_s, ha=ha_c, va="center",
            color="#aaffcc", fontsize=7.5, fontweight="bold")
    ax.text(-0.01, y, label, ha="right", va="center", color=TEXT, fontsize=8.2)
    ax.add_patch(FancyBboxPatch((1.03, y-0.22), 0.22, 0.44,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=dc+"30", edgecolor=dc, linewidth=0.7,
        transform=ax.transData, zorder=2))
    ax.text(1.14, y, dom[:5], ha="center", va="center",
            color=dc, fontsize=7, fontweight="bold")

ax.legend(handles=[
    Line2D([0],[0],marker="o",color="none",mfc=ORANGE,ms=9,
           label="Pre-council",mec="#000"),
    Line2D([0],[0],marker="*",color="none",mfc=GREEN,ms=11,
           label="Council output",mec="#000"),
], loc="lower right", fontsize=8.5, facecolor=PANEL,
   edgecolor=BORDER, labelcolor=TEXT, framealpha=0.9)
ax.text(0.0, 1.01, "F", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom")


# ── global title ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.993,
         "ENGINE Council v4 — SNAr Case Study: Thermal Reaction Design",
         ha="center", va="top", color=TEXT, fontsize=15, fontweight="bold")
fig.text(0.5, 0.977,
         "4-Fluoronitrobenzene + Piperazine · DMF · 150 °C · "
         "6 h batch (81% yield) · No photocatalyst",
         ha="center", va="top", color=MUTED, fontsize=9.5)

# ── save ──────────────────────────────────────────────────────────────────────
OUT_PNG = Path("outputs/snar_council_figure.png")
OUT_SVG = Path("outputs/snar_council_figure.svg")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {OUT_PNG}")
print(f"Saved -> {OUT_SVG}")
plt.close(fig)
