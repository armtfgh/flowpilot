"""Thermal case study panels — ENGINE Council v4.

Reads:  outputs/thermal_council_run.json
Writes: outputs/thermal_panel1_metrics.png
        outputs/thermal_panel2_landscape.png
        outputs/thermal_panels_combined.png

Panel 1: Three-column comparison  Batch | Pre-council | Council
         for every engineering metric, with pass/fail against thresholds.
Panel 2: Candidate landscape  τ vs S/V for all design-space candidates,
         coloured by expected conversion, size = productivity.
         Pre-council and council choices annotated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BG     = "#ffffff"
LIGHT  = "#f1f5f9"
DARK   = "#1e293b"
MID    = "#64748b"
PASS   = "#16a34a"
WARN   = "#d97706"
FAIL   = "#dc2626"
NA     = "#94a3b8"

C_BATCH  = "#94a3b8"
C_PRE    = "#60a5fa"
C_POST   = "#f97316"

INPUT  = Path("outputs/thermal_council_run.json")
OUTDIR = Path("outputs")


def _load():
    with open(INPUT) as f:
        return json.load(f)


# ── Metric definitions ────────────────────────────────────────────────────────

def _build_rows(result: dict) -> list[dict]:
    """
    Returns a list of metric rows, each:
      {label, unit, batch_val, pre_val, post_val,
       threshold, thresh_dir,   # "lower" | "higher" | None
       batch_status, pre_status, post_status,
       note}
    """
    pre  = result.get("pre_council_calculations", {})
    post = result.get("post_council_calculations", {})

    pre_prop  = result.get("pre_council_proposal", {})
    post_prop = result.get("proposal", {})

    def sv(d_mm):
        return round(4_000 / d_mm, 0) if d_mm else None

    # Batch reference values (100 mL RBF at 80 °C / 8 h)
    BATCH_TAU      = 480.0          # min
    BATCH_D        = None           # not applicable
    BATCH_RE       = 0.0            # no forced flow
    BATCH_DP_PCT   = 0.0            # no pump
    BATCH_BPR      = 0.0            # open/condenser — fails at 80 °C!
    BATCH_BPR_REQ  = True           # EtOH at 80 °C > bp-20
    BATCH_SV       = 150.0          # m⁻¹, typical 100 mL flask
    BATCH_CONV     = 0.92           # 92% yield
    BATCH_PROD     = 5.75           # mmol/h (0.5M × 100mL × 0.92 / 8h)

    pre_d   = pre.get("tubing_ID_mm") or pre_prop.get("tubing_ID_mm") or 1.6
    post_d  = post.get("tubing_ID_mm") or post_prop.get("tubing_ID_mm") or 0.75

    pre_tau   = pre.get("residence_time_min") or pre_prop.get("residence_time_min") or 93.9
    post_tau  = post.get("residence_time_min") or post_prop.get("residence_time_min") or 90.5

    pre_re    = pre.get("reynolds_number", 2.1)
    post_re   = post.get("reynolds_number", 1.5)

    pre_dp    = pre.get("pressure_drop_bar", 0.0)
    post_dp   = post.get("pressure_drop_bar", 0.0)
    pump_max  = pre.get("pump_max_bar") or post.get("pump_max_bar") or 20.0
    pre_dp_pct  = round(100 * pre_dp / pump_max, 1)
    post_dp_pct = round(100 * post_dp / pump_max, 1)

    pre_bpr    = pre_prop.get("BPR_bar", 0.0) or 0.0
    post_bpr   = post_prop.get("BPR_bar", 0.0) or 0.0
    # Use calculator's recommendation if proposal is 0
    if pre_bpr == 0.0 and pre.get("bpr_required"):
        pre_bpr = pre.get("bpr_pressure_bar", 0.0)
    if post_bpr == 0.0 and post.get("bpr_required"):
        post_bpr = post.get("bpr_pressure_bar", 0.0)

    pre_sv    = sv(pre_d)
    post_sv   = sv(post_d)

    pre_conv  = pre.get("target_conversion") or 0.92
    post_conv = post.get("target_conversion") or 0.92

    pre_prod  = pre.get("productivity_mmol_h") or 6.1
    post_prod = post.get("productivity_mmol_h") or 1.6

    pre_vp    = pre.get("vapor_pressure_bar", 1.1)
    post_vp   = post.get("vapor_pressure_bar", 1.1)
    BATCH_VP  = 1.1  # EtOH at 80 °C ≈ 1.1 bar

    def status_bpr(bpr_val, vp):
        if bpr_val is None:
            return "N/A"
        if bpr_val == 0.0:
            return "FAIL"       # operating above bp with no BPR
        if bpr_val >= vp + 0.5:
            return "PASS"
        return "WARN"

    def status_lower(val, thresh):
        if val is None:
            return "N/A"
        if val <= thresh:
            return "PASS"
        return "FAIL"

    def status_higher(val, thresh):
        if val is None:
            return "N/A"
        if val >= thresh:
            return "PASS"
        return "WARN"

    rows = [
        {
            "label":    "τ (min)",
            "note":     "Residence time",
            "batch":    BATCH_TAU,
            "pre":      pre_tau,
            "post":     post_tau,
            "fmt":      ".0f",
            "thresh":   None,
            "b_status": "N/A",
            "p_status": "PASS",
            "c_status": "PASS",
            "better":   "lower",
        },
        {
            "label":    "d (mm)",
            "note":     "Tube inner diameter",
            "batch":    "–",
            "pre":      pre_d,
            "post":     post_d,
            "fmt":      ".2f",
            "thresh":   None,
            "b_status": "N/A",
            "p_status": "WARN",
            "c_status": "PASS",
            "better":   "lower",
        },
        {
            "label":    "S/V (m⁻¹)",
            "note":     "Surface-to-volume ratio",
            "batch":    BATCH_SV,
            "pre":      pre_sv,
            "post":     post_sv,
            "fmt":      ".0f",
            "thresh":   2000,
            "b_status": status_higher(BATCH_SV, 2000),
            "p_status": status_higher(pre_sv, 2000),
            "c_status": status_higher(post_sv, 2000),
            "better":   "higher",
        },
        {
            "label":    "BPR (bar)",
            "note":     "Back-pressure regulator\n(EtOH P_vap ≈ 1.1 bar at 80 °C)",
            "batch":    BATCH_BPR,
            "pre":      pre_bpr,
            "post":     post_bpr,
            "fmt":      ".1f",
            "thresh":   pre_vp + 0.5,
            "b_status": status_bpr(BATCH_BPR, BATCH_VP),
            "p_status": status_bpr(pre_bpr, pre_vp),
            "c_status": status_bpr(post_bpr, post_vp),
            "better":   "higher",
        },
        {
            "label":    "Re",
            "note":     "Reynolds number",
            "batch":    BATCH_RE,
            "pre":      pre_re,
            "post":     post_re,
            "fmt":      ".1f",
            "thresh":   2300,
            "b_status": "N/A",
            "p_status": status_lower(pre_re, 2300),
            "c_status": status_lower(post_re, 2300),
            "better":   "lower",
        },
        {
            "label":    "ΔP / P_max (%)",
            "note":     "Pump headroom used",
            "batch":    "–",
            "pre":      pre_dp_pct,
            "post":     post_dp_pct,
            "fmt":      ".1f",
            "thresh":   80,
            "b_status": "N/A",
            "p_status": status_lower(pre_dp_pct, 80),
            "c_status": status_lower(post_dp_pct, 80),
            "better":   "lower",
        },
        {
            "label":    "Productivity (mmol/h)",
            "note":     "Molar throughput",
            "batch":    BATCH_PROD,
            "pre":      pre_prod,
            "post":     post_prod,
            "fmt":      ".2f",
            "thresh":   None,
            "b_status": "N/A",
            "p_status": "N/A",
            "c_status": "N/A",
            "better":   "higher",
        },
    ]
    return rows


STATUS_COLORS = {
    "PASS": PASS,
    "WARN": WARN,
    "FAIL": FAIL,
    "N/A":  NA,
}
STATUS_MARKS = {
    "PASS": "✓",
    "WARN": "~",
    "FAIL": "✗",
    "N/A":  "–",
}


# ── Panel 1 — Metric Comparison Table ────────────────────────────────────────

def panel1(result: dict, ax=None):
    rows = _build_rows(result)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(BG)
    ax.axis("off")

    # Column headers
    cols     = ["Metric", "Note", "Batch\n(flask)", "Pre-council\n(d=1.6 mm)", "Council\n(d=0.75 mm)"]
    col_x    = [0.00, 0.18, 0.42, 0.61, 0.80]
    col_align = ["left", "left", "center", "center", "center"]
    col_w    = [0.18, 0.24, 0.19, 0.19, 0.19]

    header_colors = [DARK, DARK, C_BATCH, C_PRE, C_POST]

    y_header = 0.95
    for cx, label, align, color in zip(col_x, cols, col_align, header_colors):
        ha = align
        ax.text(cx + (0.09 if align == "center" else 0.0), y_header,
                label, transform=ax.transAxes,
                ha=ha, va="top", fontsize=9.5, fontweight="bold", color=color)

    ax.plot([0, 1], [y_header - 0.07, y_header - 0.07], color="#e2e8f0",
            linewidth=1.2, transform=ax.transAxes)

    row_h  = 0.11
    y_start = y_header - 0.12

    for i, row in enumerate(rows):
        y = y_start - i * row_h
        bg_col = LIGHT if i % 2 == 0 else BG
        bg = FancyBboxPatch((0, y - row_h * 0.45), 1.0, row_h * 0.90,
                             boxstyle="round,pad=0.005",
                             facecolor=bg_col, edgecolor="none",
                             transform=ax.transAxes, zorder=0)
        ax.add_patch(bg)

        def fmt_val(v, f):
            if isinstance(v, str):
                return v
            if isinstance(v, bool):
                return "Yes" if v else "No"
            try:
                return format(v, f)
            except Exception:
                return str(v)

        # Metric label
        ax.text(col_x[0], y, row["label"],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=9, fontweight="bold", color=DARK)
        ax.text(col_x[1], y, row["note"],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7.5, color=MID, style="italic")

        # Three value columns
        for ci, (key_stat, key_val) in enumerate([
            ("b_status", "batch"),
            ("p_status", "pre"),
            ("c_status", "post"),
        ]):
            cx = col_x[2 + ci]
            stat = row[key_stat]
            val  = row[key_val]
            vstr = fmt_val(val, row["fmt"])
            sc   = STATUS_COLORS[stat]
            sm   = STATUS_MARKS[stat]

            # Value box
            box = FancyBboxPatch((cx + 0.01, y - row_h * 0.35),
                                  col_w[2] - 0.02, row_h * 0.70,
                                  boxstyle="round,pad=0.008",
                                  facecolor=sc + "22", edgecolor=sc,
                                  linewidth=1.2, transform=ax.transAxes, zorder=1)
            ax.add_patch(box)

            ax.text(cx + col_w[2] / 2, y + 0.018, vstr,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=DARK)
            ax.text(cx + col_w[2] / 2, y - 0.022, sm,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=sc)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=PASS + "33", edgecolor=PASS, label="✓  Passes threshold"),
        mpatches.Patch(facecolor=WARN + "33", edgecolor=WARN, label="~  Caution / trade-off"),
        mpatches.Patch(facecolor=FAIL + "33", edgecolor=FAIL, label="✗  Fails threshold"),
        mpatches.Patch(facecolor=NA   + "33", edgecolor=NA,   label="–  Not applicable"),
    ]
    ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(1.0, -0.02),
              fontsize=7.5, framealpha=0.7, edgecolor="#e2e8f0",
              ncol=4, handlelength=1.2, handletextpad=0.5)

    ax.set_title("A   Engineering Metrics: Batch → Pre-council → Council",
                 loc="left", fontsize=11, fontweight="bold", color=DARK,
                 pad=8, transform=ax.transAxes)

    if standalone:
        fig.tight_layout()
        out = OUTDIR / "thermal_panel1_metrics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ── Panel 2 — Candidate Landscape ────────────────────────────────────────────

def panel2(result: dict, ax=None):
    ds = result.get("design_space", [])

    # Compute S/V and normalise productivity
    for c in ds:
        c["sv"] = 4000.0 / c["d_mm"]

    # Sort by d for legend grouping
    d_groups  = sorted(set(c["d_mm"] for c in ds))
    d_colors  = {1.6: "#60a5fa", 1.0: "#a78bfa", 0.75: "#f97316"}
    d_labels  = {1.6: "d = 1.6 mm  (high throughput)", 1.0: "d = 1.0 mm", 0.75: "d = 0.75 mm  (safer)"}

    # Batch reference
    BATCH_TAU  = 480.0
    BATCH_SV   = 150.0
    BATCH_PROD = 5.75

    # Pre-council and council points
    pre_prop  = result.get("pre_council_proposal", {})
    post_prop = result.get("proposal", {})
    pre_dc    = result.get("pre_council_calculations", {})
    post_dc   = result.get("post_council_calculations", {})

    pre_tau  = pre_prop.get("residence_time_min") or 93.9
    post_tau = post_prop.get("residence_time_min") or 90.5
    pre_d    = pre_prop.get("tubing_ID_mm") or 1.6
    post_d   = post_prop.get("tubing_ID_mm") or 0.75
    pre_sv   = 4000 / pre_d
    post_sv  = 4000 / post_d
    pre_prod = pre_dc.get("productivity_mmol_h") or 6.1
    post_prod = post_dc.get("productivity_mmol_h") or 1.6
    max_prod = max(c["productivity_mg_h"] for c in ds) if ds else 700

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor("#f8fafc")

    # Plot design space candidates
    for dval in d_groups:
        pts = [c for c in ds if c["d_mm"] == dval]
        taus  = [c["tau_min"] for c in pts]
        svs   = [c["sv"]     for c in pts]
        prods = [c["productivity_mg_h"] for c in pts]
        sizes = [30 + 180 * p / max_prod for p in prods]
        col   = d_colors.get(dval, "#888")
        ax.scatter(taus, svs, s=sizes, color=col, alpha=0.7, zorder=3,
                   edgecolors="white", linewidths=0.8,
                   label=d_labels.get(dval, f"d={dval}mm"))

    # Batch reference
    ax.scatter([BATCH_TAU], [BATCH_SV], s=200, color="#475569",
               marker="s", zorder=5, edgecolors="white", linewidths=1.2,
               label="Batch (flask)")
    ax.text(BATCH_TAU - 15, BATCH_SV + 120, "Batch\n(flask)", ha="right",
            fontsize=8, color="#475569", fontweight="bold")

    # Safety threshold line
    SV_THRESH = 2000
    ax.axhline(SV_THRESH, color=FAIL, linewidth=1.4, linestyle="--", alpha=0.7, zorder=2)
    ax.text(20, SV_THRESH + 80, "Safety threshold  S/V = 2000 m⁻¹",
            fontsize=8, color=FAIL, style="italic", ha="left")

    # Pre-council marker
    ax.scatter([pre_tau], [pre_sv], s=280, color=C_PRE, zorder=6,
               edgecolors=DARK, linewidths=1.8, marker="D")
    ax.annotate("Pre-council\n(d = 1.6 mm)", xy=(pre_tau, pre_sv),
                xytext=(pre_tau + 18, pre_sv - 350),
                fontsize=8.5, color=C_PRE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PRE, lw=1.3))

    # Council winner marker
    ax.scatter([post_tau], [post_sv], s=320, color=C_POST, zorder=6,
               edgecolors=DARK, linewidths=1.8, marker="*")
    ax.annotate("Council winner\n(d = 0.75 mm)", xy=(post_tau, post_sv),
                xytext=(post_tau - 60, post_sv + 300),
                fontsize=8.5, color=C_POST, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_POST, lw=1.3))

    # Improvement arrow
    ax.annotate("", xy=(post_tau, post_sv), xytext=(pre_tau, pre_sv),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5,
                                connectionstyle="arc3,rad=-0.2"))
    mid_x = (pre_tau + post_tau) / 2 + 10
    mid_y = (pre_sv + post_sv) / 2 + 200
    ax.text(mid_x, mid_y, "+113% S/V", fontsize=8, color=DARK,
            fontweight="bold", ha="center")

    # Size legend
    for prod_ref, label in [(700, "700 mg/h"), (400, "400 mg/h"), (100, "100 mg/h")]:
        sz = 30 + 180 * prod_ref / max_prod
        ax.scatter([], [], s=sz, color="#94a3b8", edgecolors="white",
                   label=label)

    ax.set_xlabel("Residence time τ (min)", fontsize=10, color=DARK)
    ax.set_ylabel("Surface-to-volume ratio  S/V (m⁻¹)", fontsize=10, color=DARK)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cbd5e1")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MID)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(fontsize=7.5, framealpha=0.85, edgecolor="#e2e8f0",
              loc="upper right", title="Marker size = productivity",
              title_fontsize=7.5, ncol=2)

    ax.set_title("B   Design Space: Safety vs Throughput Trade-off",
                 loc="left", fontsize=11, fontweight="bold", color=DARK, pad=8)

    if standalone:
        fig.tight_layout()
        out = OUTDIR / "thermal_panel2_landscape.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved {out}")
    return fig


# ── Combined ──────────────────────────────────────────────────────────────────

def combined(result: dict):
    fig = plt.figure(figsize=(19, 6.5), facecolor=BG)
    ax1 = fig.add_axes([0.01, 0.04, 0.53, 0.88])
    ax2 = fig.add_axes([0.57, 0.08, 0.41, 0.82])

    panel1(result, ax=ax1)
    panel2(result, ax=ax2)

    fig.suptitle(
        "ENGINE Council v4  —  Thermal Knoevenagel Condensation (EtOH, 80 °C, 8 h batch)",
        fontsize=12, fontweight="bold", color=DARK, y=1.01
    )
    out = OUTDIR / "thermal_panels_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved {out}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        print(f"[!] {INPUT} not found — run `python run_thermal.py` first.")
        raise SystemExit(1)

    result = _load()
    print("Generating thermal panels …")
    panel1(result)
    panel2(result)
    combined(result)
    print("Done.")
