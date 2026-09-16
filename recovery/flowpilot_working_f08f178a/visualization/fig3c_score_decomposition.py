"""
Figure 3C — Retrieval score decomposition: semantic-only vs FlowPilot reranking.

Uses stored ChromaDB embeddings directly — zero new API calls.

For each record, its own stored embedding is used as the query vector,
then we compare:
  - Semantic-only ranking     (sort by L2 distance alone)
  - FlowPilot final ranking   (0.6 × semantic + 0.4 × field similarity)

Three sub-panels:
  A) Scatter: semantic score vs final score, colored by field score
     → shows where reranking moves papers
  B) Rank-change distribution: how many positions each paper shifts
  C) Field score breakdown: how much each component (photocatalyst /
     solvent / wavelength) contributes on average

No API calls — uses stored embeddings.

Jupyter usage:
    from fig3c_score_decomposition import make_figure
    fig = make_figure()
    fig.savefig("fig3c.png", dpi=300, bbox_inches="tight")
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from collections import defaultdict

BG         = "#FFFFFF"
TEXT_LIGHT = "#111827"
TEXT_DIM   = "#4B5563"
PANEL_EDGE = "#D0D7DE"
GRID       = "#E5E7EB"

# Weights (must match flora_translate/config.py)
W_SEMANTIC      = 0.6
W_FIELD         = 0.4
W_PHOTOCATALYST = 0.30
W_SOLVENT       = 0.20
W_WAVELENGTH    = 0.20
W_TEMPERATURE   = 0.15
W_CONCENTRATION = 0.15

PHOTOCATALYST_FAMILIES = {
    "iridium": ["ir(", "ir-", "fac-ir", "[ir"],
    "ruthenium": ["ru(", "ru-", "[ru"],
    "organic": ["eosin", "rose bengal", "methylene blue", "riboflavin",
                "acridinium", "4czipn", "rhodamine"],
    "titanium": ["tio2", "titanium dioxide"],
}


def _pc_family(name: str) -> str | None:
    if not name:
        return None
    nl = name.lower()
    for fam, patterns in PHOTOCATALYST_FAMILIES.items():
        if any(p in nl for p in patterns):
            return fam
    return "other_pc"


def _photocatalyst_score(a: str, b: str) -> float:
    fa, fb = _pc_family(a), _pc_family(b)
    if fa and fb:
        return 1.0 if fa == fb else 0.3
    return 0.0


def _solvent_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 if a.lower().strip() == b.lower().strip() else 0.0


def _wavelength_score(a, b) -> float:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0
    diff = abs(a - b)
    if diff <= 30:
        return 1.0
    if diff >= 100:
        return 0.0
    return 1.0 - (diff - 30) / 70


def _field_score(q_meta: dict, r_meta: dict) -> tuple[float, dict]:
    """Returns (total_field_score, component_dict)."""
    pc  = W_PHOTOCATALYST * _photocatalyst_score(
              q_meta.get("photocatalyst"), r_meta.get("photocatalyst"))
    sol = W_SOLVENT * _solvent_score(
              q_meta.get("solvent"), r_meta.get("solvent"))
    wl  = W_WAVELENGTH * _wavelength_score(
              q_meta.get("wavelength_nm"), r_meta.get("wavelength_nm"))
    total = pc + sol + wl
    return total, {"photocatalyst": pc, "solvent": sol, "wavelength": wl}


def _l2_to_semantic(dist: float) -> float:
    return 1.0 / (1.0 + dist)


def load_chroma_data():
    """Load all embeddings + metadata from ChromaDB."""
    from flora_translate.vector_store import VectorStore
    store  = VectorStore()
    result = store.collection.get(
        include=["embeddings", "metadatas", "documents"]
    )
    return (np.array(result["embeddings"]),
            result["metadatas"],
            result["ids"],
            result["documents"])


def run_comparison(
    n_queries: int = 80,
    top_k: int = 20,
    seed: int = 42,
):
    """
    For n_queries random records, compare semantic-only vs FlowPilot ranking.
    Returns list of result dicts.
    """
    embs, metas, ids, docs = load_chroma_data()
    N = len(ids)
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(N, size=min(n_queries, N), replace=False)

    results = []
    for qi in query_idx:
        q_emb  = embs[qi]
        q_meta = metas[qi]

        # L2 distances to all others (exclude self)
        diffs  = embs - q_emb
        l2     = np.sqrt((diffs ** 2).sum(axis=1))
        l2[qi] = np.inf  # exclude self

        # Top-K by semantic only
        sem_top_idx = np.argsort(l2)[:top_k]

        for rank_sem, ri in enumerate(sem_top_idx):
            sem_score   = _l2_to_semantic(l2[ri])
            fs, fs_comp = _field_score(q_meta, metas[ri])
            final_score = W_SEMANTIC * sem_score + W_FIELD * fs

            results.append({
                "query_id":    ids[qi],
                "result_id":   ids[ri],
                "sem_score":   sem_score,
                "field_score": fs,
                "final_score": final_score,
                "rank_sem":    rank_sem + 1,
                "fs_pc":       fs_comp["photocatalyst"],
                "fs_sol":      fs_comp["solvent"],
                "fs_wl":       fs_comp["wavelength"],
                "q_has_pc":    bool(q_meta.get("photocatalyst", "").strip()),
                "q_has_wl":    bool(q_meta.get("wavelength_nm") and
                                    q_meta["wavelength_nm"] > 0),
                "q_has_sol":   bool(q_meta.get("solvent", "").strip()),
            })

        # Rerank by final score and assign FlowPilot rank
        query_results = [r for r in results if r["query_id"] == ids[qi]]
        query_results.sort(key=lambda x: x["final_score"], reverse=True)
        for rank_flora, r in enumerate(query_results):
            r["rank_flora"] = rank_flora + 1
            r["rank_delta"] = r["rank_sem"] - r["rank_flora"]  # positive = promoted

    return results


def make_figure(
    n_queries: int = 80,
    figsize: tuple = (13, 5),
) -> plt.Figure:

    print("Loading ChromaDB embeddings...")
    data = run_comparison(n_queries=n_queries)
    print(f"  {len(data)} retrieval pairs computed.")

    sem    = np.array([d["sem_score"]   for d in data])
    field  = np.array([d["field_score"] for d in data])
    final  = np.array([d["final_score"] for d in data])
    deltas = np.array([d["rank_delta"]  for d in data])

    # Field component averages (for records where query has that field)
    pc_vals  = [d["fs_pc"]  for d in data if d["q_has_pc"]]
    sol_vals = [d["fs_sol"] for d in data if d["q_has_sol"]]
    wl_vals  = [d["fs_wl"]  for d in data if d["q_has_wl"]]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"wspace": 0.38})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(PANEL_EDGE)

    # ── Panel 1: Scatter semantic vs final, colored by field score ────────────
    ax = axes[0]
    cmap  = plt.get_cmap("plasma")
    sc    = ax.scatter(sem, final, c=field, cmap=cmap, s=6, alpha=0.45,
                       vmin=0, vmax=field.max(), zorder=3)

    # Diagonal = semantic-only line
    lim = [min(sem.min(), final.min()) - 0.01, max(sem.max(), final.max()) + 0.01]
    ax.plot(lim, lim, "--", color="#444C56", lw=1.0, label="No reranking", zorder=2)

    # Annotate regions
    ax.text(0.97, 0.10, "Promoted\n(field boost)",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=7.5, color="#3FB950", style="italic")
    ax.text(0.05, 0.90, "Demoted\n(field penalty)",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=7.5, color="#F78166", style="italic")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Field Score", fontsize=8, color=TEXT_DIM)
    cbar.ax.tick_params(labelsize=7, colors=TEXT_DIM)
    cbar.outline.set_edgecolor(PANEL_EDGE)

    ax.set_xlabel("Semantic Score (Naive)", fontsize=9, color=TEXT_DIM, labelpad=6)
    ax.set_ylabel("Final Score (FlowPilot)", fontsize=9, color=TEXT_DIM, labelpad=6)
    ax.set_title("Score Decomposition", fontsize=10, fontweight="bold",
                 color=TEXT_LIGHT, pad=10)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)

    # ── Panel 2: Rank-change histogram ────────────────────────────────────────
    ax = axes[1]
    bins  = np.arange(-19.5, 20.5, 1)
    promo = deltas[deltas > 0]
    demo  = deltas[deltas < 0]
    same  = deltas[deltas == 0]

    ax.hist(deltas[deltas > 0], bins=bins[10:], color="#3FB950",
            alpha=0.85, label=f"Promoted ({len(promo)})", zorder=3)
    ax.hist(deltas[deltas < 0], bins=bins[:10], color="#F78166",
            alpha=0.85, label=f"Demoted  ({len(demo)})", zorder=3)
    ax.hist(deltas[deltas == 0], bins=[-0.5, 0.5], color="#58A6FF",
            alpha=0.85, label=f"No change ({len(same)})", zorder=3)

    ax.axvline(0, color="#444C56", lw=1.2, ls="--")
    ax.set_xlabel("Rank Change (+ = promoted)", fontsize=9,
                  color=TEXT_DIM, labelpad=6)
    ax.set_ylabel("Count", fontsize=9, color=TEXT_DIM, labelpad=6)
    ax.set_title("Rank Shifts from\nField Reranking", fontsize=10,
                 fontweight="bold", color=TEXT_LIGHT, pad=10)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    leg = ax.legend(fontsize=7.5, framealpha=0.95, facecolor="#FFFFFF",
                    edgecolor=PANEL_EDGE, labelcolor=TEXT_LIGHT)

    # Summary stats
    pct_changed = 100 * (1 - len(same) / len(deltas))
    ax.text(0.97, 0.97,
            f"{pct_changed:.0f}% of results\nreranked",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="#E3B341", fontweight="bold")

    # ── Panel 3: Field score component breakdown ───────────────────────────────
    ax = axes[2]

    components = {
        "Photocatalyst\nClass Match": pc_vals,
        "Solvent\nMatch":             sol_vals,
        "Wavelength\nProximity":      wl_vals,
    }
    comp_colors = ["#F78166", "#D2A8FF", "#79C0FF"]
    comp_labels, comp_means, comp_nonzero = [], [], []

    for label, vals in components.items():
        if vals:
            arr = np.array(vals)
            comp_labels.append(label)
            comp_means.append(arr.mean())
            comp_nonzero.append(100 * (arr > 0).mean())

    x_pos = np.arange(len(comp_labels))
    bw = 0.38

    # Mean score bars
    ax2r = ax.twinx()
    ax2r.set_facecolor(BG)
    ax2r.tick_params(colors=TEXT_DIM, labelsize=8)
    for spine in ax2r.spines.values():
        spine.set_color(PANEL_EDGE)

    bars = ax.bar(x_pos - bw / 2, comp_means, bw,
                  color=comp_colors, alpha=0.85,
                  edgecolor=PANEL_EDGE, linewidth=0.6,
                  label="Mean score", zorder=3)
    bars2 = ax2r.bar(x_pos + bw / 2, comp_nonzero, bw,
                     color=comp_colors, alpha=0.4,
                     edgecolor=PANEL_EDGE, linewidth=0.6,
                     label="% non-zero", zorder=3, hatch="///")

    for bar, val in zip(bars, comp_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, color=TEXT_LIGHT, fontweight="bold")
    for bar, val in zip(bars2, comp_nonzero):
        ax2r.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.5,
                  f"{val:.0f}%", ha="center", va="bottom",
                  fontsize=8, color=TEXT_DIM)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp_labels, fontsize=8.5, color=TEXT_DIM)
    ax.set_ylabel("Mean Contribution to Field Score", fontsize=8.5,
                  color=TEXT_DIM, labelpad=6)
    ax2r.set_ylabel("% Queries with Match", fontsize=8.5,
                    color=TEXT_DIM, labelpad=6)
    ax.set_title("Field Score Components", fontsize=10, fontweight="bold",
                 color=TEXT_LIGHT, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_ylim(0, max(comp_means) * 1.45 if comp_means else 0.1)
    ax2r.set_ylim(0, 115)

    legend_els = [
        mpatches.Patch(facecolor="#888", label="Mean score (left axis)"),
        mpatches.Patch(facecolor="#888", alpha=0.4, hatch="///",
                       label="% queries matched (right axis)"),
    ]
    ax.legend(handles=legend_els, fontsize=7, loc="upper right",
              framealpha=0.95, facecolor="#FFFFFF",
              edgecolor=PANEL_EDGE, labelcolor=TEXT_LIGHT)

    fig.suptitle(
        "FlowPilot Retrieval: Semantic + Field Reranking vs Semantic-Only",
        fontsize=12, fontweight="bold", color=TEXT_LIGHT, y=1.02,
    )
    fig.tight_layout(pad=1.5)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
