"""Large-type layouts for the frozen S7, S9 and S10 benchmark exports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


INK = "#17212B"
MUTED = "#5B6775"
GRID = "#D9DEE5"
RED = "#C44E52"
TEAL = "#16847A"
MODELS = ["Qwen3.6-27B", "Qwen3.8-27B", "GPT-4o", "Claude Sonnet 4.6", "Claude Opus 4.6"]
CASES = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
MODEL_LABELS = ["Qwen 3.6\n27B", "Qwen 3.8\n27B", "GPT-4o", "Sonnet 4.6", "Opus 4.6"]
MODEL_COLORS = ["#4878A8", "#5A9F7A", "#DC9236", "#946BB0", "#C95F78"]


def style(size=12):
    return mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": size, "axes.labelsize": size,
        "xtick.labelsize": size, "ytick.labelsize": size, "legend.fontsize": size,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.facecolor": "white", "figure.facecolor": "white",
    })


def clean(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_axisbelow(True)


def heading(fig, letter, title, x, y, size=16):
    fig.text(x, y, letter, fontsize=size + 2, fontweight="bold", va="top")
    fig.text(x + 0.044, y - 0.001, title, fontsize=size, fontweight="bold", va="top")


def save_checked(fig, number: int, out: Path, data_checks: dict):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in texts if not fig.bbox.contains(b.x0, b.y0)
               or not fig.bbox.contains(b.x1, b.y1)]
    overlaps = [[t.get_text(), t2.get_text()] for i, (t, b) in enumerate(texts)
                for t2, b2 in texts[i + 1:] if b.overlaps(b2)]
    min_font = min(t.get_fontsize() for t, _ in texts)
    report = {"figure": f"S{number}", "minimum_font_pt": min_font,
              "canvas_inches": fig.get_size_inches().tolist(),
              "minimum_font_pt_at_180mm_width": min_font * (180 / 25.4) / fig.get_figwidth(),
              "outside_canvas": outside, "text_overlap_pairs": overlaps, **data_checks}
    (out / "documentation" / f"Figure_S{number}_readability_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not outside, outside
    assert not overlaps, overlaps
    assert report["minimum_font_pt_at_180mm_width"] >= 8
    stem = out / "figures" / f"Figure_S{number:02d}"
    # Fixed page bounds preserve the intended physical font sizes in PDF/SVG.
    for extension in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=450, facecolor="white")
    preview_dir = out / "documentation" / "print_size_checks"
    preview_dir.mkdir(exist_ok=True)
    fig.savefig(preview_dir / f"Figure_S{number:02d}_180mm_96dpi.png",
                dpi=(180 / 25.4) * 96 / fig.get_figwidth(), facecolor="white")
    plt.close(fig)


def copy_raw(bench, out, source, target):
    source = bench / "figures_revised" / "main" / "raw" / source
    shutil.copy2(source, out / "source_data" / target)
    return pd.read_csv(source)


def figure_s7(bench: Path, out: Path):
    data = copy_raw(bench, out, "figs5-1.csv", "S7_case_scores.csv")
    assert len(data) == 30 and data.n_repeats.eq(3).all()
    assert data.sample_sd.notna().all() and data.sample_sd.ge(0).all()
    with style(13):
        fig = plt.figure(figsize=(7.8, 11.8))
        fig.legend(handles=[Patch(color=RED, label="One-shot"), Patch(color=TEAL, label="FlowPilot")],
                   loc="upper center", bbox_to_anchor=(0.63, 0.998), frameon=False, ncol=2)
        for letter, case, top in zip("abc", CASES, [0.945, 0.639, 0.333]):
            heading(fig, letter, case, 0.035, top, size=15)
            ax = fig.add_axes([0.35, top - 0.249, 0.49, 0.202])
            subset = data[data.case == case].sort_values("mean")
            labels = []
            for y, row in enumerate(subset.itertuples()):
                color = TEAL if row.architecture == "FlowPilot" else RED
                ax.errorbar(row.mean, y, xerr=row.sample_sd, fmt="o", color=color,
                            markersize=6, elinewidth=1.5, capsize=3, markeredgecolor="white", zorder=3)
                name = row.model.replace("Claude ", "").replace("Qwen3", "Qwen 3").replace("-27B", "")
                labels.append(f"{name} | {'FP' if row.architecture == 'FlowPilot' else 'OS'}")
                ax.text(1.10, y, f"{row.mean:.2f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=13)
            ax.set_yticks(np.arange(len(subset)), labels)
            for tick, architecture in zip(ax.get_yticklabels(), subset.architecture):
                tick.set_color(TEAL if architecture == "FlowPilot" else RED)
            ax.set_xlim(0.48, 1.01)
            ax.set_ylim(-0.7, len(subset) - 0.3)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_xlabel("Benchmark score", labelpad=4)
            ax.text(1.10, 1.055, "Mean", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color=MUTED)
            clean(ax)
        fig.text(0.035, 0.015, "Mean +/- sample SD; 3 repeats. FP: FlowPilot; OS: one-shot. Both Qwen models: 27B.", fontsize=10.5)
        save_checked(fig, 7, out, {"score_rows": len(data), "error_bars": len(data),
                                  "zero_sd_rows": int(data.sample_sd.eq(0).sum()),
                                  "minimum_sample_sd": float(data.sample_sd.min()),
                                  "data_unchanged": True})


def figure_s9(bench: Path, out: Path):
    data = copy_raw(bench, out, "figs5-3_error_map.csv", "S9_error_map.csv")
    details = copy_raw(bench, out, "figs5-3_error_details.csv", "S9_error_details.csv")
    criteria = data[["criterion_id", "criterion_label"]].drop_duplicates().sort_values("criterion_id")
    columns = pd.MultiIndex.from_product([MODELS, ["One-shot", "FlowPilot"], ["repeat_01", "repeat_02", "repeat_03"]],
                                        names=["model", "architecture", "repeat_id"])
    assert not data.duplicated(["case", "criterion_id", "model", "architecture", "repeat_id"]).any()
    assert data.candidate_id.nunique() == 90 and len(data) == 990
    assert data.cell_value.isin([-1, 0, 1, 2, 3]).all()
    assert np.array_equal(data.cell_value, np.where(data.applicability.eq("NOT_APPLICABLE"), -1, data.judge_flags))
    no_flags = {architecture: int(group.groupby("candidate_id").judge_flags.max().eq(0).sum())
                for architecture, group in data.groupby("architecture")}
    flagged = data[data.cell_value.gt(0)]
    assert set(zip(flagged.candidate_id, flagged.criterion_id)) == set(zip(details.candidate_id, details.criterion_id))
    palette = ["#D5D9DF", "#F1F6F3", "#F4BDB8", "#DC766E", "#A72E29"]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], 5)
    counts = {"rendered_cells": 0, "flagged_cells": 0, "not_applicable_cells": 0}
    with style(13):
        fig = plt.figure(figsize=(10, 12.8))
        for letter, case, top in zip("abc", CASES, [0.985, 0.685, 0.385]):
            heading(fig, letter, case, 0.03, top, size=17)
            matrix = data[data.case == case].pivot(index="criterion_id", columns=["model", "architecture", "repeat_id"], values="cell_value")
            matrix = matrix.reindex(index=criteria.criterion_id, columns=columns)
            assert matrix.shape == (11, 30) and not matrix.isna().any().any()
            values = matrix.to_numpy(dtype=int)
            counts["rendered_cells"] += values.size
            counts["flagged_cells"] += int((values > 0).sum())
            counts["not_applicable_cells"] += int((values == -1).sum())
            ax = fig.add_axes([0.24, top - 0.260, 0.735, 0.170])
            ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            for r, c in np.argwhere(values > 0):
                value = values[r, c]
                ax.text(c, r, str(value), ha="center", va="center", fontsize=13.5,
                        color="white" if value > 1 else "#742C27", fontweight="bold")
            for r, c in np.argwhere(values == -1):
                ax.text(c, r, "-", ha="center", va="center", fontsize=13.5, color=MUTED)
            ax.set_yticks(np.arange(11), criteria.criterion_label)
            ax.set_xticks(np.arange(30), ["1", "2", "3"] * 10)
            ax.tick_params(axis="both", length=0, pad=6, labelsize=13.5)
            ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 11, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.4)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for m, model in enumerate(MODEL_LABELS):
                center = (6 * m + 3) / 30
                ax.text(center, 1.32, model.replace("\n27B", ""), transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, fontweight="bold")
                for a, architecture in enumerate(["OS", "FP"]):
                    ax.text((6 * m + 3 * a + 1.5) / 30, 1.10, architecture,
                            transform=ax.transAxes, ha="center", va="center", fontsize=12.5,
                            color=RED if architecture == "OS" else TEAL, fontweight="bold")
                if m:
                    ax.axvline(6 * m - 0.5, color=MUTED, linewidth=1.4)
                ax.axvline(6 * m + 2.5, color="#ADB7BF", linewidth=0.8)
            ax.text(-0.025, -0.09, "Repeat", transform=ax.transAxes, ha="right", va="top", fontsize=12.5)
        legend = [Patch(facecolor=c, edgecolor=GRID, label=l) for c, l in zip(palette, ["Not applicable", "No flag", "1 judge", "2 judges", "3 judges"])]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.060), frameon=False, ncol=5,
                   fontsize=12, columnspacing=1.1, handlelength=1.1, handletextpad=0.4)
        fig.text(0.03, 0.043, "OS: one-shot; FP: FlowPilot. Cells show judges flagging a critical issue, not error counts.", fontsize=12)
        fig.text(0.03, 0.022, f"No critical flags: one-shot {no_flags['One-shot']}/45 campaigns; FlowPilot {no_flags['FlowPilot']}/45. Both Qwen models: 27B.", fontsize=12)
        assert counts == {"rendered_cells": 990, "flagged_cells": 84, "not_applicable_cells": 30}
        save_checked(fig, 9, out, {**counts, "campaigns": 90, "campaigns_without_critical_flags": no_flags, "data_unchanged": True,
                                  "axes_transposed_only": True})


def figure_s10(bench: Path, out: Path):
    campaigns = copy_raw(bench, out, "figs5-4_campaigns.csv", "S10_campaigns.csv")
    summary = copy_raw(bench, out, "figs5-4_resource_summary.csv", "S10_resource_summary.csv").set_index("model").loc[MODELS]
    assert len(campaigns) == 45 and campaigns.architecture.eq("FlowPilot").all()
    assert campaigns.groupby(["model", "case"]).size().eq(3).all()
    assert np.allclose(campaigns.input_tokens + campaigns.output_tokens, campaigns.total_tokens)
    assert np.allclose((campaigns.input_tokens * campaigns.input_price_per_million_usd +
                        campaigns.output_tokens * campaigns.output_price_per_million_usd) / 1e6,
                       campaigns.generation_cost_usd)
    assert np.allclose(campaigns.mean_score_0_1 / campaigns.generation_cost_usd,
                       campaigns.campaign_quality_per_usd)
    resource_fields = [("total_tokens", "mean_total_tokens", "total_tokens_sd"),
                       ("generation_cost_usd", "mean_generation_cost_usd", "generation_cost_sd_usd"),
                       ("runtime_min", "mean_runtime_min", "runtime_sd_min")]
    for field, mean, sd in resource_fields:
        grouped = campaigns.groupby("model")[field].agg(["mean", "std"]).loc[MODELS]
        assert np.allclose(grouped["mean"], summary[mean])
        assert np.allclose(grouped["std"], summary[sd])
    quality = campaigns.groupby(["case", "model"]).campaign_quality_per_usd.agg(["mean", "std", "count"])
    quality.to_csv(out / "source_data" / "S10_quality_per_cost_summary.csv")
    with style(14):
        fig = plt.figure(figsize=(10.5, 12.5))
        specifications = [
            ("a", "Token use", "Total tokens (thousands)", summary.mean_total_tokens / 1000, summary.total_tokens_sd / 1000, False, "tokens"),
            ("b", "Generation cost", "Generation cost (USD)", summary.mean_generation_cost_usd, summary.generation_cost_sd_usd, False, "cost"),
            ("c", "Observed runtime", "Wall-clock time (min)", summary.mean_runtime_min, summary.runtime_sd_min, False, "time"),
        ]
        for letter, case in zip("def", CASES):
            q = quality.loc[case].loc[MODELS]
            title = "Photochemical\noxidation" if case == "Photochemical oxidation" else case
            specifications.append((letter, title, "Benchmark score / USD", q["mean"], q["std"], True, "quality"))
        for index, (letter, title, xlabel, mean, sd, logarithmic, kind) in enumerate(specifications):
            row, col = divmod(index, 2)
            top = 0.975 - 0.303 * row
            left = 0.175 + 0.50 * col
            heading(fig, letter, title, 0.025 + 0.50 * col, top, size=16)
            ax = fig.add_axes([left, top - 0.247, 0.23, 0.172])
            means, sds = np.asarray(mean), np.asarray(sd)
            for i, (value, err, color) in enumerate(zip(means, sds, MODEL_COLORS)):
                if logarithmic:
                    assert value - err > 0
                    ax.errorbar(value, i, xerr=err, fmt="o", markersize=7, color=color,
                                elinewidth=1.6, capsize=3, zorder=3)
                else:
                    ax.barh(i, value, height=0.54, color=color, zorder=2)
                    ax.errorbar(value, i, xerr=err, fmt="none", ecolor=MUTED, elinewidth=1.4, capsize=3, zorder=3)
                value_label = f"${value:.3f}" if kind == "cost" else f"{value:.1f}k" if kind == "tokens" else f"{value:.1f}"
                ax.text(1.33, i, value_label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=13)
            ax.text(1.33, 1.045, "Mean", transform=ax.transAxes, ha="right", va="bottom", fontsize=13, color=MUTED)
            ax.set_yticks(np.arange(5), MODEL_LABELS)
            ax.set_ylim(4.6, -0.6)
            ax.set_xlabel(xlabel, fontsize=13, labelpad=7)
            if logarithmic:
                ax.set_xscale("log")
                ax.set_xlim(0.75, 80)
                ax.set_xticks([1, 10, 50], ["1", "10", "50"])
                ax.minorticks_off()
            else:
                ax.set_xlim(0, float(max(means + sds)) * 1.12)
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, min_n_ticks=3))
            ax.tick_params(axis="both", labelsize=13)
            clean(ax)
        fig.text(0.025, 0.043, "a-c: mean +/- sample SD; 9 campaigns per model (3 cases x 3 repeats).", fontsize=12.5)
        fig.text(0.025, 0.022, "d-f: mean +/- sample SD; 3 repeats per case; log scale. Runtime depends on the environment.", fontsize=12.5)
        save_checked(fig, 10, out, {"campaigns": 45, "resource_aggregates_recomputed": True,
                                   "costs_verified_against_stored_prices": True,
                                   "quality_mean_of_campaign_ratios": True, "error_bars": 30,
                                   "data_unchanged": True})
