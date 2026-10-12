"""Replot the audited rule exports without the low-information concept network."""
from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textwrap import fill

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/october1_postmeeting_work/figures'
DATA = ROOT / 'visualization/panel_data_exports'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT.parent / 'raw'
    raw.mkdir(exist_ok=True)
    files = ['fig2a_rule_landscape.csv', 'fig2b_formula_coverage.csv', 'fig2c_coverage_heatmap_matrix.csv']
    for name in files:
        shutil.copy2(DATA / name, raw / name)
    landscape, formula, heat = [pd.read_csv(DATA / n) for n in files]
    totals = landscape.groupby(['category_key', 'category_label'], as_index=False)['count'].sum().nlargest(14, 'count')
    labels = totals.category_label.tolist()[::-1]
    pivot = landscape.pivot(index='category_label', columns='severity', values='count').fillna(0).reindex(labels)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelsize': 14,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(10.4, 10.7), facecolor='white')
    a = fig.add_axes([.215, .63, .32, .26])
    b = fig.add_axes([.635, .63, .31, .26])
    c = fig.add_axes([.215, .19, .655, .30])
    cb = fig.add_axes([.891, .21, .015, .26])
    fig.text(.035, .965, 'a', weight='bold', size=19)
    fig.text(.09, .965, 'Rule categories and severity', weight='bold', size=15)
    fig.text(.58, .965, 'b', weight='bold', size=19)
    fig.text(.635, .965, 'Expression coverage', weight='bold', size=15)
    fig.text(.035, .535, 'c', weight='bold', size=19)
    fig.text(.09, .535, 'Rule associations across chemistry classes', weight='bold', size=15)
    colors = {'hard_rule': '#b95449', 'guideline': '#397b9b', 'tip': '#629b87', 'safety': '#b78932'}
    left = np.zeros(len(labels))
    for severity, color in colors.items():
        if severity in pivot:
            vals = pivot[severity].to_numpy()
            a.barh(np.arange(len(labels)), vals, left=left, label=severity.replace('_', ' '), color=color)
            left += vals
    a.set_yticks(range(len(labels)), [fill(x, 20) for x in labels], fontsize=12)
    a.set_xlabel('Number of rules')
    fig.legend(*a.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.18, .943), ncol=2,
               frameon=False, fontsize=11.5, columnspacing=1.0, handlelength=1.2)
    # Identical row order eliminates a second set of long category labels.
    values = formula.set_index('category_label').reindex(labels).percent_with_formula
    b.barh(range(len(labels)), values, color='#468e88')
    b.set_yticks(range(len(labels)), ['']*len(labels))
    b.set_xlim(0, 109)
    b.set_xticks([0, 50, 100])
    b.set_xlabel('Rules with expressions (%)')
    for i, v in enumerate(values):
        b.text(v-2, i, f'{v:.0f}', ha='right', va='center', color='white', fontsize=11)
    fig.text(.635, .90, 'Same category order as (a)', fontsize=11, color='#4d555b')
    for ax in (a, b):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e5e9eb', linewidth=.6)
    h = heat.nlargest(13, 'row_total').set_index('category_label')
    vals = h.drop(columns=['category_key', 'row_total'])
    im = c.imshow(np.log1p(vals), aspect='auto', cmap='YlGnBu', interpolation='nearest')
    c.set_yticks(range(len(vals)), [fill(x, 20) for x in vals.index], fontsize=11.5)
    c.set_xticks(range(len(vals.columns)), [fill(x, 21) for x in vals.columns], rotation=55, ha='right', rotation_mode='anchor', fontsize=11.5)
    c.tick_params(length=0)
    for spine in c.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(im, cax=cb)
    colorbar.set_label('log(1 + association count)', fontsize=12, labelpad=7)
    colorbar.ax.tick_params(labelsize=10)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'Figure_S4_rules_three_panels.{ext}', dpi=350)
    plt.close(fig)
    (raw / 'rule_figure_basis.json').write_text(json.dumps({'rule_total': int(landscape['count'].sum()),
        'panel_a': '14 largest categories; stored severity labels',
        'panel_b': 'Same categories/order; detected-expression proportion, not equation verification',
        'panel_c': '13 largest association totals; non-exclusive chemistry labels; log1p color scale',
        'removed': 'Concept co-occurrence network'}, indent=2))


if __name__ == '__main__':
    main()
