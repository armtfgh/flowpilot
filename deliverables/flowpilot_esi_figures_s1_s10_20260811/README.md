# FlowPilot ESI Figures S1-S10

Generated from the current repository and frozen manuscript data exports.

## Contents

- `FlowPilot_ESI_Figures_S1-S10.docx`: editable ten-page review document with captions
- `FlowPilot_ESI_Figures_S1-S10.pdf`: ten-page A4 review PDF with captions
- `figures/Figure_S01.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S02.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S03.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S04.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S05.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S06.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S07.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S08.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S09.png` (400 dpi), `.pdf`, and `.svg`
- `figures/Figure_S10.png` (400 dpi), `.pdf`, and `.svg`

## Source data

Each figure has one or more CSV/JSON files in `source_data/`. Files copied from the manuscript visualization cache retain their original names; new files begin with the corresponding `S#` prefix.

## Interpretation boundaries

- S7 uses the frozen classified corpus (`n=464`), not every raw JSON file currently present in the records directory.
- S8 quantitative-expression coverage is machine-detected expression coverage, not independent equation validation.
- S10 evaluates retrieval metadata alignment and rank behavior, not chemical yield or design optimality.
- S6 is one representative executable case and does not imply all runs contain the same artifact/event counts.

## Reproduction

From the project root, run:

```bash
/home/amirreza/anaconda3/envs/flent/bin/python scripts/build_esi_figures_s1_s10.py
/home/amirreza/anaconda3/envs/flent/bin/python scripts/build_esi_s1_s10_review.py
```

The first script rebuilds standalone figures and source exports. The second assembles the editable review document; LibreOffice was used to create the supplied PDF.
