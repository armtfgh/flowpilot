"""Build case-figure prototypes from archived exports; never call the pipeline.

Run: python3 -B scripts/build_case_figures_20260922.py
Requires matplotlib, Pillow and PyMuPDF. Chemical schemes use explicitly
abbreviated vector formulas, not inferred or decorative molecular structures.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flowpilot_case_figures_mpl")

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/revision_20260922/figures"
SOURCE = {
    5: ROOT / "outputs/figure5_pure_oxygen_physics_20260918/slides",
    6: ROOT / "outputs/khu_revised_six_20260915/collaborator_slides_20260915_163845",
}
WIDTH, HEIGHT = 6.5, 6.8
INK, MUTED, RULE = "#202b31", "#52616a", "#cbd4d8"
ACCENT = {5: "#176b72", 6: "#943c62"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.hashsalt": "flowpilot-case-figures-20260922",
    "savefig.facecolor": "white", "mathtext.default": "regular",
})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str) -> float:
    return float(row[key])


def near(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=0.0001, rel_tol=0):
        raise ValueError(f"{label}: {actual} != {expected}")


def load_cases() -> dict:
    cases = {}
    for figure, source in SOURCE.items():
        cases[figure] = []
        for index in (1, 2, 3):
            folder = source / f"figure{figure}_set{index}"
            stages = sorted(rows(folder / "stage_parameters.csv"),
                            key=lambda r: number(r, "stage"))
            feeds = rows(folder / "feed_parameters.csv")
            if len(stages) != 2:
                raise ValueError(f"Expected two stages: {folder}")
            for stage in stages:
                q = number(stage, "liquid_flow_mL_min") + number(stage, "gas_inlet_STP_mL_min")
                near(number(stage, "volume_mL") / q,
                     number(stage, "nominal_inlet_residence_min"), str(folder))
                near(number(stage, "BPR_bar"), 7, str(folder))
            if figure == 5:
                near(number(stages[0], "volume_mL"), 5 if index == 2 else 2, "F5 V1")
                near(number(stages[0], "nominal_inlet_residence_min"),
                     250 if index == 2 else 100, "F5 t1")
                near(number(stages[1], "volume_mL"), 20, "F5 V2")
                near(number(stages[1], "nominal_inlet_residence_min"), 181.8182, "F5 t2 index")
                near(number(stages[0], "gas_inlet_STP_mL_min"), 0, "F5 gas placement")
                for stage in stages:
                    near(number(stage, "liquid_flow_mL_min"), .02, "F5 liquid flow")
                    near(number(stage, "temperature_C"), 25, "F5 temperature")
                gas = next(r for r in feeds if r["phase"] == "gas")
                if gas["component"] != "O2" or number(gas, "stage") != 2:
                    raise ValueError("Figure 5 must use pure oxygen at Stage 2")
                near(number(gas, "flow_mL_min"), .09, "F5 oxygen STP flow")
                near(number(gas, "equivalents"), 2.0078, "F5 oxygen equivalents")
            else:
                targets = (31.25, 25) if index == 1 else (35.7142857143, 14.2857142857)
                for stage, target in zip(stages, targets):
                    near(number(stage, "nominal_inlet_residence_min"), target, "F6 stage time")
                    near(number(stage, "temperature_C"), 95, "F6 temperature")
                    near(number(stage, "gas_inlet_STP_mL_min"), 0, "F6 liquid only")
                amine = next(r for r in feeds if r["component"].casefold() == "benzylamine")
                near(number(amine, "stage"), 2, "F6 amine stage")
                near(number(amine, "concentration_M"), 2.1, "F6 amine stock")
            cases[figure].append({"folder": folder, "set": index, "stages": stages, "feeds": feeds})
    # Compare numerical conditions, not set IDs or model narratives.
    for figure, pair in ((5, (0, 2)), (6, (1, 2))):
        first, second = (cases[figure][i] for i in pair)
        fields = {
            "stages": ("stage", "volume_mL", "ID_mm", "temperature_C", "liquid_flow_mL_min",
                       "gas_inlet_STP_mL_min", "nominal_inlet_residence_min", "BPR_bar"),
            "feeds": ("stage", "concentration_M", "flow_mL_min", "molar_flow_mmol_min", "equivalents"),
        }
        for category, keys in fields.items():
            clean = lambda rs: [tuple(float(r[k]) if r[k] else None for k in keys) for r in rs]
            if clean(first[category]) != clean(second[category]):
                raise ValueError(f"Expected duplicate physical conditions in Figure {figure}")
    return cases


class Page:
    """Fixed-inch layout with machine-checked text bounds and embedded GUI pixels."""

    def __init__(self, figure: int, height=HEIGHT):
        self.figure = figure
        self.height = height
        self.fig = plt.figure(figsize=(WIDTH, height), dpi=150)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(height, 0))
        self.ax.axis("off")
        self.texts = []
        self.images = []

    def text(self, x, y, value, size=8.3, weight="normal", color=INK,
             align="left", width=None):
        artist = self.ax.text(x, y, value, fontsize=size, fontweight=weight,
                              color=color, va="top", ha=align, linespacing=1.35)
        self.texts.append((artist, width))
        return artist

    def line(self, y, x=.12, right=6.38, color=RULE, lw=.6):
        self.ax.plot([x, right], [y, y], color=color, lw=lw, clip_on=False)

    def heading(self, letter, y, title):
        self.text(.12, y, letter, 9, "bold", ACCENT[self.figure])
        self.text(.34, y, title, 9, "bold", width=6.0)

    def image(self, path, top, max_height=3.28):
        with Image.open(path) as source:
            original = source.copy()
        width = 6.30
        height = width * original.height / original.width
        if height > max_height:
            width *= max_height / height
            height = max_height
        x = (WIDTH - width) / 2
        axis = self.fig.add_axes([x / WIDTH, 1 - (top + height) / self.height,
                                 width / WIDTH, height / self.height], zorder=0)
        axis.imshow(original, interpolation="none", aspect="equal")
        axis.axis("off")
        self.images.append({"source": str(path.relative_to(ROOT)),
                            "sha256": sha(path), "native_pixels": list(original.size),
                            "placed_width_inches": width, "placed_height_inches": height,
                            "effective_source_dpi": original.width / width,
                            "cropped": False, "redrawn": False})
        return top + height

    def table(self, y, headers, values, widths, row_height=.21, size=8.0):
        x0 = .15
        total = sum(widths)
        self.ax.add_patch(Rectangle((x0, y), total, row_height,
                                   facecolor="#edf2f3", edgecolor="none", zorder=-1))
        for i, row in enumerate([headers, *values]):
            x = x0
            for item, width in zip(row, widths):
                self.text(x + .045, y + i * row_height + .038, str(item), size,
                          "bold" if i == 0 else "normal", width=width - .085)
                x += width
            if i:
                self.line(y + (i + 1) * row_height, x0, x0 + total, lw=.35)

    def save(self, stem: str, dpi: int) -> dict:
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        boxes = []
        for artist, limit in self.texts:
            box = artist.get_window_extent(renderer)
            if limit and box.width / self.fig.dpi > limit + .015:
                raise ValueError(f"Text too wide ({box.width / self.fig.dpi:.2f} > {limit}): {artist.get_text()}")
            if box.x0 < 0 or box.y0 < 0 or box.x1 > WIDTH * self.fig.dpi or box.y1 > self.height * self.fig.dpi:
                raise ValueError(f"Text outside page: {artist.get_text()}")
            boxes.append((artist.get_text(), box))
        for i, (label, box) in enumerate(boxes):
            for other, bounds in boxes[i + 1:]:
                overlap_w = min(box.x1, bounds.x1) - max(box.x0, bounds.x0)
                overlap_h = min(box.y1, bounds.y1) - max(box.y0, bounds.y0)
                if overlap_w > 1 and overlap_h > 1:
                    raise ValueError(f"Overlapping labels: {label!r} / {other!r}")
        outputs = {}
        for suffix in ("png", "pdf", "svg"):
            path = OUT / f"{stem}.{suffix}"
            self.fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
            outputs[suffix] = {"file": path.name, "sha256": sha(path)}
        # A modest preview makes visual inspection practical without changing deliverables.
        preview = OUT / "previews" / f"{stem}.png"
        self.fig.savefig(preview, dpi=160, bbox_inches=None, pad_inches=0)
        plt.close(self.fig)
        return {"outputs": outputs, "page_inches": [WIDTH, self.height], "png_dpi": dpi,
                "text_bounds_and_overlap_check": "passed", "embedded_topologies": self.images,
                "minimum_authored_font_pt": min(t.get_fontsize() for t, _ in self.texts),
                "topology_labels": "Original raster labels retained; key values repeated in vector text."}


def scheme(page: Page, y: float):
    if page.figure == 5:
        formulas = [r"$\mathrm{Ar{-}S{-}CH_2{-}SiMe_3}$",
                    r"$\mathrm{Ar{-}S{-}(CH_2)_3{-}CN}$",
                    r"$\mathrm{Ar{-}S(=O){-}(CH_2)_3{-}CN}$"]
        labels = [r"$+\ \mathrm{CH_2{=}CH{-}CN}$", "Thioether intermediate", "Sulfoxide product"]
        note = "Abbreviated formulas: Ar = 4-methoxyphenyl; Me = methyl. Stage 2 adds pure O2."
    else:
        formulas = [r"$\mathrm{Ar{-}C(=O){-}OH}$",
                    r"$\mathrm{Ar{-}C(=O){-}S{-}(2\!{-}\!Py)}$",
                    r"$\mathrm{Ar{-}C(=O){-}NH{-}CH_2Ph}$"]
        labels = ["3-Methyl-4-nitrobenzoic acid", "2-Pyridyl thioester", "N-Benzylamide"]
        note = "Abbreviations: Ar = 3-methyl-4-nitrophenyl; 2-Py = pyridin-2-yl; Ph = phenyl."
    for x, formula, label in zip((1.08, 3.19, 5.38), formulas, labels):
        page.text(x, y, formula, 9.0, align="center", width=2.06)
        page.text(x, y + .25, label, 7.8, align="center", width=2.06)
    for start, end, stage in ((1.97, 2.30, "1"), (4.12, 4.40, "2")):
        page.ax.add_patch(FancyArrowPatch((start, y + .11), (end, y + .11),
                                         arrowstyle="-|>", mutation_scale=9,
                                         linewidth=.9, color=ACCENT[page.figure]))
        page.text((start + end) / 2, y - .025, stage, 7.2, align="center")
    page.text(.15, y + .45, note, 7.5, color=MUTED, width=6.2)


def main_figure(figure: int, cases: list[dict], dpi: int) -> dict:
    page = Page(figure)
    page.text(.12, .06, "Archived design proposals; Set 1 shown. Laboratory review required; no safety validation.",
              7.7, color=MUTED, width=6.26)
    page.line(.24, color=ACCENT[figure], lw=1)
    page.heading("a", .30, "Protocol and design decisions")
    if figure == 5:
        left = "Batch: 4 h under Ar + 6 h in air, at 25 C.\nAr-sparged EtOH/pH 9 buffer (5:1); Ir, 0.5 mol%.\nSubstrate 0.10 M; acrylonitrile 0.20 M."
        right = "Oxygen-free feed to Stage 1; O2 added at Stage 2.\nLiquid 0.020; O2 0.090 mL/min (STP); 2.008 eq.\n25 C; 450/448 nm; BPR 7 bar(g)."
    else:
        left = "Batch: 30 + 30 min at 95 C, with cooling between.\nFeed A: acid 0.50 M; DPDTC 0.525 M; DMAP 0.05 M.\nBenzylamine is introduced only at Stage 2."
        right = "Set 1: feed A 0.160 + feed B 0.040 mL/min.\nFeed B: benzylamine 2.10 M (1.05 eq).\nTwo 5 mL ETFE coils; 95 C; BPR 7 bar(g)."
    page.text(.15, .50, left, 7.8, width=3.06)
    page.text(3.38, .50, right, 7.8, width=2.97)
    page.heading("b", .96, "Representative archived GUI topology | Set 1")
    page.image(cases[0]["folder"] / "topology.png", 1.15, max_height=3.27)
    page.heading("c", 4.47, "Chemical sequence")
    scheme(page, 4.70)
    page.heading("d", 5.41, "Three response sets | archived conditions")
    if figure == 5:
        headers = ["Set", "V1 (mL)", "t1 (min)", "V2 (mL)", "t2, STP index (min)"]
        values = [[c["set"], f'{number(c["stages"][0], "volume_mL"):g}',
                   f'{number(c["stages"][0], "nominal_inlet_residence_min"):g}',
                   f'{number(c["stages"][1], "volume_mL"):g}',
                   f'{number(c["stages"][1], "nominal_inlet_residence_min"):.2f}'] for c in cases]
        widths = [.50, 1.05, 1.05, 1.05, 2.55]
        foot = "t1 = V1/Qliquid; t2 = V2/(Qliquid + QO2,STP), not measured contact time."
        duplicate = "Sets 1 and 3 share conditions; they are not experimental replicates."
        placeholder = "XXX"
    else:
        headers = ["Set", "V1 / V2 (mL)", "QA / QB (mL/min)", "t1 / t2 (min)"]
        values = []
        for case in cases:
            a, b = case["stages"]
            qb = next(r for r in case["feeds"] if r["component"].casefold() == "benzylamine")
            precision = 2 if case["set"] == 1 else 4
            values.append([case["set"], f'{number(a, "volume_mL"):g} / {number(b, "volume_mL"):g}',
                           f'{number(a, "liquid_flow_mL_min"):.3f} / {number(qb, "flow_mL_min"):.3f}',
                           f'{number(a, "nominal_inlet_residence_min"):.{precision}f} / '
                           f'{number(b, "nominal_inlet_residence_min"):.{precision}f}'])
        widths = [.5, 1.3, 1.9, 2.5]
        foot = "t1 = V1/QA; t2 = V2/(QA + QB). All sets: 95 C; BPR 7 bar(g)."
        duplicate = "Sets 2 and 3 share conditions; they are not experimental replicates."
        placeholder = "YYY"
    page.table(5.64, headers, values, widths, row_height=.16, size=7.8)
    page.text(.15, 6.32, foot, 7.3, color=MUTED, width=6.2)
    page.text(.15, 6.47, duplicate, 7.3, color=MUTED, width=6.2)
    # Blank results stay visibly separate from archived computational conditions.
    page.ax.add_patch(Rectangle((.12, 6.61), 6.26, .16,
                               facecolor="#edf2f3", edgecolor="none", zorder=-1))
    page.text(.15, 6.64, f"Measured yields pending: Set 1 {placeholder}%; Set 2 {placeholder}%; Set 3 {placeholder}%.",
              7.8, "bold", ACCENT[figure], width=6.2)
    return page.save(f"figure{figure}_prototype", dpi)


def supplement(figure: int, case: dict, dpi: int) -> dict:
    page = Page(figure, height=7.2)
    number_esi = 20 if figure == 5 else 21
    letter = "abc"[case["set"] - 1]
    page.text(.12, .12, f"Figure S{number_esi}{letter}  |  Figure {figure}, response Set {case['set']}",
              11, "bold", width=6.26)
    page.text(.12, .39, "Unchanged archived GUI topology and source-CSV conditions; laboratory review required.",
              7.8, color=MUTED, width=6.26)
    page.line(.63, color=ACCENT[figure], lw=1)
    page.image(case["folder"] / "topology.png", .82, max_height=3.40)
    page.heading("a", 4.35, "Stage conditions | BPR 7 bar(g)")
    values = []
    for r in case["stages"]:
        values.append([int(number(r, "stage")), f'{number(r, "volume_mL"):g}',
                       f'{number(r, "temperature_C"):g}', f'{number(r, "liquid_flow_mL_min"):.3f}',
                       f'{number(r, "gas_inlet_STP_mL_min"):.3f}',
                       f'{number(r, "nominal_inlet_residence_min"):.4f}'])
    page.table(4.62, ["Stage", "V (mL)", "T (C)", "Qliq (mL/min)", "Qgas,STP (mL/min)", "Time (min)"],
               values, [.53, .66, .56, 1.32, 1.37, 1.76], row_height=.25, size=8)
    page.heading("b", 5.52, "Feed composition")
    aliases = {"(((4-methoxyphenyl)thio)methyl)trimethylsilane": "Silylmethyl aryl sulfide",
               "[Ir(dF(CF3)ppy)2(dtbpy)]PF6": "Ir photocatalyst",
               "3-methyl-4-nitrobenzoic acid": "3-Methyl-4-nitrobenzoic acid"}
    values = [[r["stream"], aliases.get(r["component"], r["component"]),
               f'{number(r, "concentration_M"):g}' if r["concentration_M"] else "Gas",
               f'{number(r, "flow_mL_min"):.3f}', f'{number(r, "equivalents"):.4g}']
              for r in case["feeds"]]
    page.table(5.79, ["Feed", "Component", "Stock (M)", "Q (mL/min)", "Eq"],
               values, [.45, 2.6, .95, 1.24, .96], row_height=.19, size=7.9)
    if figure == 5:
        note = "Stage 2 time is an inlet/STP index, not measured contact time; STP: 273.15 K, 1.01325 bar."
        placeholder = "XXX"
    else:
        note = "Liquid-stage times use local cumulative flow. Same selected conditions: Sets 2 and 3."
        placeholder = "YYY"
    page.text(.15, 6.83, note, 7.3, color=MUTED, width=6.2)
    page.text(.15, 7.02, f"Measured yield: {placeholder}% (pending). Response sets are not experimental replicates.",
              7.8, "bold", ACCENT[figure], width=6.2)
    return page.save(f"figureS{number_esi}{letter}_figure{figure}_set{case['set']}", dpi)


def extract_preprint() -> dict:
    path = ROOT / "manuscript/preprint.pdf"
    with fitz.open(path) as document:
        page = document[13]
        info = next((r for r in page.get_images(full=True) if r[0] == 29), None)
        if not info or info[1:4] != (51, 1935, 1359) or not page.get_image_rects(29):
            raise ValueError("Preprint page 14 no longer matches the requested xref 29 / SMask 51")
        base = document.extract_image(29)
        mask = document.extract_image(51)
        # Compose straight (unpremultiplied) RGBA from original decoded RGB + SMask.
        # A page screenshot or a premultiplied Pixmap would not preserve these pixels.
        rgb = Image.open(io.BytesIO(base["image"])).convert("RGB")
        alpha = Image.open(io.BytesIO(mask["image"])).convert("L")
        if rgb.size != (1935, 1359) or alpha.size != rgb.size:
            raise ValueError("Unexpected embedded image/mask dimensions")
        rgba = rgb.copy()
        rgba.putalpha(alpha)
        target = OUT / "figureS22_preprint_figure6_exact.png"
        rgba.save(target)
        with Image.open(target) as saved:
            if saved.mode != "RGBA" or saved.tobytes() != rgba.tobytes():
                raise ValueError("Exact extracted RGBA failed round-trip verification")
        evidence = {
            "source": str(path.relative_to(ROOT)), "source_pdf_sha256": sha(path),
            "page_one_based": 14, "xref": 29, "smask_xref": 51,
            "pixels": list(rgba.size), "mode": "RGBA", "alpha_extrema": list(alpha.getextrema()),
            "rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "alpha_pixel_sha256": hashlib.sha256(alpha.tobytes()).hexdigest(),
            "rgba_pixel_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            "pixel_hash_definition": "SHA-256 of decoded unpremultiplied row-major RGBA bytes",
            "file": target.name, "png_sha256": sha(target),
            "method": "Extract embedded JPEG xref 29 and PNG SMask xref 51; decode and attach alpha; no resize/crop/redraw.",
            "historical_scope": "Original thermal alpha-bromination Figure 6; not the new amidation case.",
            "original_claims": "Historical labels preserved verbatim, not adopted as current safety validation.",
            "pymupdf_version": fitz.VersionBind,
        }
    dump(OUT / "figureS22_extraction_manifest.json", evidence)
    return evidence


def archive_sources(cases: dict) -> list[dict]:
    records = []
    for figure, items in cases.items():
        for case in items:
            folder = case["folder"]
            target = OUT / "source_data" / folder.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ("stage_parameters.csv", "feed_parameters.csv", "topology.png", "topology.svg"):
                source = folder / name
                if not source.exists():
                    if name == "topology.svg":
                        continue
                    raise FileNotFoundError(source)
                destination = target / name
                shutil.copyfile(source, destination)
                if sha(source) != sha(destination):
                    raise ValueError(f"Source copy mismatch: {source}")
                records.append({"figure": figure, "set": case["set"],
                                "source": str(source.relative_to(ROOT)),
                                "copy": str(destination.relative_to(OUT)), "sha256": sha(source)})
        target = OUT / "source_data" / f"figure{figure}_aggregate"
        target.mkdir(exist_ok=True)
        for source in sorted(SOURCE[figure].glob("*.csv")):
            destination = target / source.name
            shutil.copyfile(source, destination)
            records.append({"figure": figure, "source": str(source.relative_to(ROOT)),
                            "copy": str(destination.relative_to(OUT)), "sha256": sha(source),
                            "scope": "Unfiltered original; Figure 6 archive also contains historical Figure 5 rows."})
    return records


def verify_exports(figures: dict, dpi: int) -> dict:
    checks = {}
    for name, record in figures.items():
        outputs = record["outputs"]
        width, height = record["page_inches"]
        with Image.open(OUT / outputs["png"]["file"]) as png:
            if png.size != (round(width * dpi), round(height * dpi)):
                raise ValueError(f"Unexpected PNG dimensions: {name}")
        source = ROOT / record["embedded_topologies"][0]["source"]
        with Image.open(source) as original:
            reference = original.convert("RGBA")
        svg = ET.parse(OUT / outputs["svg"]["file"])
        embedded = svg.find(".//{http://www.w3.org/2000/svg}image")
        if embedded is None:
            raise ValueError(f"Missing SVG topology: {name}")
        href = embedded.attrib["{http://www.w3.org/1999/xlink}href"]
        with Image.open(io.BytesIO(base64.b64decode(href.split(",", 1)[1]))) as png:
            if png.convert("RGBA").tobytes() != reference.tobytes():
                raise ValueError(f"SVG changed original topology pixels: {name}")
        with fitz.open(OUT / outputs["pdf"]["file"]) as document:
            if len(document) != 1:
                raise ValueError(f"Expected one PDF page: {name}")
            page = document[0]
            near(page.rect.width, width * 72, f"{name} PDF width")
            near(page.rect.height, height * 72, f"{name} PDF height")
            spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                     for line in block["lines"] for span in line["spans"]]
            if any(not page.rect.contains(fitz.Rect(s["bbox"])) for s in spans):
                raise ValueError(f"PDF text extends beyond the page: {name}")
            images = page.get_images()
            if len(images) != 1:
                raise ValueError(f"Expected one unchanged topology image: {name}")
            item = images[0]
            image = Image.open(io.BytesIO(document.extract_image(item[0])["image"])).convert("RGBA")
            if item[1]:
                mask = Image.open(io.BytesIO(document.extract_image(item[1])["image"])).convert("L")
                image.putalpha(mask)
            # Matplotlib stores PDF image scanlines bottom-up and applies a flip
            # in the placement matrix. Normalize storage orientation for comparison.
            direct = image.tobytes() == reference.tobytes()
            flipped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes() == reference.tobytes()
            if not (direct or flipped):
                raise ValueError(f"PDF changed original topology pixels: {name}")
            placeholder = "XXX" if "figure5" in outputs["pdf"]["file"] else "YYY"
            expected_count = 3 if name in ("figure5", "figure6") else 1
            if page.get_text().count(placeholder) != expected_count:
                raise ValueError(f"Missing measured-result placeholders: {name}")
            preview = OUT / "previews" / (Path(outputs["pdf"]["file"]).stem + "_pdf.png")
            page.get_pixmap(matrix=fitz.Matrix(160 / 72, 160 / 72)).save(preview)
            checks[name] = {"png_dimensions": "passed", "pdf_page_and_text_bounds": "passed",
                            "pdf_original_topology_pixels": "exact",
                            "pdf_image_scanlines": "bottom-up" if flipped else "top-down",
                            "svg_original_topology_pixels": "exact",
                            "wet_lab_placeholder_count": expected_count,
                            "pdf_render_preview": str(preview.relative_to(OUT))}
    dump(OUT / "export_verification.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Use at least 300 dpi for publication prototypes")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    cases = load_cases()
    sources = archive_sources(cases)
    preprint = extract_preprint()
    figures = {}
    for figure, items in cases.items():
        figures[f"figure{figure}"] = main_figure(figure, items, args.dpi)
        for case in items:
            key = f"figureS{20 if figure == 5 else 21}{'abc'[case['set'] - 1]}"
            figures[key] = supplement(figure, case, args.dpi)
    for record in sources:
        if sha(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("An input changed while building figures")
    verification = verify_exports(figures, args.dpi)
    manifest = {"builder": str(Path(__file__).relative_to(ROOT)), "builder_sha256": sha(Path(__file__)),
                "scope": "Reporting-only prototypes from archived designs. No model calls or new experiments.",
                "chemical_drawing": "Explicitly abbreviated vector formulas; no RDKit dependency or inferred structures.",
                "wet_lab_results": {"figure5": "XXX", "figure6": "YYY", "status": "measured-results pending"},
                "duplicate_conditions_not_replicates": {"figure5": [1, 3], "figure6": [2, 3]},
                "figures": figures, "sources": sources, "preprint_extraction": preprint,
                "export_verification": verification}
    dump(OUT / "build_manifest.json", manifest)
    (OUT / "README.md").write_text(
        "# Case figure prototypes, 2026-09-22\n\n"
        "Generated only from archived exports; no new model run or measured result. No DOCX was edited.\n\n"
        "- `figure5_prototype` and `figure6_prototype`: PNG (600 dpi by default), PDF and SVG; 6.5 x 6.8 inches, no duplicated figure title.\n"
        "- `figureS20a/b/c_*` and `figureS21a/b/c_*`: one full topology per 6.5 x 7.2-inch page, with source-CSV conditions.\n"
        "- `figureS22_preprint_figure6_exact.png`: original preprint page-14 image, xref 29, SMask 51, 1935 x 1359 RGBA.\n"
        "- `source_data/`: byte-identical original stage/feed CSVs and topologies; aggregates are not filtered.\n"
        "- `previews/`: 160-dpi review images. SVG text and abbreviated chemical formulas are vector; GUI topologies remain original embedded rasters.\n\n"
        "## Interpretation\n\n"
        "Figure 5 uses archived pure-O2 September 18 designs, not the earlier air design. Displayed 2.008 eq rounds the CSV's 2.0078 eq. "
        "Stage 2's 181.82 min is an inlet/STP index, not measured in-channel contact time. STP is 273.15 K and 1.01325 bar. "
        "The gas-branch check valve does not establish liquid-branch backflow protection. No safety or yield validation is claimed.\n\n"
        "Figure 6 uses archived September 15 amidation designs. Benzylamine enters only at Stage 2; local cumulative liquid flow determines each stage time. "
        "Figure 5 Sets 1/3 and Figure 6 Sets 2/3 have identical selected conditions, not experimental replicates. "
        "All new wet-lab yield entries are XXX or YYY and remain pending.\n\n"
        "S22 is the distinct historical alpha-bromination figure, extracted without alteration; its original validation language is historical, not a new claim.\n\n"
        "## Main-caption panel mapping\n\n"
        "Both main figures use the same mapping: (a) protocol and design decisions; (b) representative archived GUI topology, Set 1; "
        "(c) chemical sequence; (d) three-set condition table. The measured-results strip below the table is unnumbered. "
        "Figure 5 uses XXX and Figure 6 uses YYY for all pending wet-lab yields. The figure number and title belong only in the Word caption. "
        "At 6.5 x 6.8 inches, a nominal 9-inch text area retains 2.2 inches for caption and spacing.\n\n"
        "## Provenance and checks\n\n"
        "`build_manifest.json` records input/output hashes, dimensions, numerical checks, and text-bounds/overlap checks. "
        "`export_verification.json` checks PDF page/text bounds, yield placeholders and pixel-exact topology embedding in both PDF and SVG. "
        "`figureS22_extraction_manifest.json` records separate RGB, alpha and straight-RGBA pixel hashes. "
        "Exact GUI topology labels are necessarily small at single-page width; principal numerical conditions are repeated in readable vector tables. "
        "Original full-resolution topologies are also retained in source_data.\n\n"
        "Rebuild: `python3 -B scripts/build_case_figures_20260922.py`.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(OUT), "figure_pages": len(figures),
                      "checks": "passed", "S22_RGBA_sha256": preprint["rgba_pixel_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
