from __future__ import annotations

import base64
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image

from visualization.benchmark_figure_studio import (
    FIGURE_SPECS,
    palette_defaults,
    render_svg_png,
    save_figure_export,
    transform_figure_svg,
)


def test_transform_is_non_destructive_and_scales_fonts() -> None:
    spec = FIGURE_SPECS["figs5-1"]
    before = hashlib.sha256(spec.svg_path.read_bytes()).hexdigest()

    transformed = transform_figure_svg(
        "figs5-1",
        palette=palette_defaults("figs5-1"),
        font_scale=1.5,
    ).decode("utf-8")

    after = hashlib.sha256(spec.svg_path.read_bytes()).hexdigest()
    assert before == after
    assert "font-size: 16.800px" in transformed


def test_semantic_vector_color_is_replaced() -> None:
    palette = palette_defaults("figs5-1")
    palette["one_shot"] = "#123456"

    transformed = transform_figure_svg("figs5-1", palette=palette).decode("utf-8")

    assert "#123456" in transformed
    assert "#d46459" not in transformed.lower()


def test_embedded_error_map_palette_is_replaced() -> None:
    palette = palette_defaults("figs5-3")
    palette["no_error"] = "#DDF0FF"
    transformed = transform_figure_svg("figs5-3", palette=palette).decode("utf-8")
    encoded_images = re.findall(
        r'xlink:href="data:image/png;base64,\s*([^\"]+)"',
        transformed,
        flags=re.DOTALL,
    )

    assert encoded_images
    pixels = []
    for encoded in encoded_images:
        image = Image.open(
            io.BytesIO(base64.b64decode(re.sub(r"\s+", "", encoded)))
        ).convert("RGB")
        pixels.append(np.asarray(image).reshape(-1, 3))
    all_pixels = np.concatenate(pixels)
    assert np.any(np.all(all_pixels == np.array([0xDD, 0xF0, 0xFF]), axis=1))


def test_png_and_reproducible_export(tmp_path) -> None:
    settings = {
        "figure_id": "figs5-4",
        "palette": palette_defaults("figs5-4"),
        "font_scale": 1.0,
        "line_scale": 1.0,
        "width_px": 1200,
    }
    svg = transform_figure_svg("figs5-4", palette=settings["palette"])
    png = render_svg_png(svg, width_px=1200)

    destination = save_figure_export(
        "figs5-4", svg, png, settings, export_root=tmp_path
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert (destination / "figs5-4.svg").read_bytes() == svg
    assert (destination / "figs5-4.png").read_bytes() == png
    assert json.loads((destination / "settings.json").read_text())["figure_id"] == "figs5-4"
