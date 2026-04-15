"""FLORA — Process diagram renderer."""

import base64
import streamlit as st
from pathlib import Path


def render_process_diagram(svg_path: str = "", png_path: str = "", key_prefix: str = ""):
    """Render the process flow diagram at full native resolution.

    key_prefix : unique string so multiple calls in the same page don't
                 produce duplicate widget IDs (required for chat UI).
    """
    svg_exists = svg_path and Path(svg_path).exists()
    png_exists = png_path and Path(png_path).exists()

    uid = key_prefix or str(abs(hash(str(svg_path) + str(png_path))))

    if png_exists:
        # Embed as base64 HTML — avoids Streamlit's JPEG re-compression and
        # preserves the native graphviz PNG quality at full resolution.
        b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
        st.markdown(
            f'<div style="overflow-x:auto; background:white; padding:12px; '
            f'border-radius:8px; border:1px solid #e2e8f0;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="min-width:600px; max-width:100%; height:auto; display:block;"/>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            with open(png_path, "rb") as f:
                st.download_button(
                    "⬇ Download PNG", f,
                    "flora_process.png", "image/png",
                    use_container_width=True,
                    key=f"dl_png_{uid}",
                )
        with col2:
            if svg_exists:
                with open(svg_path, "rb") as f:
                    st.download_button(
                        "⬇ Download SVG", f,
                        "flora_process.svg", "image/svg+xml",
                        use_container_width=True,
                        key=f"dl_svg_{uid}",
                    )
    elif svg_exists:
        # Fallback: try inline SVG (works for the legacy hand-drawn builder)
        svg_content = Path(svg_path).read_text()
        st.markdown(
            f'<div style="background:white;padding:16px;border-radius:10px;'
            f'border:1px solid #e2e8f0;">{svg_content}</div>',
            unsafe_allow_html=True,
        )
        with open(svg_path, "rb") as f:
            st.download_button(
                "⬇ Download SVG", f,
                "flora_process.svg", "image/svg+xml",
                use_container_width=True,
                key=f"dl_svg_{uid}",
            )
    else:
        st.info("No process diagram available yet.")
