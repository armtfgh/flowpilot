"""FLORA — Process diagram renderer."""

import streamlit as st
from pathlib import Path


def render_process_diagram(svg_path: str = "", png_path: str = ""):
    """Render the process flow diagram (SVG inline, PNG downloadable)."""
    svg_exists = svg_path and Path(svg_path).exists()
    png_exists = png_path and Path(png_path).exists()

    if svg_exists:
        svg_content = Path(svg_path).read_text()
        st.markdown(
            f'<div style="background:white;padding:16px;border-radius:10px;'
            f'border:1px solid #e2e8f0;">{svg_content}</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            with open(svg_path, "rb") as f:
                st.download_button(
                    "Download SVG", f,
                    "flora_process.svg", "image/svg+xml",
                    use_container_width=True,
                )
        with col2:
            if png_exists:
                with open(png_path, "rb") as f:
                    st.download_button(
                        "Download PNG", f,
                        "flora_process.png", "image/png",
                        use_container_width=True,
                    )
    elif png_exists:
        st.image(png_path, use_container_width=True)
    else:
        st.info("No process diagram available yet.")
