"""Render an autosaved FlowPilot result for browser-based GUI regression checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="FlowPilot GUI Result Smoke", layout="wide")

css_path = PROJECT_ROOT / "assets/style.css"
if css_path.is_file():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

result_path = Path(os.environ.get("FLOWPILOT_GUI_RESULT", ""))
if not result_path.is_absolute():
    result_path = PROJECT_ROOT / result_path
st.title("FlowPilot Result Visual Check")
if not result_path.is_file():
    st.error("Set FLOWPILOT_GUI_RESULT to an existing autosaved result.json file.")
    st.stop()

from pages.flora_design_unified import _render_result

result = json.loads(result_path.read_text(encoding="utf-8"))
st.caption(str(result_path.resolve()))
_render_result(result, key_prefix="visual_smoke")
