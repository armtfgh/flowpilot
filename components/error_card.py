"""FLORA — Clean error display component."""

import traceback
import streamlit as st


def render_error(exception: Exception, context: str = "FLORA"):
    """Render a classified error card with suggested fix."""
    err_str = str(exception)
    tb_str = traceback.format_exc()

    if "chromadb" in err_str.lower() or "collection" in err_str.lower():
        title = "Corpus not available"
        fix = "Go to Knowledge Extraction and index some papers first."
    elif "api_key" in err_str.lower() or "anthropic" in err_str.lower() or "openai" in err_str.lower():
        title = "API key error"
        fix = "Check ANTHROPIC_API_KEY and OPENAI_API_KEY in your .env file."
    elif "cairosvg" in err_str.lower():
        title = "Diagram generation failed"
        fix = "Install cairosvg: `pip install cairosvg`"
    elif "validation error" in err_str.lower() or "pydantic" in err_str.lower():
        title = "Data validation error"
        fix = "The LLM returned an unexpected format. Try running again."
    else:
        title = f"Error in {context}"
        fix = "See technical details below."

    st.error(f"**{title}**\n\n{err_str[:300]}")
    st.caption(f"Suggested fix: {fix}")
    with st.expander("Technical details"):
        st.code(tb_str, language="python")
