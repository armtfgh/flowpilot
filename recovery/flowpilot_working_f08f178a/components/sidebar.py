"""FLORA — Sidebar navigation."""

import streamlit as st


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## FLORA")
        st.caption("Flow Literature Oriented Retrieval Agent")
        st.divider()

        st.markdown("##### DESIGN")
        if st.button("FLORA Design", use_container_width=True, key="nav_design"):
            st.session_state.page = "flora_design"

        st.divider()

        st.markdown("##### EVALUATE")
        if st.button("Protocol Diagnostics", use_container_width=True, key="nav_diagnose"):
            st.session_state.page = "diagnose"
        if st.button("Condition Optimization", use_container_width=True, key="nav_optimize"):
            st.session_state.page = "optimize"

        st.divider()

        with st.expander("KNOWLEDGE", expanded=False):
            if st.button("Fundamentals", use_container_width=True, key="nav_fundamentals"):
                st.session_state.page = "fundamentals"
            if st.button("Literature Mining", use_container_width=True, key="nav_scout"):
                st.session_state.page = "scout"
            if st.button("Knowledge Extraction", use_container_width=True, key="nav_prism"):
                st.session_state.page = "prism"
            if st.button("Knowledge Base", use_container_width=True, key="nav_corpus"):
                st.session_state.page = "corpus"

        st.divider()
        _corpus_status()

    return st.session_state.get("page", "flora_design")


def _corpus_status():
    try:
        import chromadb
        client = chromadb.PersistentClient(path="flora_translate/data/chroma_db")
        col = client.get_or_create_collection("flora_records")
        n = col.count()
        if n > 0:
            st.success(f"Corpus: {n} records")
        else:
            st.warning("Corpus empty — index papers first")
    except Exception:
        st.error("ChromaDB not available")
