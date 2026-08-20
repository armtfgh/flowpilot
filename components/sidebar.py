"""FlowPilot sidebar navigation."""

import streamlit as st


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## FlowPilot")
        st.caption("Inventory-constrained batch-to-flow design")
        _model_routing_status()
        st.divider()

        st.markdown("##### DESIGN")
        _nav_button("FlowPilot Design", "flora_design", "nav_design")
        _nav_button("Inventory Manager", "inventory", "nav_inventory")

        st.divider()

        st.markdown("##### EVALUATE")
        _nav_button("Protocol Diagnostics", "diagnose", "nav_diagnose")
        _nav_button("Condition Optimization", "optimize", "nav_optimize")

        st.divider()

        with st.expander("KNOWLEDGE", expanded=False):
            _nav_button("Fundamentals", "fundamentals", "nav_fundamentals")
            _nav_button("Literature Mining", "scout", "nav_scout")
            _nav_button("Knowledge Extraction", "prism", "nav_prism")
            _nav_button("Knowledge Base", "corpus", "nav_corpus")

        st.divider()
        _corpus_status()

    return st.session_state.get("page", "flora_design")


def _nav_button(label: str, page: str, key: str) -> None:
    current = st.session_state.get("page", "flora_design")
    if st.button(
        label,
        use_container_width=True,
        key=key,
        type="primary" if current == page else "secondary",
    ):
        st.session_state.page = page
        st.rerun()


def _model_routing_status():
    try:
        import flora_translate.config as cfg

        upstream_models = [
            cfg.MODEL_INPUT_PARSER,
            cfg.MODEL_CHEMISTRY_AGENT,
            cfg.MODEL_TRANSLATION,
            cfg.MODEL_OUTPUT_FORMATTER,
            cfg.MODEL_CONVERSATION_AGENT,
        ]
        upstream_is_claude = all(str(model).startswith("claude") for model in upstream_models)
        council_is_4o = cfg.ENGINE_PROVIDER == "openai" and cfg.ENGINE_MODEL_OPENAI == "gpt-4o"

        with st.expander("Model routing", expanded=False):
            if upstream_is_claude:
                st.success("Upstream: Claude")
            else:
                st.warning("Upstream: mixed/non-Claude")
            st.caption(
                f"Parser: {cfg.MODEL_INPUT_PARSER}\n\n"
                f"Chemistry: {cfg.MODEL_CHEMISTRY_AGENT}\n\n"
                f"Translation: {cfg.MODEL_TRANSLATION}"
            )

            if council_is_4o:
                st.success("Council/downstream: OpenAI GPT-4o")
            else:
                st.warning("Council/downstream is not GPT-4o")
            st.caption(
                f"Provider: {cfg.ENGINE_PROVIDER}\n\n"
                f"OpenAI model: {cfg.ENGINE_MODEL_OPENAI}\n\n"
                f"Lightweight upstream mode: {cfg.LIGHTWEIGHT_UPSTREAM_MODE}"
            )
    except Exception as exc:
        st.warning(f"Model routing unavailable: {exc}")


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
