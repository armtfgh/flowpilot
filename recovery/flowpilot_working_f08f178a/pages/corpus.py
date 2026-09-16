"""FLORA — Knowledge Base / Corpus Browser page."""

import streamlit as st


def render():
    st.title("Knowledge Base")
    st.caption("KNOWLEDGE")
    st.markdown("Browse, filter, and inspect all records in the corpus.")

    try:
        import chromadb
        import pandas as pd

        client = chromadb.PersistentClient(path="flora_translate/data/chroma_db")
        col = client.get_or_create_collection("flora_records")
        all_data = col.get(include=["metadatas", "documents"])

        if not all_data["ids"]:
            st.warning("Corpus is empty. Run Knowledge Extraction first.")
            return

        metas = all_data["metadatas"]
        df = pd.DataFrame(metas)
        df["record_id"] = all_data["ids"]
        st.markdown(f"**{len(df)} records** in corpus.")

        # ── Filters ──
        with st.sidebar:
            st.markdown("##### Filter corpus")
            if "chemistry_class" in df.columns:
                classes = sorted(df["chemistry_class"].dropna().unique().tolist())
                sel_classes = st.multiselect("Reaction class", classes)
            else:
                sel_classes = []

            if "year" in df.columns:
                yr_min = int(df["year"].min()) if df["year"].min() > 0 else 2010
                yr_max = int(df["year"].max()) if df["year"].max() > 0 else 2026
                yr_range = st.slider("Year", yr_min, yr_max, (yr_min, yr_max))
            else:
                yr_range = (2010, 2026)

            flow_only = st.checkbox("Flow records only", value=False)

        # Apply filters
        mask = pd.Series([True] * len(df))
        if sel_classes and "chemistry_class" in df.columns:
            mask &= df["chemistry_class"].isin(sel_classes)
        if "year" in df.columns:
            mask &= df["year"].between(*yr_range)
        if flow_only and "process_mode" in df.columns:
            mask &= df["process_mode"].isin(["flow", "both"])

        df_f = df[mask].reset_index(drop=True)
        st.caption(f"Showing {len(df_f)} of {len(df)} records.")

        # ── Table ──
        display = ["record_id", "doi", "title", "year", "chemistry_class",
                    "photocatalyst", "process_mode", "confidence"]
        avail = [c for c in display if c in df_f.columns]
        st.dataframe(df_f[avail], use_container_width=True, hide_index=True)

        # ── Inspector ──
        st.divider()
        st.subheader("Record inspector")
        if not df_f.empty:
            sel_id = st.selectbox("Select record", df_f["record_id"].tolist())
            if sel_id:
                idx = all_data["ids"].index(sel_id)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Metadata**")
                    st.json(all_data["metadatas"][idx])
                with c2:
                    st.markdown("**Embedding summary**")
                    st.text_area(
                        "Summary", all_data["documents"][idx],
                        height=250, disabled=True,
                    )

        st.download_button(
            "Export filtered records (CSV)",
            df_f.to_csv(index=False), "corpus.csv", "text/csv",
        )

    except Exception as e:
        from components.error_card import render_error
        render_error(e, "Knowledge Base")
