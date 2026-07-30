from flora_translate.retriever import VectorRetriever


def test_retriever_signature_accepts_exclusions():
    annotations = VectorRetriever.retrieve.__annotations__
    assert "exclude_record_ids" in annotations

