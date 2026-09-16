import pytest

import flora_translate.embedding_engine as embedding_engine
from flora_translate.retriever import VectorRetriever
from flora_translate.schemas import BatchRecord, ChemistryPlan
from flora_translate.vector_store import VectorStore


class _Collection:
    def __init__(self) -> None:
        self._documents = [
            "Palladium Suzuki coupling in DMF produces a biaryl in flow.",
            "Blue-light photocatalytic oxidation with oxygen gas.",
            "Hydrogenation over palladium in a packed-bed reactor.",
        ]

    def count(self) -> int:
        return len(self._documents)

    def get(self, **kwargs) -> dict:
        return {
            "ids": ["suzuki", "photo", "hydrogen"],
            "documents": self._documents,
            "metadatas": [
                {"confidence": 3, "solvent": "DMF"},
                {"confidence": 3, "solvent": "MeCN"},
                {"confidence": 3, "solvent": "MeOH"},
            ],
        }


class _FailingEmbeddingEngine:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider unavailable")


class _FallbackStore:
    def __init__(self) -> None:
        self.lexical_calls = 0

    def query(self, **kwargs) -> dict:
        raise AssertionError("semantic query must not run after embedding failure")

    def query_lexical(self, **kwargs) -> dict:
        self.lexical_calls += 1
        return {
            "ids": [["suzuki", "photo", "hydrogen"]],
            "documents": [[
                "Suzuki coupling in DMF.",
                "Photochemical oxidation.",
                "Packed-bed hydrogenation.",
            ]],
            "metadatas": [[
                {"solvent": "DMF"},
                {"solvent": "MeCN"},
                {"solvent": "MeOH"},
            ]],
            "distances": [[0.1, 1.0, 1.2]],
        }


def test_lexical_query_ranks_relevant_record_first() -> None:
    collection = _Collection()
    store = VectorStore.__new__(VectorStore)
    store.collection = collection
    store.pairs_collection = collection

    result = store.query_lexical(
        "Suzuki palladium biaryl coupling in DMF",
        n_results=3,
    )

    assert result["ids"][0][0] == "suzuki"
    assert result["distances"][0][0] < result["distances"][0][1]


def test_retriever_uses_lexical_fallback_after_embedding_failure() -> None:
    store = _FallbackStore()
    retriever = VectorRetriever(store=store)
    retriever.engine = _FailingEmbeddingEngine()

    results = retriever.retrieve(
        BatchRecord(solvent="DMF", temperature_C=50.0),
        chemistry_plan=ChemistryPlan(
            reaction_name="Suzuki coupling",
            reaction_class="cross-coupling",
            mechanism_type="",
        ),
    )

    assert store.lexical_calls == 1
    assert results[0]["record_id"] == "suzuki"


def test_explicit_lexical_mode_never_calls_embedding_provider() -> None:
    batch = BatchRecord(solvent="DMF", temperature_C=50.0)
    store = _FallbackStore()
    retriever = VectorRetriever(store=store)

    class _ForbiddenEmbeddingEngine:
        def embed(self, text: str):
            raise AssertionError("embedding provider must not run in lexical mode")

    retriever.engine = _ForbiddenEmbeddingEngine()
    results = retriever.retrieve(
        batch,
        top_k=1,
        chemistry_plan=ChemistryPlan(reaction_name="Suzuki coupling"),
        retrieval_mode="lexical",
    )

    assert results
    assert store.lexical_calls == 1


def test_embedding_failure_opens_process_local_circuit(monkeypatch) -> None:
    calls = 0

    class _Embeddings:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("quota exhausted")

    class _Client:
        embeddings = _Embeddings()

    monkeypatch.setattr(embedding_engine, "_embedding_provider_available", None)
    monkeypatch.setattr(embedding_engine, "_get_openai", lambda: _Client())
    engine = embedding_engine.EmbeddingEngine()

    with pytest.raises(RuntimeError, match="quota exhausted"):
        engine.embed("first")
    with pytest.raises(RuntimeError, match="disabled"):
        engine.embed("second")

    assert calls == 1
