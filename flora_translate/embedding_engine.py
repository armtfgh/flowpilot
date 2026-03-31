"""FLORA-Translate — Embedding engine: generates summaries and embeddings."""

import json
import logging

import anthropic
from openai import OpenAI

from flora_translate.config import EMBEDDING_MODEL, SUMMARY_MODEL
from flora_translate.schemas import BatchRecord, ProcessRecord

logger = logging.getLogger("flora.embedding")


def _get_claude():
    return anthropic.Anthropic()


def _get_openai():
    return OpenAI()

SUMMARY_SYSTEM = (
    "You are a chemistry expert. Summarize this flow chemistry process record "
    "in 4-6 sentences of natural language. Include: reaction type, photocatalyst "
    "(if any), key conditions (temperature, residence time, solvent), reactor "
    "type, and the main reason why flow improved on batch. Be specific and "
    "include numbers. Do not say 'this paper' — write as a direct factual "
    "description."
)

QUERY_SUMMARY_SYSTEM = (
    "You are a chemistry expert. Given this batch protocol, write a 3-4 sentence "
    "natural language description for semantic search. Include: reaction type, "
    "photocatalyst name and class, solvent, temperature, reaction time, and any "
    "special features (deoxygenation, sensitizer, etc.). Be specific and include "
    "all numerical values present."
)


class EmbeddingEngine:
    """Generate text summaries and vector embeddings."""

    def generate_record_summary(self, record: ProcessRecord) -> str:
        """Generate a natural language summary of a process record for embedding."""
        resp = _get_claude().messages.create(
            model=SUMMARY_MODEL,
            max_tokens=512,
            system=SUMMARY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(record.model_dump(), indent=2, default=str),
                }
            ],
        )
        return resp.content[0].text.strip()

    def generate_query_summary(self, batch_record: BatchRecord) -> str:
        """Generate a search-optimized summary of a batch protocol."""
        resp = _get_claude().messages.create(
            model=SUMMARY_MODEL,
            max_tokens=256,
            system=QUERY_SUMMARY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        batch_record.model_dump(exclude_none=True), indent=2
                    ),
                }
            ],
        )
        return resp.content[0].text.strip()

    def embed(self, text: str) -> list[float]:
        """Generate a vector embedding using OpenAI text-embedding-3-small."""
        resp = _get_openai().embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call."""
        resp = _get_openai().embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
        )
        return [item.embedding for item in resp.data]
