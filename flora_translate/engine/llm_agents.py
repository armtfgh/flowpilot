"""
FLORA ENGINE — LLM provider abstraction.

Single entry point for all council LLM calls.
Provider is configured via ENGINE_PROVIDER in flora_translate/config.py:

    ENGINE_PROVIDER = "anthropic"   →  Claude Sonnet (ENGINE_MODEL_ANTHROPIC)
    ENGINE_PROVIDER = "openai"      →  GPT-4o        (ENGINE_MODEL_OPENAI)

To switch: change ENGINE_PROVIDER in config.py. That is the only change needed.
"""

from __future__ import annotations

import logging

import anthropic

from flora_translate.config import (
    ENGINE_PROVIDER,
    ENGINE_MODEL_ANTHROPIC,
    ENGINE_MODEL_OPENAI,
    ENGINE_MODEL_OLLAMA,
    OLLAMA_BASE_URL,
    ENGINE_MAX_ROUNDS,
)

logger = logging.getLogger("flora.engine.llm_agents")

_ANTHROPIC_CLIENT = None
_OPENAI_CLIENT    = None
_OLLAMA_CLIENT    = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        import openai
        _OPENAI_CLIENT = openai.OpenAI()
    return _OPENAI_CLIENT


def _get_ollama_client():
    """OpenAI-compatible client pointing at the local Ollama server."""
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is None:
        import openai
        _OLLAMA_CLIENT = openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",          # required by the library, ignored by Ollama
        )
    return _OLLAMA_CLIENT


def get_max_rounds() -> int:
    """Return the configured number of council rounds for the active provider."""
    return ENGINE_MAX_ROUNDS.get(ENGINE_PROVIDER, 2)


def call_llm(system: str, user_content: str, max_tokens: int) -> str:
    """Route a council LLM call to the configured provider.

    Provider is set by ENGINE_PROVIDER in config.py:
      "anthropic" → Claude Sonnet
      "openai"    → GPT-4o (or ENGINE_MODEL_OPENAI)
      "ollama"    → local model at OLLAMA_BASE_URL
    """
    if ENGINE_PROVIDER in ("openai",):
        resp = _get_openai_client().chat.completions.create(
            model=ENGINE_MODEL_OPENAI,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()

    elif ENGINE_PROVIDER == "ollama":
        client = _get_ollama_client()

        # gemma4 (and similar) are "thinking" models: they spend all tokens on an
        # internal scratchpad ("reasoning" field) and never write to "content".
        # Disable thinking via the Ollama API flag (0.9+) AND via the /no_think
        # user-message prefix (all versions that support it).
        resp = client.chat.completions.create(
            model=ENGINE_MODEL_OLLAMA,
            max_tokens=max_tokens,
            stream=False,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"/no_think\n{user_content}"},
            ],
            extra_body={"think": False},
        )

        choice  = resp.choices[0] if resp.choices else None
        content = choice.message.content if choice else None

        # Fallback: if thinking was NOT disabled, the answer is in "reasoning"
        if not content and choice:
            extra     = getattr(choice.message, "model_extra", None) or {}
            reasoning = extra.get("reasoning") or getattr(choice.message, "reasoning", None)
            if reasoning:
                logger.info("Ollama: content empty — extracting from reasoning field")
                content = reasoning

        if not content:
            logger.error(
                "Ollama returned no content (model=%s finish_reason=%s)",
                ENGINE_MODEL_OLLAMA,
                choice.finish_reason if choice else "N/A",
            )
            return ""

        return content.strip()

    else:  # "anthropic" (default)
        resp = _get_anthropic_client().messages.create(
            model=ENGINE_MODEL_ANTHROPIC,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text.strip()
