from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import flora_translate.config as cfg
from flora_translate.engine import llm_agents


MODEL_KEYS = (
    "MODEL_INPUT_PARSER",
    "MODEL_CHEMISTRY_AGENT",
    "MODEL_TRANSLATION",
    "MODEL_OUTPUT_FORMATTER",
    "MODEL_REVISION_AGENT",
    "MODEL_CONVERSATION_AGENT",
    "MODEL_EMBEDDING_SUMMARY",
    "MODEL_TOPOLOGY_POLISHER",
)


def credential_status(bundle: dict[str, Any]) -> tuple[bool, str]:
    provider = bundle["provider"]
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY is not set"
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is not set"
    return True, ""


@contextmanager
def activate_bundle(bundle: dict[str, Any]) -> Iterator[None]:
    """Temporarily route every upstream and council component to one bundle."""
    original_cfg = {key: getattr(cfg, key) for key in MODEL_KEYS}
    original_cfg.update(
        {
            "LIGHTWEIGHT_UPSTREAM_MODE": cfg.LIGHTWEIGHT_UPSTREAM_MODE,
            "ENGINE_PROVIDER": cfg.ENGINE_PROVIDER,
            "ENGINE_MODEL_ANTHROPIC": cfg.ENGINE_MODEL_ANTHROPIC,
            "ENGINE_MODEL_OPENAI": cfg.ENGINE_MODEL_OPENAI,
            "ENGINE_MODEL_OLLAMA": cfg.ENGINE_MODEL_OLLAMA,
            "OLLAMA_BASE_URL": cfg.OLLAMA_BASE_URL,
        }
    )
    original_llm = {
        "ENGINE_PROVIDER": llm_agents.ENGINE_PROVIDER,
        "ENGINE_MODEL_ANTHROPIC": llm_agents.ENGINE_MODEL_ANTHROPIC,
        "ENGINE_MODEL_OPENAI": llm_agents.ENGINE_MODEL_OPENAI,
        "ENGINE_MODEL_OLLAMA": llm_agents.ENGINE_MODEL_OLLAMA,
        "OLLAMA_BASE_URL": llm_agents.OLLAMA_BASE_URL,
        "_ANTHROPIC_CLIENT": llm_agents._ANTHROPIC_CLIENT,
        "_OPENAI_CLIENT": llm_agents._OPENAI_CLIENT,
        "_OLLAMA_CLIENT": llm_agents._OLLAMA_CLIENT,
    }

    provider = bundle["provider"]
    model = bundle["model"]
    try:
        for key in MODEL_KEYS:
            setattr(cfg, key, model)
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = bundle.get("upstream_mode", "auto")
        cfg.ENGINE_PROVIDER = provider
        llm_agents.ENGINE_PROVIDER = provider

        if provider == "anthropic":
            cfg.ENGINE_MODEL_ANTHROPIC = model
            llm_agents.ENGINE_MODEL_ANTHROPIC = model
        elif provider == "openai":
            cfg.ENGINE_MODEL_OPENAI = model
            llm_agents.ENGINE_MODEL_OPENAI = model
        else:
            cfg.ENGINE_MODEL_OLLAMA = model
            llm_agents.ENGINE_MODEL_OLLAMA = model
            cfg.OLLAMA_BASE_URL = bundle["base_url"]
            llm_agents.OLLAMA_BASE_URL = bundle["base_url"]
            llm_agents._OLLAMA_CLIENT = None
        yield
    finally:
        for key in MODEL_KEYS:
            setattr(cfg, key, original_cfg[key])
        for key in (
            "LIGHTWEIGHT_UPSTREAM_MODE",
            "ENGINE_PROVIDER",
            "ENGINE_MODEL_ANTHROPIC",
            "ENGINE_MODEL_OPENAI",
            "ENGINE_MODEL_OLLAMA",
            "OLLAMA_BASE_URL",
        ):
            setattr(cfg, key, original_cfg[key])
        for key, value in original_llm.items():
            setattr(llm_agents, key, value)


def endpoint_health(bundle: dict[str, Any]) -> dict[str, Any]:
    provider = bundle["provider"]
    if provider in {"anthropic", "openai"}:
        ready, reason = credential_status(bundle)
        if not ready:
            return {
                "provider": provider,
                "model": bundle["model"],
                "reachable": False,
                "reason": reason,
            }
        try:
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(
                    http_client=llm_agents.build_verified_httpx_client()
                )
                models = client.models.list(limit=100).data
            else:
                import openai

                client = openai.OpenAI(
                    http_client=llm_agents.build_verified_httpx_client()
                )
                models = client.models.list().data
            ids = sorted(item.id for item in models)
            configured = bundle["model"]
            return {
                "provider": provider,
                "model": configured,
                "reachable": True,
                "model_advertised": configured in ids,
                "advertised_model_count": len(ids),
            }
        except Exception as exc:
            return {
                "provider": provider,
                "model": bundle["model"],
                "reachable": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    if provider != "ollama":
        return {
            "provider": provider,
            "model": bundle["model"],
            "reachable": False,
            "reason": f"Unsupported provider: {provider}",
        }

    import urllib.error
    import urllib.request
    import json

    url = bundle["base_url"].rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = [item.get("id") for item in payload.get("data", [])]
        configured = bundle["model"]
        return {
            "provider": provider,
            "model": configured,
            "base_url": bundle["base_url"],
            "reachable": True,
            "advertised_models": ids,
            "model_advertised": configured in ids,
        }
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "provider": provider,
            "model": bundle["model"],
            "base_url": bundle["base_url"],
            "reachable": False,
            "reason": str(exc),
        }
