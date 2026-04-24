"""
FLORA ENGINE — LLM provider abstraction.

Single entry point for all council LLM calls.
Provider is configured via ENGINE_PROVIDER in flora_translate/config.py:

    ENGINE_PROVIDER = "anthropic"   →  Claude Sonnet (ENGINE_MODEL_ANTHROPIC)
    ENGINE_PROVIDER = "openai"      →  GPT-4o        (ENGINE_MODEL_OPENAI)

To switch: change ENGINE_PROVIDER in config.py. That is the only change needed.
"""

from __future__ import annotations

import json
import logging
import time

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
_LLM_OBSERVER     = None
_RUNTIME_OVERRIDES: dict = {}


def set_llm_observer(observer) -> None:
    """Register an optional observer(event_dict) for benchmark telemetry."""
    global _LLM_OBSERVER
    _LLM_OBSERVER = observer


def clear_llm_observer() -> None:
    global _LLM_OBSERVER
    _LLM_OBSERVER = None


def set_llm_runtime_overrides(**kwargs) -> None:
    """Set optional runtime overrides such as temperature or seed."""
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = {k: v for k, v in kwargs.items() if v is not None}


def clear_llm_runtime_overrides() -> None:
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = {}


def get_llm_runtime_overrides() -> dict:
    return dict(_RUNTIME_OVERRIDES)


def _emit_llm_event(event: dict) -> None:
    if _LLM_OBSERVER is None:
        return
    try:
        _LLM_OBSERVER(event)
    except Exception as exc:
        logger.debug("LLM observer raised: %s", exc)


def _usage_to_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    def _get(name: str):
        value = getattr(usage, name, None)
        return value if isinstance(value, (int, float)) else None

    data = {
        "input_tokens": _get("input_tokens"),
        "output_tokens": _get("output_tokens"),
        "prompt_tokens": _get("prompt_tokens"),
        "completion_tokens": _get("completion_tokens"),
        "total_tokens": _get("total_tokens"),
        "cache_creation_input_tokens": _get("cache_creation_input_tokens"),
        "cache_read_input_tokens": _get("cache_read_input_tokens"),
    }
    return {k: v for k, v in data.items() if v is not None}


def _base_event(
    *,
    api_name: str,
    provider: str,
    model: str,
    max_tokens: int,
    system: str,
    user_content: str,
) -> dict:
    event = {
        "api_name": api_name,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "system_chars": len(system or ""),
        "user_chars": len(user_content or ""),
    }
    if "temperature" in _RUNTIME_OVERRIDES:
        event["temperature"] = _RUNTIME_OVERRIDES["temperature"]
    if "seed" in _RUNTIME_OVERRIDES:
        event["seed"] = _RUNTIME_OVERRIDES["seed"]
    return event


def emit_component_llm_event(
    *,
    component: str,
    provider: str,
    model: str,
    max_tokens: int,
    system: str,
    user_content: str,
    resp,
    started: float,
    extra: dict | None = None,
) -> None:
    event = {
        **_base_event(
            api_name=component,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            system=system,
            user_content=user_content,
        ),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": _usage_to_dict(resp),
    }
    if extra:
        event.update(extra)
    _emit_llm_event(event)


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
        kwargs = {}
        if "temperature" in _RUNTIME_OVERRIDES:
            kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
        if "seed" in _RUNTIME_OVERRIDES:
            kwargs["seed"] = _RUNTIME_OVERRIDES["seed"]
        started = time.perf_counter()
        resp = _get_openai_client().chat.completions.create(
            model=ENGINE_MODEL_OPENAI,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            **kwargs,
        )
        content = resp.choices[0].message.content.strip()
        _emit_llm_event({
            **_base_event(
                api_name="call_llm",
                provider="openai",
                model=ENGINE_MODEL_OPENAI,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": _usage_to_dict(resp),
            "response_chars": len(content),
        })
        return content

    elif ENGINE_PROVIDER == "ollama":
        client = _get_ollama_client()
        extra_body = {"think": False}
        kwargs = {}
        if "temperature" in _RUNTIME_OVERRIDES:
            kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
        if "seed" in _RUNTIME_OVERRIDES:
            kwargs["seed"] = _RUNTIME_OVERRIDES["seed"]

        # gemma4 (and similar) are "thinking" models: they spend all tokens on an
        # internal scratchpad ("reasoning" field) and never write to "content".
        # Disable thinking via the Ollama API flag (0.9+) AND via the /no_think
        # user-message prefix (all versions that support it).
        started = time.perf_counter()
        resp = client.chat.completions.create(
            model=ENGINE_MODEL_OLLAMA,
            max_tokens=max_tokens,
            stream=False,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"/no_think\n{user_content}"},
            ],
            extra_body=extra_body,
            **kwargs,
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
            _emit_llm_event({
                **_base_event(
                    api_name="call_llm",
                    provider="ollama",
                    model=ENGINE_MODEL_OLLAMA,
                    max_tokens=max_tokens,
                    system=system,
                    user_content=user_content,
                ),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": _usage_to_dict(resp),
                "response_chars": 0,
                "finish_reason": choice.finish_reason if choice else None,
                "empty_content": True,
            })
            return ""

        content = content.strip()
        _emit_llm_event({
            **_base_event(
                api_name="call_llm",
                provider="ollama",
                model=ENGINE_MODEL_OLLAMA,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": _usage_to_dict(resp),
            "response_chars": len(content),
            "finish_reason": choice.finish_reason if choice else None,
        })
        return content

    else:  # "anthropic" (default)
        kwargs = {}
        if "temperature" in _RUNTIME_OVERRIDES:
            kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
        started = time.perf_counter()
        resp = _get_anthropic_client().messages.create(
            model=ENGINE_MODEL_ANTHROPIC,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            **kwargs,
        )
        content = resp.content[0].text.strip()
        _emit_llm_event({
            **_base_event(
                api_name="call_llm",
                provider="anthropic",
                model=ENGINE_MODEL_ANTHROPIC,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": _usage_to_dict(resp),
            "response_chars": len(content),
            "stop_reason": getattr(resp, "stop_reason", None),
        })
        return content


def call_llm_with_tools(
    system: str,
    user_content: str,
    tools: list[dict],
    tool_executor,
    max_tokens: int,
    max_tool_turns: int = 4,
) -> tuple[str, list[dict]]:
    """Route a tool-using council LLM call to the configured provider.

    Returns (final_text, tool_calls_log) where each log entry is
    {"tool": name, "input": dict, "result": dict}.

    For Ollama: gracefully falls back to call_llm() with empty log
    (most local models don't reliably support tools).
    """
    tool_calls_log: list[dict] = []

    # ── Ollama fallback (no reliable tool support) ───────────────────────────
    if ENGINE_PROVIDER == "ollama":
        logger.info("call_llm_with_tools: Ollama detected — falling back to call_llm (no tools)")
        try:
            text = call_llm(system, user_content, max_tokens=max_tokens)
        except Exception as e:
            logger.warning("call_llm_with_tools Ollama fallback failed: %s", e)
            text = ""
        return text, tool_calls_log

    # ── Anthropic tool loop ──────────────────────────────────────────────────
    if ENGINE_PROVIDER not in ("openai",):  # default: anthropic
        try:
            client = _get_anthropic_client()
            messages = [{"role": "user", "content": user_content}]
            for _ in range(max_tool_turns):
                kwargs = {}
                if "temperature" in _RUNTIME_OVERRIDES:
                    kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
                started = time.perf_counter()
                resp = client.messages.create(
                    model=ENGINE_MODEL_ANTHROPIC,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                    **kwargs,
                )
                _emit_llm_event({
                    **_base_event(
                        api_name="call_llm_with_tools_turn",
                        provider="anthropic",
                        model=ENGINE_MODEL_ANTHROPIC,
                        max_tokens=max_tokens,
                        system=system,
                        user_content=user_content,
                    ),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "usage": _usage_to_dict(resp),
                    "tool_calls_requested": len([b for b in resp.content if getattr(b, "type", "") == "tool_use"]),
                    "stop_reason": getattr(resp, "stop_reason", None),
                })
                if resp.stop_reason != "tool_use":
                    text_blocks = [b for b in resp.content if hasattr(b, "text")]
                    final_text = text_blocks[0].text.strip() if text_blocks else ""
                    return final_text, tool_calls_log
                # Execute all tool_use blocks
                tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
                tool_results = []
                for block in tool_use_blocks:
                    result = tool_executor(block.name, dict(block.input))
                    tool_calls_log.append({
                        "tool": block.name,
                        "input": dict(block.input),
                        "result": result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
            # Tool turns exhausted — force one final text-only response using
            # accumulated tool results already in the message history.
            logger.warning(
                "call_llm_with_tools: exhausted %d tool turns — forcing final text response",
                max_tool_turns,
            )
            try:
                force_msgs = messages + [{
                    "role": "user",
                    "content": (
                        "You have completed all necessary tool calls. "
                        "Now produce your final JSON response. "
                        "Do NOT call any more tools. Output only the JSON."
                    ),
                }]
                started = time.perf_counter()
                final_resp = client.messages.create(
                    model=ENGINE_MODEL_ANTHROPIC,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    tool_choice={"type": "none"},
                    messages=force_msgs,
                    **kwargs,
                )
                _emit_llm_event({
                    **_base_event(
                        api_name="call_llm_with_tools_force_final",
                        provider="anthropic",
                        model=ENGINE_MODEL_ANTHROPIC,
                        max_tokens=max_tokens,
                        system=system,
                        user_content=user_content,
                    ),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "usage": _usage_to_dict(final_resp),
                    "tool_calls_requested": 0,
                    "stop_reason": getattr(final_resp, "stop_reason", None),
                })
                text_blocks = [b for b in final_resp.content if hasattr(b, "text")]
                final_text = text_blocks[0].text.strip() if text_blocks else ""
                if final_text:
                    return final_text, tool_calls_log
            except Exception as force_err:
                logger.warning("call_llm_with_tools: forced final call failed: %s", force_err)
            return "", tool_calls_log
        except Exception as e:
            logger.warning("call_llm_with_tools (anthropic) failed: %s — falling back to call_llm", e)
            try:
                text = call_llm(system, user_content, max_tokens=max_tokens)
            except Exception as e2:
                logger.warning("call_llm_with_tools fallback also failed: %s", e2)
                text = ""
            return text, tool_calls_log

    # ── OpenAI tool loop ─────────────────────────────────────────────────────
    try:
        from flora_translate.engine.tool_definitions import to_openai_tools
        client = _get_openai_client()
        openai_tools = to_openai_tools(tools)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        for _ in range(max_tool_turns):
            kwargs = {}
            if "temperature" in _RUNTIME_OVERRIDES:
                kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
            if "seed" in _RUNTIME_OVERRIDES:
                kwargs["seed"] = _RUNTIME_OVERRIDES["seed"]
            started = time.perf_counter()
            resp = client.chat.completions.create(
                model=ENGINE_MODEL_OPENAI,
                max_tokens=max_tokens,
                tools=openai_tools,
                tool_choice="auto",
                messages=messages,
                **kwargs,
            )
            choice = resp.choices[0]
            _emit_llm_event({
                **_base_event(
                    api_name="call_llm_with_tools_turn",
                    provider="openai",
                    model=ENGINE_MODEL_OPENAI,
                    max_tokens=max_tokens,
                    system=system,
                    user_content=user_content,
                ),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": _usage_to_dict(resp),
                "tool_calls_requested": len(choice.message.tool_calls or []),
                "finish_reason": choice.finish_reason,
            })
            msg = choice.message
            if not msg.tool_calls:
                return (msg.content or "").strip(), tool_calls_log
            # Execute all tool calls
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    fn_input = {}
                result = tool_executor(fn_name, fn_input)
                tool_calls_log.append({
                    "tool": fn_name,
                    "input": fn_input,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        logger.warning("call_llm_with_tools: exhausted %d tool turns without final text", max_tool_turns)
        return "", tool_calls_log
    except Exception as e:
        logger.warning("call_llm_with_tools (openai) failed: %s — falling back to call_llm", e)
        try:
            text = call_llm(system, user_content, max_tokens=max_tokens)
        except Exception as e2:
            logger.warning("call_llm_with_tools fallback also failed: %s", e2)
            text = ""
        return text, tool_calls_log
