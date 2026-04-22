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
                resp = client.messages.create(
                    model=ENGINE_MODEL_ANTHROPIC,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                )
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
            # Exhausted max turns without a final text response
            logger.warning("call_llm_with_tools: exhausted %d tool turns without final text", max_tool_turns)
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
            resp = client.chat.completions.create(
                model=ENGINE_MODEL_OPENAI,
                max_tokens=max_tokens,
                tools=openai_tools,
                tool_choice="auto",
                messages=messages,
            )
            choice = resp.choices[0]
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
