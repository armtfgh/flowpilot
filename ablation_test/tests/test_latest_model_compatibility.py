from flora_translate.engine import llm_agents
from ablation_test.src.runner import _parse_json_object


def test_latest_models_use_provider_compatible_token_limit():
    assert llm_agents._openai_token_limit("gpt-5.6-terra", 123) == {
        "max_completion_tokens": 123
    }
    assert llm_agents._openai_token_limit("gpt-4o", 123) == {
        "max_tokens": 123
    }


def test_latest_models_omit_unsupported_temperature(monkeypatch):
    monkeypatch.setattr(
        llm_agents,
        "_RUNTIME_OVERRIDES",
        {"temperature": 0.0, "seed": 42},
    )
    assert llm_agents._runtime_kwargs("anthropic", "claude-sonnet-5") == {}
    assert llm_agents._runtime_kwargs("openai", "gpt-5.6-terra") == {
        "seed": 42
    }
    assert llm_agents._runtime_kwargs("openai", "gpt-4o") == {
        "temperature": 0.0,
        "seed": 42,
    }


def test_anthropic_text_ignores_adaptive_thinking_blocks():
    class Block:
        def __init__(self, block_type, text=None):
            self.type = block_type
            self.text = text

    assert llm_agents._anthropic_text(
        [Block("thinking"), Block("text", '{"status":"ok"}')]
    ) == '{"status":"ok"}'


def test_json_parser_raises_for_truncated_one_shot_output():
    import json

    try:
        _parse_json_object('```json\n{"residence_time_min": 5')
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Truncated JSON must remain an invalid outcome")


def test_reduced_frontier_config_is_matched_and_cost_controlled():
    import json

    from ablation_test.src.cases import ROOT

    config = json.loads(
        (ROOT / "configs" / "reduced_frontier_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(config["profiles"]["publication"]["case_ids"]) == 5
    assert len(config["conditions"]) == 3
    assert config["repeats"]["publication"] == 3
    assert {
        value["variant"] for value in config["conditions"].values()
    } == {"full", "general_one_shot"}
