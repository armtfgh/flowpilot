from flora_translate.engine import llm_agents


def test_prompt_content_capture_is_opt_in():
    llm_agents.clear_llm_runtime_overrides()
    hidden = llm_agents._base_event(
        api_name="test",
        provider="test",
        model="test",
        max_tokens=1,
        system="system secret",
        user_content="user secret",
    )
    assert "system_prompt" not in hidden
    llm_agents.set_llm_runtime_overrides(capture_content=True)
    visible = llm_agents._base_event(
        api_name="test",
        provider="test",
        model="test",
        max_tokens=1,
        system="system secret",
        user_content="user secret",
    )
    llm_agents.clear_llm_runtime_overrides()
    assert visible["system_prompt"] == "system secret"
    assert visible["user_prompt"] == "user secret"

