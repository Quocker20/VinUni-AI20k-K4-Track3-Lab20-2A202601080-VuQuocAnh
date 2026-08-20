from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_model == "openai/gpt-4o-mini"
    assert settings.openai_model
    assert settings.max_iterations >= 1
