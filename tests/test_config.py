import os
from igris.models.config import Config


def test_config_defaults():
    # Ensure defaults are loaded when no environment variables are set
    os.environ.pop("LOCAL_LLM_PROVIDER", None)
    cfg = Config.load()
    assert cfg.local_llm.provider == "ollama"
    assert cfg.local_llm.model == "phi4-mini"
    assert cfg.fallback_llm.provider == "openai"