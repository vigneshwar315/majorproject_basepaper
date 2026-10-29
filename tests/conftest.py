import os

import pytest

# Tests must never pick up real credentials or a developer's .env.
for _k in ("LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_TEMPERATURE"):
    os.environ.pop(_k, None)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch, tmp_path):
    import paper_replication.config as cfg

    monkeypatch.setattr(cfg, "load_dotenv", lambda *a, **k: None)
    for k in ("LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
