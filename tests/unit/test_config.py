import pytest

from paper_replication import config
from paper_replication.config import ConfigError, Settings, parse_env_file, settings_from_env
from paper_replication.llm import get_provider
from paper_replication.redact import redact, redact_obj

FAKE_GEMINI = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKE12345"
FAKE_OPENAI = "sk-FAKEFAKEFAKEFAKEFAKE1234"


def test_default_is_mock_without_credentials():
    s = settings_from_env(load_file=False)
    assert s.is_mock and s.has_credentials()
    assert get_provider(s).is_mock


def test_real_provider_requires_model_and_key():
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        get_provider(Settings(provider="gemini", gemini_api_key=FAKE_GEMINI))
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        get_provider(Settings(provider="gemini", model="m"))
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_provider(Settings(provider="openai", model="m"))
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        get_provider(Settings(provider="claude", model="m"))


def test_no_silent_fallback_to_mock():
    with pytest.raises(ConfigError):
        get_provider(Settings(provider="openai", model="m"))  # missing key must raise, not return the mock


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Gemini ")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_TIMEOUT_S", "30")
    s = settings_from_env(load_file=False)
    assert (s.provider, s.model, s.temperature, s.timeout_s) == ("gemini", "some-model", 0.2, 30.0)
    monkeypatch.setenv("LLM_TIMEOUT_S", "abc")
    with pytest.raises(ConfigError):
        settings_from_env(load_file=False)


def test_temperature_unset_is_none(monkeypatch):
    assert settings_from_env(load_file=False).temperature is None


def test_public_summary_and_repr_never_contain_keys():
    s = Settings(provider="gemini", model="m", gemini_api_key=FAKE_GEMINI, openai_api_key=FAKE_OPENAI)
    assert FAKE_GEMINI not in repr(s) and FAKE_OPENAI not in repr(s)
    summary = str(s.public_summary())
    assert FAKE_GEMINI not in summary and "credentials_present" in summary


def test_env_file_parser_and_gitignore(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\nLLM_PROVIDER=openai\nOPENAI_API_KEY='abc123456'\nBROKEN\n", encoding="utf-8")
    assert parse_env_file(f) == {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "abc123456"}
    assert parse_env_file(tmp_path / "missing") == {}
    ignore = (config.PROJECT_ROOT / ".gitignore").read_text()
    assert ".env" in ignore.splitlines()
    example = (config.PROJECT_ROOT / ".env.example").read_text()
    assert "AIza" not in example and "sk-" not in example


def test_redaction_of_env_values_and_patterns(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "customsecretvalue99")
    assert "customsecretvalue99" not in redact("failed with key customsecretvalue99 here")
    assert FAKE_GEMINI not in redact(f"url?key={FAKE_GEMINI}")
    assert FAKE_OPENAI not in redact(f"Authorization: Bearer {FAKE_OPENAI}")
    assert "hunter2hunter2" not in redact("api_key=hunter2hunter2")
    out = redact_obj({"a": [f"x {FAKE_OPENAI}"], "b": {"c": 1}})
    assert FAKE_OPENAI not in str(out) and out["b"]["c"] == 1
    assert redact("nothing secret here") == "nothing secret here"


def test_cli_accepts_provider_flag_before_and_after_subcommand():
    from paper_replication.cli import build_parser

    ap = build_parser()
    a = ap.parse_args(["--provider", "mock", "run-all"])
    b = ap.parse_args(["run-all", "--provider", "mock", "--no-verify"])
    assert a.provider == "mock" and b.provider == "mock" and b.verify is False and a.verify is True
