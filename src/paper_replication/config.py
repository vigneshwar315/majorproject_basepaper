"""Runtime configuration read from environment variables / a local ``.env`` file.

Secrets are never stored in ``repr`` output and never logged. Provider and model are configured
explicitly; there is no silent fallback to another provider or model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"

VALID_PROVIDERS = ("gemini", "openai", "mock")


class ConfigError(ValueError):
    """Raised for invalid or incomplete configuration (never contains secret values)."""


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, '#' comments, optional quotes). No variable expansion."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def load_dotenv(path: Path | None = None, override: bool = False) -> None:
    """Load ``.env`` from the project root into ``os.environ`` (existing variables win unless override)."""
    for k, v in parse_env_file(path or (PROJECT_ROOT / ".env")).items():
        if override or k not in os.environ:
            os.environ[k] = v


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    model: str = ""
    gemini_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    timeout_s: float = 60.0
    max_retries: int = 3
    temperature: Optional[float] = None  # None = provider default (some models reject a temperature)

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def api_key(self) -> str:
        return self.gemini_api_key if self.provider == "gemini" else self.openai_api_key

    def has_credentials(self) -> bool:
        return self.is_mock or bool(self.api_key())

    def validate(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {self.provider!r}")
        if self.is_mock:
            return
        if not self.model:
            raise ConfigError(
                "LLM_MODEL is required for provider "
                f"{self.provider!r}. Run `python -m paper_replication.cli list-models` to see ids valid for your key."
            )
        if not self.api_key():
            name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
            raise ConfigError(f"{name} is not set. Put it in your local .env (never in source code).")

    def public_summary(self) -> dict:
        """Status information that is safe to show in the UI / logs (no secrets)."""
        return {
            "provider": self.provider,
            "model": self.model or ("mock-deterministic" if self.is_mock else "(not set)"),
            "credentials_present": self.has_credentials(),
            "offline_mock_mode": self.is_mock,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
        }


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def settings_from_env(load_file: bool = True) -> Settings:
    if load_file:
        load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "mock").strip().lower() or "mock"
    return Settings(
        provider=provider,
        model=os.environ.get("LLM_MODEL", "").strip(),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        timeout_s=_float("LLM_TIMEOUT_S", 60.0),
        max_retries=int(_float("LLM_MAX_RETRIES", 3)),
        temperature=(_float("LLM_TEMPERATURE", 0.0) if os.environ.get("LLM_TEMPERATURE", "").strip() else None),
    )
