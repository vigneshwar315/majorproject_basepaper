"""Session-state keys and provider selection for the UI."""

from __future__ import annotations

import dataclasses
from typing import Optional

import streamlit as st

from paper_replication.config import ConfigError, Settings, settings_from_env
from paper_replication.llm import LLMProvider, get_provider
from paper_replication.redact import redact

MODE_MOCK = "Offline mock (no API key needed)"
MODE_CONFIGURED = "Provider from .env"


def init_state() -> None:
    defaults = {
        "llm_mode": MODE_MOCK, "instance": None, "instance_origin": "", "result": None, "selected_plan": None,
        "preference": "", "metric_requests": [], "vehicles_df": None, "requests_df": None, "settings_form": None,
        "distance_matrix": None, "last_error": None, "dmg_tester": None, "mcdm_alt": None, "doc_to_open": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def current_settings() -> Settings:
    s = settings_from_env()
    if st.session_state.get("llm_mode") == MODE_MOCK:
        return dataclasses.replace(s, provider="mock", model="")
    return s


def current_provider() -> tuple[Optional[LLMProvider], Optional[str]]:
    """(provider, error). No silent fallback: an unusable configured provider returns (None, message)."""
    s = current_settings()
    try:
        return get_provider(s), None
    except ConfigError as exc:
        return None, redact(exc)


def provider_info() -> dict:
    p, err = current_provider()
    if p is None:
        return {"error": err}
    return {**p.info(), "settings": current_settings().public_summary()}
