"""Secret redaction helpers. Used for logs, exceptions, reports and UI output."""

from __future__ import annotations

import os
import re

_SECRET_ENV_NAMES = ("GEMINI_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google API key shape
    re.compile(r"sk-[0-9A-Za-z_\-]{16,}"),  # OpenAI key shape
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)(\s*[=:]\s*|\s+)[^\s,;\"']{6,}"),
]
REDACTED = "[REDACTED]"


def redact(text: object) -> str:
    """Return ``text`` as a string with known secret values and key-shaped tokens removed."""
    s = str(text)
    for name in _SECRET_ENV_NAMES:
        val = os.environ.get(name)
        if val and len(val) >= 6:
            s = s.replace(val, REDACTED)
    for pat in _PATTERNS:
        s = pat.sub(lambda m: REDACTED if m.lastindex is None else f"{m.group(1)}={REDACTED}", s)
    return s


def redact_obj(obj):
    """Recursively redact strings in dict/list structures (for JSON logs)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
