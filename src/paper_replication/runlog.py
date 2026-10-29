"""Run ids, JSON artefact writing and the append-only run log. Everything is redacted before it touches disk."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import LOGS_DIR, RESULTS_DIR
from .redact import redact_obj


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_obj(obj), indent=2, sort_keys=False, default=str), encoding="utf-8")
    return path


def append_log(event: str, run_id: Optional[str] = None, log_path: Optional[Path] = None, **fields: Any) -> None:
    path = log_path or (LOGS_DIR / "run_log.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utc_now(), "event": event, "run_id": run_id, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact_obj(rec), default=str) + "\n")


def read_log(log_path: Optional[Path] = None, limit: int = 500) -> list[dict]:
    path = log_path or (LOGS_DIR / "run_log.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_dir(run_id: str, kind: str = "runs", results_dir: Optional[Path] = None) -> Path:
    return (results_dir or RESULTS_DIR) / kind / run_id
