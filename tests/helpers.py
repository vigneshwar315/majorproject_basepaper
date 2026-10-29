"""Shared test doubles."""

from __future__ import annotations

import json

from paper_replication.llm.base import LLMProvider, LLMRequest, LLMResponse


class ScriptedProvider(LLMProvider):
    """Returns pre-scripted responses per task (callables or dicts) and records every request."""

    name = "scripted"
    is_mock = True

    def __init__(self, script: dict[str, list], model: str = "scripted-1"):
        super().__init__(model, max_retries=0)
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[LLMRequest] = []

    def _call(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        queue = self.script.get(req.task, [])
        if not queue:
            raise AssertionError(f"no scripted response left for task {req.task}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        data = item(req) if callable(item) else item
        return LLMResponse(data, json.dumps(data), self.name, self.model, True, 0.0, None, None)


MATCH = {"verdict": "match", "reason": "ok"}
Z1_SPEC = json.dumps({"name": "z1", "description": "d", "expr": {"op": "safe_div", "args": [
    {"op": "agg", "fn": "count", "over": "requests", "where": {"op": "field", "name": "accepted"}},
    {"op": "agg", "fn": "count", "over": "requests"}]}})
