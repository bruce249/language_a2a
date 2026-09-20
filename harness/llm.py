from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from harness.config import experiment
from harness.secrets import api_model_id, require_openai

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _client() -> OpenAI:
    require_openai()
    return OpenAI()


def complete_json(
    *,
    model: str,
    system: str,
    user: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """One OpenAI Responses call. Returns parsed JSON plus usage. Never logs secrets."""
    effort = reasoning_effort or str(experiment().get("monitor_reasoning_effort", "low"))
    client = _client()
    kwargs: dict[str, Any] = {
        "model": api_model_id(model),
        "instructions": system,
        "input": user,
        "reasoning": {"effort": effort},
    }
    response = client.responses.create(**kwargs)
    text = getattr(response, "output_text", None) or ""
    if not text:
        parts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    parts.append(value)
        text = "\n".join(parts)
    usage = getattr(response, "usage", None)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    parsed = _parse_json(text)
    parsed["_usage"] = {"tokens_in": tokens_in, "tokens_out": tokens_out, "raw_text": text}
    return parsed


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    return {"flag": False, "score": 0.5, "rationale": "unparseable_monitor_output"}
