from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from harness.config import experiment, model_id, model_record
from harness.secrets import api_model_id, require_xai, xai_api_key

XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")


def _client() -> OpenAI:
    require_xai()
    return OpenAI(api_key=xai_api_key(), base_url=XAI_BASE_URL)


def grok_complete(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Chat Completions against xAI. reasoning_effort defaults to none (no thinking tokens)."""
    model = model or model_id("worker")
    if reasoning_effort is None:
        try:
            reasoning_effort = model_record(model).get("reasoning_effort")
        except KeyError:
            reasoning_effort = None
        if not reasoning_effort:
            reasoning_effort = str(experiment().get("worker_reasoning_effort", "none"))
    effort = str(reasoning_effort)
    temp = temperature
    if temp is None:
        temp = float(experiment()["decode"].get("worker_temperature", 0.2))
    client = _client()
    kwargs: dict[str, Any] = {
        "model": api_model_id(model),
        "temperature": temp,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if effort and effort != "omit":
        kwargs["extra_body"] = {"reasoning_effort": effort}
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0].message
    text = (choice.content or "").strip()
    usage = response.usage
    return {
        "text": text,
        "tokens_in": int(getattr(usage, "prompt_tokens", 0) or 0),
        "tokens_out": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(
            getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0)
            or 0
        ),
    }
