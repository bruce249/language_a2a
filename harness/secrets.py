from __future__ import annotations

import os

from dotenv import load_dotenv

from harness.config import ROOT

load_dotenv(ROOT / ".env", override=False)


def openai_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def xai_key_present() -> bool:
    return bool(
        os.environ.get("XAI_API_KEY", "").strip()
        or os.environ.get("GROK_API_KEY", "").strip()
    )


def require_openai() -> None:
    if not openai_key_present():
        raise RuntimeError("OPENAI_API_KEY is empty. Put it in .env at the project root.")


def require_xai() -> None:
    if not xai_key_present():
        raise RuntimeError("XAI_API_KEY is empty. Put it in .env at the project root.")


def xai_api_key() -> str:
    require_xai()
    return os.environ.get("XAI_API_KEY", "").strip() or os.environ.get("GROK_API_KEY", "").strip()


def api_model_id(inspect_id: str) -> str:
    """openai/gpt-5.6-luna -> gpt-5.6-luna; grok/grok-4.6 -> grok-4.6"""
    if "/" in inspect_id:
        return inspect_id.split("/", 1)[1]
    return inspect_id
