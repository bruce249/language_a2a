from __future__ import annotations

"""Author-written card packs live in cards/localized/. No raw MT in the Study A run."""

from agents.card_builder import LOCALIZED_DIR, load_localized


def available_card_languages() -> list[str]:
    return sorted(p.stem for p in LOCALIZED_DIR.glob("*.json"))


def card_pack(language: str) -> dict:
    return load_localized(language)
