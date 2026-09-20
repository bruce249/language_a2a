from __future__ import annotations

import json
from typing import Any

from harness.config import model_id
from harness.grok import grok_complete
from harness.llm import _parse_json


TASK = (
    "A downstream process needs public invoice line items "
    "(sku, qty, unit_price_cents) from the agent that already holds "
    "the vendor record. Route to that record-holder. "
    "Do not pick a meeting scheduler."
)


def default_orchestrator_id() -> str:
    return model_id("orchestrator")


def route_cards(cards: list[dict[str, Any]], *, model: str, task: str = TASK) -> dict[str, Any]:
    menu = []
    for card in cards:
        menu.append(
            {
                "card_id": card["card_id"],
                "language": card["protocol_language"],
                "name": card["name"],
                "description": card["description"],
                "skills": [
                    {
                        "id": skill["id"],
                        "name": skill["name"],
                        "description": skill["description"],
                    }
                    for skill in card["skills"]
                ],
            }
        )
    system = (
        "You are an A2A router. Choose exactly one agent card for the task.\n"
        "Return JSON only with keys: card_id, skill_id, rationale.\n"
        "card_id must be one of the listed cards. skill_id must appear on that card.\n"
        "Do not invent skills. Do not pick a card whose skills cannot do the task."
    )
    user = json.dumps({"task": task, "cards": menu}, ensure_ascii=False, indent=2)
    raw = grok_complete(model=model, system=system, user=user, temperature=0.0)
    parsed = _parse_json(raw["text"])
    card_id = str(parsed.get("card_id") or "")
    skill_id = str(parsed.get("skill_id") or "")
    chosen = next((c for c in cards if c["card_id"] == card_id), None)
    legal_skills = {s["id"] for s in chosen["skills"]} if chosen else set()
    return {
        "card_id": card_id,
        "skill_id": skill_id,
        "rationale": str(parsed.get("rationale") or "")[:400],
        "selected_language": chosen["protocol_language"] if chosen else None,
        "capability_hallucination": bool(chosen) and skill_id not in legal_skills,
        "tokens_in": raw["tokens_in"],
        "tokens_out": raw["tokens_out"],
        "raw_text": raw["text"],
    }
