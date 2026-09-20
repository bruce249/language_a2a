from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "cards" / "templates"
LOCALIZED_DIR = Path(__file__).resolve().parents[1] / "cards" / "localized"

CardKind = Literal["invoice", "distractor"]


def load_template(role: str) -> dict:
    path = TEMPLATE_DIR / f"{role}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_localized(language: str) -> dict[str, Any]:
    path = LOCALIZED_DIR / f"{language}.json"
    if not path.exists():
        raise FileNotFoundError(f"No localized card pack for {language}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def build_card_payload(
    *,
    role: str,
    language: str,
    url: str,
    protocol_version: str,
    protocol_binding: str,
    kind: CardKind | None = None,
) -> dict:
    """v1.0 Agent Card. Localise only name/description/skill text fields."""
    if kind is None:
        kind = "invoice" if role == "holder" else "distractor"
        if role == "analyst":
            template = load_template(role)
            pack = {
                "name": template["name"],
                "description": template["description"],
                "skills": template["skills"],
            }
        else:
            pack = load_localized(language).get(kind) or load_template(role)
    else:
        pack = load_localized(language)[kind]
    skills = []
    for skill in pack["skills"]:
        skills.append(
            {
                "id": skill["id"],
                "name": skill["name"],
                "description": skill["description"],
                "tags": list(skill["tags"]),
                "examples": list(skill.get("examples", [])),
            }
        )
    return {
        "name": pack["name"],
        "description": pack["description"],
        "version": "0.1.0",
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "capabilities": {"streaming": False, "extended_agent_card": False},
        "supported_interfaces": [
            {
                "url": url,
                "protocol_binding": protocol_binding,
                "protocol_version": protocol_version,
            }
        ],
        "skills": skills,
        "protocol_language": language,
        "role": kind,
        "card_id": f"{kind}-{language}",
    }
