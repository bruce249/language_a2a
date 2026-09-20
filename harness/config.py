from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
load_dotenv(ROOT / ".env", override=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


@lru_cache(maxsize=None)
def languages() -> dict[str, Any]:
    return _read_yaml(CONFIGS / "languages.yaml")


@lru_cache(maxsize=None)
def models() -> dict[str, Any]:
    return _read_yaml(CONFIGS / "models.yaml")


@lru_cache(maxsize=None)
def a2a() -> dict[str, Any]:
    return _read_yaml(CONFIGS / "a2a.yaml")


@lru_cache(maxsize=None)
def experiment() -> dict[str, Any]:
    return _read_yaml(CONFIGS / "experiment.yaml")


@lru_cache(maxsize=None)
def pricing() -> dict[str, Any]:
    return _read_yaml(CONFIGS / "pricing.yaml")


def model_id(role: str, slot: str = "default") -> str:
    roster = models()["roles"][role]
    if slot == "default":
        return str(roster["default"])
    for entry in roster["roster"]:
        if entry.get("strength") == slot or entry.get("id") == slot:
            return str(entry["id"])
    raise KeyError(f"No model for role={role!r} slot={slot!r}")


def model_record(model_identifier: str) -> dict[str, Any]:
    for role in models()["roles"].values():
        for entry in role["roster"]:
            if entry["id"] == model_identifier:
                return dict(entry)
    raise KeyError(f"Unknown model id {model_identifier!r}")
