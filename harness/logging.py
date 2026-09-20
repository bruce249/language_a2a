from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ModelStamp(BaseModel):
    id: str
    version: str
    temperature: float | None = None


class A2AStamp(BaseModel):
    protocol_version: str
    task_id: str
    context_id: str
    final_state: str


class GroundTruth(BaseModel):
    label: str
    violation_type: str | None = None
    canary: str | None = None
    correct_language: str | None = None


class Outcome(BaseModel):
    success: bool
    turns: int
    tokens_in: int = 0
    tokens_out: int = 0
    solo_success: bool | None = None


class MonitorStamp(BaseModel):
    flag: bool
    score: float
    flag_turn: int | None = None
    rationale: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


class EpisodeLog(BaseModel):
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    study: str
    env: str
    instance_id: str
    card_language: str
    exec_language: str
    script: str
    worker_model: ModelStamp
    monitor_model: ModelStamp
    seed: int
    a2a: A2AStamp
    ground_truth: GroundTruth
    outcome: Outcome
    monitor: MonitorStamp
    canary_recovered: bool
    transcript_ref: str
    selected_language: str | None = None
    capability_hallucination: bool = False

    def to_json(self) -> str:
        return self.model_dump_json()


def write_jsonl(path: Path, episodes: list[EpisodeLog]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(episode.to_json() + "\n")


def append_jsonl(path: Path, episode: EpisodeLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(episode.to_json() + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
