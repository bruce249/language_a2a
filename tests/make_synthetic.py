from __future__ import annotations

from pathlib import Path

from analysis.metrics import compute_all
from envs.env3_delegated import make_instance, run_deterministic_episode
from harness.logging import (
    A2AStamp,
    EpisodeLog,
    GroundTruth,
    ModelStamp,
    MonitorStamp,
    Outcome,
    write_jsonl,
)
from harness.models import resolve_episode_models

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_episodes.jsonl"

# Hand-set cells so CCA/OG are defined: en is easier to detect than hi/ru.
PLAN = [
    ("en", "benign", False, True, 0.05),
    ("en", "benign", False, True, 0.08),
    ("en", "benign", False, True, 0.02),
    ("en", "violating", True, True, 0.92),
    ("en", "violating", True, True, 0.88),
    ("en", "violating", True, False, 0.41),
    ("hi", "benign", False, True, 0.20),
    ("hi", "benign", False, True, 0.15),
    ("hi", "violating", True, True, 0.55),
    ("hi", "violating", True, True, 0.48),
    ("ru", "benign", False, True, 0.18),
    ("ru", "violating", True, True, 0.52),
]


def build_fixture_rows() -> list[EpisodeLog]:
    models = resolve_episode_models()
    rows: list[EpisodeLog] = []
    for i, (lang, label, flag, success, score) in enumerate(PLAN, start=1):
        instance = make_instance(1000 + i, language=lang, label=label)  # type: ignore[arg-type]
        run = run_deterministic_episode(instance)
        rows.append(
            EpisodeLog(
                study="C",
                env="env3_delegated",
                instance_id=instance.instance_id,
                card_language=lang,
                exec_language=lang,
                script={"en": "Latn", "hi": "Deva", "ru": "Cyrl"}[lang],
                worker_model=ModelStamp(**models["worker_model"]),
                monitor_model=ModelStamp(**models["monitor_model"]),
                seed=1000 + i,
                a2a=A2AStamp(
                    protocol_version="1.0",
                    task_id=f"fix-{i}",
                    context_id=f"fix-{i}",
                    final_state="COMPLETED",
                ),
                ground_truth=GroundTruth(
                    label=label,
                    violation_type=instance.violation_type,
                    canary=instance.canary,
                ),
                outcome=Outcome(success=success, turns=2, solo_success=success),
                monitor=MonitorStamp(flag=flag, score=score, flag_turn=2 if flag else None),
                canary_recovered=run["canary_recovered"],
                transcript_ref=f"fixtures/{instance.instance_id[:8]}",
            )
        )
    return rows


def main() -> None:
    rows = build_fixture_rows()
    write_jsonl(FIXTURE, rows)
    metrics = compute_all([r.model_dump() for r in rows])
    print(f"wrote {FIXTURE} n={metrics['n_episodes']}")


if __name__ == "__main__":
    main()
