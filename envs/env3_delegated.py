from __future__ import annotations

from typing import Literal

from envs.generators import (
    Env3Instance,
    analyst_reply,
    canary_recovered,
    generate_env3_instance,
    holder_reply,
    parse_total,
)


def make_instance(
    seed: int,
    language: str = "en",
    label: Literal["benign", "violating"] | None = None,
    violation_base_rate: float = 0.30,
) -> Env3Instance:
    return generate_env3_instance(
        seed,
        language=language,
        label=label,
        violation_base_rate=violation_base_rate,
    )


def run_deterministic_episode(instance: Env3Instance) -> dict:
    """In-process ENV-3: holder private state is not part of the client message."""
    arm: Literal["benign", "violating"] = instance.label
    client_msg = instance.client_message()
    if instance.canary in client_msg:
        raise RuntimeError("Canary leaked into the client message; monitor would be confounded")
    holder = holder_reply(instance, arm)
    analyst = analyst_reply(holder)
    transcript = "\n".join([client_msg, holder, analyst])
    recovered = canary_recovered(transcript, instance.canary)
    predicted = parse_total(analyst)
    success = predicted == instance.gold_total_cents
    return {
        "holder_text": holder,
        "analyst_text": analyst,
        "transcript": transcript,
        "canary_recovered": recovered,
        "predicted_total": predicted,
        "success": success,
        "turns": 2,
    }
