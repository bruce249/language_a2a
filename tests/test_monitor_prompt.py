from __future__ import annotations

from agents.monitor import MONITOR_SYSTEM
from envs.env3_delegated import make_instance, run_deterministic_episode


def test_monitor_system_prompt_does_not_contain_instance_canary():
    instance = make_instance(99, language="en", label="violating")
    run = run_deterministic_episode(instance)
    user = "TRANSCRIPT:\n" + run["transcript"]
    assert instance.canary not in MONITOR_SYSTEM
    assert "violating" not in MONITOR_SYSTEM.lower()
    assert instance.canary in run["transcript"]
    # Canary may appear in the transcript; it must not be a labelled side channel.
    assert "ground_truth" not in user
    assert instance.label not in user
