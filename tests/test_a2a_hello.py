from __future__ import annotations

import pytest

from harness.runner import run_in_process


def test_in_process_hello_writes_valid_episode():
    benign = run_in_process(3, label="benign")
    violating = run_in_process(3, label="violating")
    assert benign.ground_truth.label == "benign"
    assert violating.ground_truth.label == "violating"
    assert benign.canary_recovered is False
    assert violating.canary_recovered is True
    assert benign.outcome.success is True
    assert violating.outcome.success is True
    assert benign.a2a.protocol_version == "1.0"
    assert benign.worker_model.id.startswith("grok/")
    assert benign.monitor_model.id == "openai/gpt-5.6-luna"


@pytest.mark.integration
def test_a2a_two_agent_env3(monkeypatch):
    pytest.importorskip("a2a")
    from harness.runner import run_a2a

    episode = run_a2a(5, label="violating")
    assert episode.canary_recovered is True
    assert episode.outcome.success is True
    assert episode.a2a.final_state == "COMPLETED"
