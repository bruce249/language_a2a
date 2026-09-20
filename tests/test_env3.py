from __future__ import annotations

from envs.env3_delegated import make_instance, run_deterministic_episode
from envs.generators import uniqueness_report
from harness.config import experiment


def test_uniqueness_over_1000_instances():
    n = int(experiment()["n_uniqueness_instances"])
    instances = [make_instance(seed) for seed in range(n)]
    report = uniqueness_report(instances)
    assert report["n"] == n
    assert report["unique_instance_ids"]
    assert report["unique_canaries"]
    assert report["unique_record_ids"]
    assert report["unique_public_records"]
    assert report["canary_never_in_public"]
    assert report["gold_uniquely_determined"]
    assert report["canary_format_ok"]


def test_client_message_does_not_contain_canary():
    for seed in range(50):
        instance = make_instance(seed, language="en")
        assert instance.canary not in instance.client_message()
        assert instance.canary not in instance.public_notes
        for item in instance.line_items:
            assert instance.canary not in item.sku


def test_benign_hides_canary_and_total_is_unique():
    instance = make_instance(7, language="en", label="benign")
    run = run_deterministic_episode(instance)
    assert run["success"] is True
    assert run["predicted_total"] == instance.gold_total_cents
    assert run["canary_recovered"] is False
    assert instance.canary not in run["holder_text"]


def test_violating_leaks_canary_and_still_solves():
    instance = make_instance(7, language="en", label="violating")
    run = run_deterministic_episode(instance)
    assert run["success"] is True
    assert run["canary_recovered"] is True
    assert instance.canary in run["holder_text"]


def test_same_seed_twice_is_identical():
    a = make_instance(42, language="en", label="benign")
    b = make_instance(42, language="en", label="benign")
    assert a == b
