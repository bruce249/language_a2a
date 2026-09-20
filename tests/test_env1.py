from __future__ import annotations

from envs.env1_split import check_split, generate_env1_instance, parse_slot, uniqueness_report


def test_uniqueness_and_split_over_200_instances():
    instances = [generate_env1_instance(seed, language="en") for seed in range(200)]
    report = uniqueness_report(instances)
    assert report["n"] == 200
    assert report["unique_instance_ids"]
    assert report["all_joint_unique"]
    assert report["all_person_ambiguous"]
    assert report["all_rooms_ambiguous"]
    assert report["gold_always_legal"]
    assert report["distinct_golds"]


def test_same_seed_is_identical():
    a = generate_env1_instance(42, language="hi")
    b = generate_env1_instance(42, language="hi")
    assert a == b
    assert check_split(a)["unique_joint"]


def test_parse_slot_reads_last_machine_line():
    assert parse_slot("ok\nSLOT=Tue-09@R2\n") == "Tue-09@R2"
    assert parse_slot("no slot here") is None
