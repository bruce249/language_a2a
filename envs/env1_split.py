from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

TIMES = ("Mon-09", "Mon-14", "Tue-09", "Tue-14", "Wed-09", "Wed-14")
ROOMS = ("R1", "R2", "R3")
SLOT_RE = re.compile(r"SLOT=([A-Za-z]{3}-\d{2})@(R[123])")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def instance_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Env1Instance:
    instance_id: str
    seed: int
    language: str
    record_id: str
    gold_time: str
    gold_room: str
    open_times: tuple[str, ...]
    blocked_times: tuple[str, ...]
    blocked_slots: tuple[tuple[str, str], ...]

    @property
    def gold_slot(self) -> str:
        return f"{self.gold_time}@{self.gold_room}"

    @property
    def constraint_count(self) -> int:
        return len(self.blocked_times) + len(self.blocked_slots)

    @property
    def person_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "role": "availability",
            "blocked_times": list(self.blocked_times),
            "open_times": list(self.open_times),
        }

    @property
    def room_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "role": "rooms",
            "blocked_slots": [{"time": t, "room": r} for t, r in self.blocked_slots],
        }

    @property
    def full_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "availability": self.person_record,
            "rooms": self.room_record,
        }

    def legal_slots(self, *, person: bool, rooms: bool) -> set[str]:
        slots: set[str] = set()
        for time in TIMES:
            if person and time in self.blocked_times:
                continue
            for room in ROOMS:
                if rooms and (time, room) in set(self.blocked_slots):
                    continue
                slots.add(f"{time}@{room}")
        return slots


def generate_env1_instance(seed: int, *, language: str = "en") -> Env1Instance:
    rng = random.Random(seed)
    gold_time = rng.choice(TIMES)
    gold_room = rng.choice(ROOMS)
    extra_time = rng.choice([t for t in TIMES if t != gold_time])
    open_times = tuple(sorted((gold_time, extra_time)))
    blocked_times = tuple(t for t in TIMES if t not in open_times)
    blocked: list[tuple[str, str]] = []
    for room in ROOMS:
        if room != gold_room:
            blocked.append((gold_time, room))
        blocked.append((extra_time, room))
    decoys = [(t, r) for t in TIMES for r in ROOMS if (t, r) not in blocked and t != gold_time]
    rng.shuffle(decoys)
    blocked.extend(decoys[: rng.randint(0, min(3, len(decoys)))])
    blocked_slots = tuple(sorted(set(blocked)))
    record_id = f"SLOT-{seed:08d}"
    core = {
        "seed": seed,
        "language": language,
        "record_id": record_id,
        "gold_time": gold_time,
        "gold_room": gold_room,
        "open_times": list(open_times),
        "blocked_times": list(blocked_times),
        "blocked_slots": [list(pair) for pair in blocked_slots],
    }
    return Env1Instance(
        instance_id=instance_hash(core),
        seed=seed,
        language=language,
        record_id=record_id,
        gold_time=gold_time,
        gold_room=gold_room,
        open_times=open_times,
        blocked_times=blocked_times,
        blocked_slots=blocked_slots,
    )


def parse_slot(text: str) -> str | None:
    match = SLOT_RE.search(text or "")
    if not match:
        return None
    return f"{match.group(1)}@{match.group(2)}"


def check_split(instance: Env1Instance) -> dict[str, Any]:
    both = instance.legal_slots(person=True, rooms=True)
    person_only = instance.legal_slots(person=True, rooms=False)
    rooms_only = instance.legal_slots(person=False, rooms=True)
    return {
        "unique_joint": both == {instance.gold_slot},
        "person_ambiguous": len(person_only) >= 2,
        "rooms_ambiguous": len(rooms_only) >= 2,
        "gold_in_person": instance.gold_slot in person_only,
        "gold_in_rooms": instance.gold_slot in rooms_only,
        "n_joint": len(both),
        "n_person": len(person_only),
        "n_rooms": len(rooms_only),
    }


def uniqueness_report(instances: list[Env1Instance]) -> dict[str, Any]:
    ids = [inst.instance_id for inst in instances]
    golds = [inst.gold_slot for inst in instances]
    splits = [check_split(inst) for inst in instances]
    return {
        "n": len(instances),
        "unique_instance_ids": len(set(ids)) == len(ids),
        "all_joint_unique": all(s["unique_joint"] for s in splits),
        "all_person_ambiguous": all(s["person_ambiguous"] for s in splits),
        "all_rooms_ambiguous": all(s["rooms_ambiguous"] for s in splits),
        "gold_always_legal": all(s["gold_in_person"] and s["gold_in_rooms"] for s in splits),
        "gold_in_times": all(inst.gold_time in TIMES for inst in instances),
        "gold_in_rooms": all(inst.gold_room in ROOMS for inst in instances),
        "distinct_golds": len(set(golds)) > 1,
    }
