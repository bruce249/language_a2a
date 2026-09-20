from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from agents.card_builder import build_card_payload
from agents.orchestrator import TASK, default_orchestrator_id, route_cards
from analysis.metrics import compute_all, language_preference_index
from harness.config import ROOT, a2a, experiment, languages, pricing
from harness.logging import (
    A2AStamp,
    EpisodeLog,
    GroundTruth,
    ModelStamp,
    MonitorStamp,
    Outcome,
    append_jsonl,
    read_jsonl,
)
from harness.secrets import xai_key_present


def _script(code: str) -> str:
    return str(languages()["languages"][code]["script"])


def _usd(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = pricing()["per_million"][model]
    return (tokens_in * float(rates["input"]) + tokens_out * float(rates["output"])) / 1_000_000


class SpendGuard:
    def __init__(self, grok_stop: float) -> None:
        self.grok_stop = grok_stop
        self.grok = 0.0

    def charge(self, model: str, tokens_in: int, tokens_out: int) -> None:
        self.grok += _usd(model, tokens_in, tokens_out)

    def allow(self, model: str, est_in: int, est_out: int) -> bool:
        return self.grok + _usd(model, est_in, est_out) <= self.grok_stop

    def snapshot(self) -> dict[str, float]:
        return {"grok_usd": round(self.grok, 4), "grok_cap": self.grok_stop}


def lineup(correct_language: str, codes: list[str], cfg: dict) -> list[dict]:
    cards = []
    for lang in codes:
        kind = "invoice" if lang == correct_language else "distractor"
        spec = cfg["hosts"]["holder"]
        cards.append(
            build_card_payload(
                role="holder",
                language=lang,
                url=f"http://{spec['host']}:{spec['port']}",
                protocol_version=str(cfg["protocol_version"]),
                protocol_binding=str(cfg["protocol_binding"]),
                kind=kind,
            )
        )
    return cards


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {
        (r["ground_truth"].get("correct_language"), int(r["seed"]))
        for r in read_jsonl(path)
        if r.get("study") == "A"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(experiment().get("study_a_n", 100)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--languages", default=None)
    args = parser.parse_args()
    if not xai_key_present():
        raise RuntimeError("XAI_API_KEY is empty in .env")

    model = default_orchestrator_id()
    codes = (
        [c.strip() for c in args.languages.split(",")]
        if args.languages
        else list(languages()["collab_codes"])
    )
    n = 2 if args.smoke else args.n
    if args.smoke:
        codes = ["en"]
    seed_base = int(experiment().get("study_a_seed_base", 50000))
    cfg = a2a()
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "study_a.jsonl"
    spend = SpendGuard(float(experiment().get("study_a_grok_stop_usd", 11.0)))
    done = _done(log_path)

    stopped = False
    for lang_i, correct in enumerate(codes):
        if stopped:
            break
        for i in range(n):
            seed = seed_base + lang_i * 1000 + i
            if (correct, seed) in done:
                continue
            if not spend.allow(model, 2500, 200):
                print(json.dumps({"stop": "grok_budget", **spend.snapshot()}), flush=True)
                stopped = True
                break
            rng = random.Random(seed)
            cards = lineup(correct, codes if not args.smoke else list(languages()["collab_codes"]), cfg)
            rng.shuffle(cards)
            result = None
            last_error = None
            for attempt in range(3):
                try:
                    result = route_cards(cards, model=model)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(2 * (attempt + 1))
            if result is None:
                print(
                    json.dumps(
                        {
                            "correct": correct,
                            "i": i + 1,
                            "error": f"{type(last_error).__name__}: {last_error}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            spend.charge(model, result["tokens_in"], result["tokens_out"])
            selected = result["selected_language"]
            success = selected == correct
            transcript = json.dumps(
                {"task": TASK, "order": [c["card_id"] for c in cards], "route": result},
                ensure_ascii=False,
                indent=2,
            )
            tpath = out_dir / f"study-a-{correct}-{seed}.json"
            tpath.write_text(transcript, encoding="utf-8")
            append_jsonl(
                log_path,
                EpisodeLog(
                    study="A",
                    env="card_routing",
                    instance_id=f"route-{seed}",
                    card_language=correct,
                    exec_language="en",
                    script=_script(correct),
                    worker_model=ModelStamp(id=model, version=model, temperature=0.0),
                    monitor_model=ModelStamp(id="none", version="none", temperature=0.0),
                    seed=seed,
                    a2a=A2AStamp(
                        protocol_version=str(cfg["protocol_version"]),
                        task_id=f"study-a-{seed}",
                        context_id=f"study-a-{seed}",
                        final_state="COMPLETED" if selected else "FAILED",
                    ),
                    ground_truth=GroundTruth(
                        label="routing",
                        violation_type=None,
                        canary=None,
                        correct_language=correct,
                    ),
                    outcome=Outcome(
                        success=success,
                        turns=1,
                        tokens_in=result["tokens_in"],
                        tokens_out=result["tokens_out"],
                    ),
                    monitor=MonitorStamp(flag=False, score=0.0, rationale="study_a_no_monitor"),
                    canary_recovered=False,
                    transcript_ref=str(tpath),
                    selected_language=selected,
                    capability_hallucination=result["capability_hallucination"],
                ),
            )
            print(
                json.dumps(
                    {
                        "correct": correct,
                        "i": i + 1,
                        "n": n,
                        "selected": selected,
                        "success": success,
                        "hallucination": result["capability_hallucination"],
                        **spend.snapshot(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    rows = [r for r in read_jsonl(log_path) if r.get("study") == "A"] if log_path.exists() else []
    by_correct: dict[str, list] = {}
    for row in rows:
        by_correct.setdefault(row["ground_truth"]["correct_language"], []).append(row)
    cells = {}
    for lang, group in by_correct.items():
        n_g = len(group)
        cells[lang] = {
            "n": n_g,
            "routing_accuracy": sum(1 for r in group if r["outcome"]["success"]) / n_g,
            "hallucination_rate": sum(1 for r in group if r.get("capability_hallucination")) / n_g,
            "LPI": language_preference_index(rows, lang),
        }
    summary = {
        "model": model,
        "n_per_language": n,
        "languages": codes,
        "spend": spend.snapshot(),
        "cells": cells,
        "metrics": compute_all(rows) if rows else {},
        "log": str(log_path),
    }
    out = ROOT / "results" / "processed" / "study_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
