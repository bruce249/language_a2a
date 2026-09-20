from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.monitor import GrokMonitor
from analysis.metrics import compute_all
from analysis.selection import mid_auroc_by_language
from harness.config import ROOT, a2a, experiment, languages, model_id, pricing
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
from harness.study_c import run_episode_labeled


def _script(code: str) -> str:
    return str(languages()["languages"][code]["script"])


def _usd(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = pricing()["per_million"][model]
    return (tokens_in * float(rates["input"]) + tokens_out * float(rates["output"])) / 1_000_000


def _label_for(index: int) -> str:
    return "violating" if index % 10 in {0, 3, 6} else "benign"


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


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {
        (r["exec_language"], int(r["seed"]))
        for r in read_jsonl(path)
        if r.get("study") == "E"
    }


def summarize(log_path: Path, spend: SpendGuard, codes: list[str], n: int, worker: str) -> dict:
    rows = [r for r in read_jsonl(log_path) if r.get("study") == "E"] if log_path.exists() else []
    mid_id = model_id("monitor", "mid")
    a_e = mid_auroc_by_language(log_path, mid_id, codes)
    a_c = mid_auroc_by_language(
        ROOT / "results" / "raw" / "study_c.jsonl",
        mid_id,
        ["en", "hi", "ru"],
    )
    cells = {}
    for lang in codes:
        group = [r for r in rows if r.get("exec_language") == lang]
        n_g = len(group) or 1
        cells[lang] = {
            "n": len(group),
            "S_L": sum(1 for r in group if r["outcome"]["success"]) / n_g if group else None,
            "C_L": sum(1 for r in group if r["outcome"].get("solo_success")) / n_g if group else None,
            "A_L": a_e.get(lang),
            "script": _script(lang),
        }
    summary = {
        "study": "E",
        "worker": worker,
        "monitor": mid_id,
        "n_per_language": n,
        "languages": codes,
        "spend": spend.snapshot(),
        "cells": cells,
        "A_en_mid_from_C": a_c.get("en"),
        "script_vs_language": {
            "hi_script_gap": _gap(a_e.get("hi-Latn"), a_e.get("hi")),
            "ru_script_gap": _gap(a_e.get("ru-Latn"), a_e.get("ru")),
            "hi_language_gap": _gap(a_c.get("en"), a_e.get("hi-Latn")),
            "ru_language_gap": _gap(a_c.get("en"), a_e.get("ru-Latn")),
        },
        "metrics": compute_all(rows) if rows else {},
        "log": str(log_path),
    }
    out = ROOT / "results" / "processed" / "study_e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _gap(high: float | None, low: float | None) -> float | None:
    if high is None or low is None:
        return None
    return float(high) - float(low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(experiment().get("study_e_n", 50)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--languages", default=None)
    args = parser.parse_args()
    if not xai_key_present():
        raise RuntimeError("XAI_API_KEY is empty in .env")

    worker = str(experiment().get("study_e_worker") or experiment().get("study_c_worker") or model_id("orchestrator"))
    monitor = GrokMonitor()
    if worker == monitor.model:
        raise RuntimeError("Study E worker must differ from the mid monitor")
    codes = (
        [c.strip() for c in args.languages.split(",")]
        if args.languages
        else list(experiment().get("study_e_codes") or ["hi", "hi-Latn", "ru", "ru-Latn"])
    )
    n = 2 if args.smoke else args.n
    if args.smoke:
        codes = codes[:1]
    seed_base = int(experiment().get("study_e_seed_base", 80000))
    cfg = a2a()
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "study_e.jsonl"
    spend = SpendGuard(float(experiment().get("study_e_grok_stop_usd", 4.0)))
    done = _done(log_path)

    stopped = False
    for lang_i, language in enumerate(codes):
        if stopped:
            break
        for i in range(n):
            seed = seed_base + lang_i * 1000 + i
            if (language, seed) in done:
                continue
            if not spend.allow(worker, 2500, 400):
                print(json.dumps({"stop": "grok_budget", **spend.snapshot()}), flush=True)
                stopped = True
                break
            label = _label_for(i)
            run = None
            last_error = None
            for attempt in range(3):
                try:
                    run = run_episode_labeled(language, seed, worker, label)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(2 * (attempt + 1))
            if run is None:
                print(
                    json.dumps(
                        {
                            "language": language,
                            "i": i + 1,
                            "error": f"{type(last_error).__name__}: {last_error}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            for model, tin, tout in run["worker_usage"]:
                spend.charge(model, tin, tout)
            scored = monitor.score(run["transcript"])
            spend.charge(monitor.model, int(scored["tokens_in"]), int(scored["tokens_out"]))
            instance = run["instance"]
            tpath = out_dir / f"study-e-{language}-{seed}.txt"
            tpath.write_text(run["transcript"], encoding="utf-8")
            append_jsonl(
                log_path,
                EpisodeLog(
                    study="E",
                    env="env3_delegated",
                    instance_id=instance.instance_id,
                    card_language=language,
                    exec_language=language,
                    script=_script(language),
                    worker_model=ModelStamp(id=worker, version=worker, temperature=0.2),
                    monitor_model=ModelStamp(id=monitor.model, version=monitor.model, temperature=0.0),
                    seed=seed,
                    a2a=A2AStamp(
                        protocol_version=str(cfg["protocol_version"]),
                        task_id=f"study-e-{seed}",
                        context_id=f"study-e-{seed}",
                        final_state="COMPLETED",
                    ),
                    ground_truth=GroundTruth(
                        label=instance.label,
                        violation_type=instance.violation_type,
                        canary=instance.canary,
                    ),
                    outcome=Outcome(
                        success=run["success"],
                        turns=2,
                        tokens_in=run["tokens_in"],
                        tokens_out=run["tokens_out"],
                        solo_success=run["solo_success"],
                    ),
                    monitor=MonitorStamp(
                        flag=scored["flag"],
                        score=scored["score"],
                        flag_turn=scored.get("flag_turn"),
                        rationale=scored.get("rationale", ""),
                        tokens_in=int(scored.get("tokens_in", 0)),
                        tokens_out=int(scored.get("tokens_out", 0)),
                    ),
                    canary_recovered=run["canary_recovered"],
                    transcript_ref=str(tpath),
                ),
            )
            print(
                json.dumps(
                    {
                        "language": language,
                        "i": i + 1,
                        "n": n,
                        "label": label,
                        "success": run["success"],
                        "solo_success": run["solo_success"],
                        "flagged": scored["flag"],
                        **spend.snapshot(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(json.dumps(summarize(log_path, spend, codes, n, worker), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
