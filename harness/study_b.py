from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.worker import (
    GrokEnv1AlphaPolicy,
    GrokEnv1BetaPolicy,
    GrokEnv1SoloPolicy,
    localized_env1_client,
)
from analysis.metrics import collaboration_efficiency, compute_all
from envs.env1_split import generate_env1_instance, parse_slot
from harness.config import ROOT, a2a, experiment, languages, model_id, pricing
from harness.logging import (
    A2AStamp,
    EpisodeLog,
    GroundTruth,
    ModelStamp,
    MonitorStamp,
    Outcome,
    read_jsonl,
)
from harness.prompts import script_fraction
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


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {
        (r["exec_language"], int(r["seed"]))
        for r in read_jsonl(path)
        if r.get("study") == "B" and r.get("env") == "env1_split"
    }


def run_episode(language: str, seed: int, model: str) -> dict:
    instance = generate_env1_instance(seed, language=language)
    client_msg = localized_env1_client(instance)
    solo = GrokEnv1SoloPolicy(instance, model).respond()
    alpha = GrokEnv1AlphaPolicy(instance, model).respond(client_msg)
    beta = GrokEnv1BetaPolicy(instance, model).respond(alpha["text"])
    transcript = "\n".join([client_msg, alpha["text"], beta["text"]])
    joint_pred = parse_slot(beta["text"])
    solo_pred = parse_slot(solo["text"])
    gold = instance.gold_slot
    script = _script(language)
    return {
        "instance": instance,
        "transcript": transcript,
        "success": joint_pred == gold,
        "solo_success": solo_pred == gold,
        "parse_ok": joint_pred is not None,
        "solo_parse_ok": solo_pred is not None,
        "joint_pred": joint_pred,
        "solo_pred": solo_pred,
        "gold": gold,
        "constraint_count": instance.constraint_count,
        "script_adherence": (
            script_fraction(alpha["text"], script) + script_fraction(beta["text"], script)
        )
        / 2,
        "tokens_in": solo["tokens_in"] + alpha["tokens_in"] + beta["tokens_in"],
        "tokens_out": solo["tokens_out"] + alpha["tokens_out"] + beta["tokens_out"],
        "usage": [
            (model, solo["tokens_in"], solo["tokens_out"]),
            (model, alpha["tokens_in"], alpha["tokens_out"]),
            (model, beta["tokens_in"], beta["tokens_out"]),
        ],
    }


def summarize(log_path: Path, spend: SpendGuard, codes: list[str], n: int, model: str) -> dict:
    rows = [r for r in read_jsonl(log_path) if r.get("study") == "B"] if log_path.exists() else []
    by_lang: dict[str, list[dict]] = {}
    for row in rows:
        by_lang.setdefault(row["exec_language"], []).append(row)
    cells = {}
    for lang, group in by_lang.items():
        n_g = len(group)
        s_l = sum(1 for r in group if r["outcome"]["success"]) / n_g
        c_l = sum(1 for r in group if r["outcome"].get("solo_success")) / n_g
        tokens = [r["outcome"]["tokens_in"] + r["outcome"]["tokens_out"] for r in group]
        bits = [int(r.get("constraint_count") or 1) for r in group]
        cells[lang] = {
            "n": n_g,
            "S_L": s_l,
            "C_L": c_l,
            "CER_L": collaboration_efficiency(s_l, c_l),
            "protocol_failure_rate": sum(
                1 for r in group if r.get("a2a", {}).get("final_state") == "FAILED"
            )
            / n_g,
            "mean_tokens": sum(tokens) / n_g,
            "tokens_per_constraint": sum(t / b for t, b in zip(tokens, bits, strict=True)) / n_g,
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
    out = ROOT / "results" / "processed" / "study_b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(experiment().get("study_b_n", 50)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--languages", default=None)
    args = parser.parse_args()
    if not xai_key_present():
        raise RuntimeError("XAI_API_KEY is empty in .env")

    model = model_id("worker")
    codes = (
        [c.strip() for c in args.languages.split(",")]
        if args.languages
        else list(languages()["collab_codes"])
    )
    n = 2 if args.smoke else args.n
    if args.smoke:
        codes = ["en"]
    seed_base = int(experiment().get("study_b_seed_base", 60000))
    cfg = a2a()
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "study_b.jsonl"
    spend = SpendGuard(float(experiment().get("study_b_grok_stop_usd", 9.0)))
    done = _done(log_path)

    stopped = False
    for lang_i, language in enumerate(codes):
        if stopped:
            break
        for i in range(n):
            seed = seed_base + lang_i * 1000 + i
            if (language, seed) in done:
                continue
            if not spend.allow(model, 2500, 400):
                print(json.dumps({"stop": "grok_budget", **spend.snapshot()}), flush=True)
                stopped = True
                break
            run = None
            last_error = None
            for attempt in range(3):
                try:
                    run = run_episode(language, seed, model)
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
            for call_model, tin, tout in run["usage"]:
                spend.charge(call_model, tin, tout)
            instance = run["instance"]
            tpath = out_dir / f"study-b-{language}-{seed}.txt"
            tpath.write_text(run["transcript"], encoding="utf-8")
            row = EpisodeLog(
                study="B",
                env="env1_split",
                instance_id=instance.instance_id,
                card_language=language,
                exec_language=language,
                script=_script(language),
                worker_model=ModelStamp(id=model, version=model, temperature=0.2),
                monitor_model=ModelStamp(id="none", version="none", temperature=0.0),
                seed=seed,
                a2a=A2AStamp(
                    protocol_version=str(cfg["protocol_version"]),
                    task_id=f"study-b-{seed}",
                    context_id=f"study-b-{seed}",
                    final_state="COMPLETED" if run["parse_ok"] else "FAILED",
                ),
                ground_truth=GroundTruth(label="benign", violation_type=None, canary=None),
                outcome=Outcome(
                    success=run["success"],
                    turns=2,
                    tokens_in=run["tokens_in"],
                    tokens_out=run["tokens_out"],
                    solo_success=run["solo_success"],
                ),
                monitor=MonitorStamp(flag=False, score=0.0, rationale="study_b_no_monitor"),
                canary_recovered=False,
                transcript_ref=str(tpath),
            )
            payload = json.loads(row.to_json())
            payload["constraint_count"] = run["constraint_count"]
            payload["joint_pred"] = run["joint_pred"]
            payload["solo_pred"] = run["solo_pred"]
            payload["gold_slot"] = run["gold"]
            payload["script_adherence"] = run["script_adherence"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "language": language,
                        "i": i + 1,
                        "n": n,
                        "success": run["success"],
                        "solo_success": run["solo_success"],
                        "gold": run["gold"],
                        "joint_pred": run["joint_pred"],
                        "solo_pred": run["solo_pred"],
                        **spend.snapshot(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(json.dumps(summarize(log_path, spend, codes, n, model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
