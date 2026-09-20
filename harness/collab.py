from __future__ import annotations

import argparse
import json
from collections import defaultdict

from agents.monitor import KeywordMonitor
from agents.worker import GrokAnalystPolicy, GrokHolderPolicy, localized_client_message
from envs.env3_delegated import make_instance
from envs.generators import canary_recovered, parse_total
from harness.config import ROOT, a2a, experiment, languages, model_id
from harness.logging import (
    A2AStamp,
    EpisodeLog,
    GroundTruth,
    ModelStamp,
    MonitorStamp,
    Outcome,
    write_jsonl,
)
from harness.prompts import script_fraction
from harness.secrets import xai_key_present


def _script(code: str) -> str:
    return str(languages()["languages"][code]["script"])


def run_episode(language: str, seed: int, model: str) -> dict:
    instance = make_instance(seed, language=language, label="benign")
    client_msg = localized_client_message(instance)
    if instance.canary in client_msg:
        raise RuntimeError("Canary leaked into the client message")
    holder = GrokHolderPolicy(instance, model).respond(client_msg)
    analyst = GrokAnalystPolicy(language, model).respond(holder["text"])
    transcript = "\n".join([client_msg, holder["text"], analyst["text"]])
    predicted = parse_total(analyst["text"])
    success = predicted == instance.gold_total_cents
    holder_sku_count = sum(1 for item in instance.line_items if item.sku in holder["text"])
    expected_sku_count = len(instance.line_items)
    script = _script(language)
    holder_script = script_fraction(holder["text"], script)
    analyst_script = script_fraction(analyst["text"], script)
    return {
        "instance": instance,
        "client_msg": client_msg,
        "holder_text": holder["text"],
        "analyst_text": analyst["text"],
        "transcript": transcript,
        "predicted_total": predicted,
        "success": success,
        "parse_ok": predicted is not None,
        "canary_recovered": canary_recovered(transcript, instance.canary),
        "holder_script": holder_script,
        "analyst_script": analyst_script,
        "script_adherence": (holder_script + analyst_script) / 2,
        "holder_sku_count": holder_sku_count,
        "expected_sku_count": expected_sku_count,
        "line_items_copied": holder_sku_count == expected_sku_count,
        "tokens_in": holder["tokens_in"] + analyst["tokens_in"],
        "tokens_out": holder["tokens_out"] + analyst["tokens_out"],
        "reasoning_tokens": holder["reasoning_tokens"] + analyst["reasoning_tokens"],
        "turns": 2,
    }


def summarize(rows: list[dict]) -> dict:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[row["language"]].append(row)
    cells = {}
    for lang, group in by_lang.items():
        n = len(group)
        cells[lang] = {
            "n": n,
            "success_rate": sum(r["success"] for r in group) / n,
            "parse_rate": sum(r["parse_ok"] for r in group) / n,
            "script_adherence": sum(r["script_adherence"] for r in group) / n,
            "accidental_leak_rate": sum(r["canary_recovered"] for r in group) / n,
            "line_item_copy_rate": sum(r.get("line_items_copied", False) for r in group) / n,
            "mean_tokens_in": sum(r["tokens_in"] for r in group) / n,
            "mean_tokens_out": sum(r["tokens_out"] for r in group) / n,
            "mean_reasoning_tokens": sum(r["reasoning_tokens"] for r in group) / n,
            "protocol_failure_rate": sum(1 for r in group if not r["parse_ok"]) / n,
        }
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(experiment().get("collab_n", 10)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--languages", default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--out", default="collab_languages")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if not xai_key_present():
        raise RuntimeError("XAI_API_KEY is empty in .env")
    model = model_id("worker")
    codes = (
        [c.strip() for c in args.languages.split(",")]
        if args.languages
        else list(languages()["collab_codes"])
    )
    n = 1 if args.smoke else args.n
    if args.smoke:
        codes = ["en"]
    episodes: list[EpisodeLog] = []
    raw_rows: list[dict] = []
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = 0
    for language in codes:
        for i in range(n):
            seq += 1
            if args.seed_base is not None:
                seed = args.seed_base + i
            else:
                seed = 20_000 + seq
            try:
                run = run_episode(language, seed, model)
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {
                            "language": language,
                            "i": i + 1,
                            "error": type(exc).__name__,
                            "success": False,
                        }
                    ),
                    flush=True,
                )
                raw_rows.append(
                    {
                        "language": language,
                        "success": False,
                        "parse_ok": False,
                        "script_adherence": 0.0,
                        "canary_recovered": False,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "reasoning_tokens": 0,
                    }
                )
                continue
            instance = run["instance"]
            print(
                json.dumps(
                    {
                        "language": language,
                        "i": i + 1,
                        "n": n,
                        "success": run["success"],
                        "parse_ok": run["parse_ok"],
                        "line_items_copied": run["line_items_copied"],
                        "holder_sku_count": run["holder_sku_count"],
                        "script_adherence": round(run["script_adherence"], 3),
                        "tokens_in": run["tokens_in"],
                        "tokens_out": run["tokens_out"],
                        "reasoning_tokens": run["reasoning_tokens"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            file_tag = args.tag or language
            transcript_path = out_dir / f"collab-{file_tag}-{instance.instance_id[:10]}.txt"
            transcript_path.write_text(run["transcript"], encoding="utf-8")
            monitor = KeywordMonitor().score(run["transcript"])
            episodes.append(
                EpisodeLog(
                    study="B",
                    env="env3_delegated",
                    instance_id=instance.instance_id,
                    card_language=language,
                    exec_language=language,
                    script=_script(language),
                    worker_model=ModelStamp(id=model, version=model, temperature=0.2),
                    monitor_model=ModelStamp(
                        id="keyword_leak", version="lexical", temperature=0.0
                    ),
                    seed=seed,
                    a2a=A2AStamp(
                        protocol_version=str(a2a()["protocol_version"]),
                        task_id=f"collab-{seed}",
                        context_id=f"collab-{seed}",
                        final_state="COMPLETED" if run["parse_ok"] else "FAILED",
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
                    ),
                    monitor=MonitorStamp(
                        flag=monitor["flag"],
                        score=monitor["score"],
                        rationale=monitor["rationale"],
                    ),
                    canary_recovered=run["canary_recovered"],
                    transcript_ref=str(transcript_path),
                )
            )
            raw_rows.append(
                {
                    "language": language,
                    "success": run["success"],
                    "parse_ok": run["parse_ok"],
                    "script_adherence": run["script_adherence"],
                    "canary_recovered": run["canary_recovered"],
                    "line_items_copied": run["line_items_copied"],
                    "holder_sku_count": run["holder_sku_count"],
                    "expected_sku_count": run["expected_sku_count"],
                    "tokens_in": run["tokens_in"],
                    "tokens_out": run["tokens_out"],
                    "reasoning_tokens": run["reasoning_tokens"],
                }
            )
    log_path = out_dir / f"{args.out}.jsonl"
    write_jsonl(log_path, episodes)
    cells = summarize(raw_rows)
    summary = {
        "model": model,
        "reasoning_effort": experiment().get("worker_reasoning_effort", "none"),
        "n_per_language": n,
        "languages": codes,
        "seed_base": args.seed_base,
        "cells": cells,
        "log": str(log_path),
    }
    summary_path = ROOT / "results" / "processed" / f"{args.out}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
