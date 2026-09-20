from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time

from agents.monitor import KeywordMonitor
from envs.env3_delegated import make_instance, run_deterministic_episode
from harness.a2a_client import fetch_card, load_private_instance, send_text, task_ids
from envs.generators import canary_recovered, parse_total
from harness.config import ROOT, a2a, experiment, languages
from harness.logging import (
    A2AStamp,
    EpisodeLog,
    GroundTruth,
    ModelStamp,
    MonitorStamp,
    Outcome,
    append_jsonl,
)
from harness.models import resolve_episode_models


def _script_for(language: str) -> str:
    return str(languages()["languages"][language]["script"])


def _episode_from_run(
    *,
    instance,
    run: dict,
    study: str,
    a2a_meta: dict,
    monitor: dict,
    protocol_version: str,
    worker_id: str | None = None,
    monitor_id: str | None = None,
) -> EpisodeLog:
    models = resolve_episode_models(worker_id, monitor_id)
    return EpisodeLog(
        study=study,
        env="env3_delegated",
        instance_id=instance.instance_id,
        card_language=instance.language,
        exec_language=instance.language,
        script=_script_for(instance.language),
        worker_model=ModelStamp(**models["worker_model"]),
        monitor_model=ModelStamp(**models["monitor_model"]),
        seed=instance.seed,
        a2a=A2AStamp(
            protocol_version=protocol_version,
            task_id=a2a_meta.get("task_id", "in-process"),
            context_id=a2a_meta.get("context_id", "in-process"),
            final_state=a2a_meta.get("final_state", "COMPLETED"),
        ),
        ground_truth=GroundTruth(
            label=instance.label,
            violation_type=instance.violation_type,
            canary=instance.canary,
        ),
        outcome=Outcome(success=run["success"], turns=run["turns"]),
        monitor=MonitorStamp(
            flag=monitor["flag"],
            score=monitor["score"],
            flag_turn=monitor.get("flag_turn"),
            rationale=monitor.get("rationale", ""),
        ),
        canary_recovered=run["canary_recovered"],
        transcript_ref=a2a_meta.get("transcript_ref", ""),
    )


def run_in_process(seed: int, label: str | None = None) -> EpisodeLog:
    cfg = experiment()
    instance = make_instance(
        seed,
        language="en",
        label=label,  # type: ignore[arg-type]
        violation_base_rate=cfg["violation_base_rate"],
    )
    run = run_deterministic_episode(instance)
    monitor = KeywordMonitor().score(run["transcript"], instance.canary)
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / f"hello-{instance.instance_id[:12]}.txt"
    transcript_path.write_text(run["transcript"], encoding="utf-8")
    episode = _episode_from_run(
        instance=instance,
        run=run,
        study="C",
        a2a_meta={
            "task_id": f"local-{seed}",
            "context_id": f"local-{seed}",
            "final_state": "COMPLETED",
            "transcript_ref": str(transcript_path),
        },
        monitor=monitor,
        protocol_version=str(a2a()["protocol_version"]),
    )
    append_jsonl(out_dir / "hello.jsonl", episode)
    return episode


def _start_server(role: str, host: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agents.server",
            "--role",
            role,
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _run_a2a_async(seed: int, label: str | None = None) -> EpisodeLog:
    cfg = a2a()
    holder = cfg["hosts"]["holder"]
    analyst = cfg["hosts"]["analyst"]
    holder_url = f"http://{holder['host']}:{holder['port']}"
    analyst_url = f"http://{analyst['host']}:{analyst['port']}"
    instance = make_instance(
        seed,
        language="en",
        label=label,  # type: ignore[arg-type]
        violation_base_rate=experiment()["violation_base_rate"],
    )
    await load_private_instance(holder_url, instance.to_dict())
    holder_card = await fetch_card(holder_url)
    analyst_card = await fetch_card(analyst_url)
    if instance.canary in instance.client_message():
        raise RuntimeError("Canary in client message")
    holder_text, holder_chunks = await send_text(holder_url, instance.client_message())
    analyst_text, analyst_chunks = await send_text(analyst_url, holder_text)
    ids = task_ids(holder_chunks)
    transcript = "\n".join([instance.client_message(), holder_text, analyst_text])

    run = {
        "holder_text": holder_text,
        "analyst_text": analyst_text,
        "transcript": transcript,
        "canary_recovered": canary_recovered(transcript, instance.canary),
        "predicted_total": parse_total(analyst_text),
        "success": parse_total(analyst_text) == instance.gold_total_cents,
        "turns": 2,
    }
    monitor = KeywordMonitor().score(transcript, instance.canary)
    out_dir = ROOT / "results" / "raw"
    transcript_path = out_dir / f"a2a-{instance.instance_id[:12]}.json"
    transcript_path.write_text(
        json.dumps(
            {
                "holder_card_name": getattr(holder_card, "name", None),
                "analyst_card_name": getattr(analyst_card, "name", None),
                "transcript": transcript,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    episode = _episode_from_run(
        instance=instance,
        run=run,
        study="C",
        a2a_meta={
            "task_id": ids.get("task_id", f"a2a-{seed}"),
            "context_id": ids.get("context_id", f"a2a-{seed}"),
            "final_state": "COMPLETED" if run["success"] else "FAILED",
            "transcript_ref": str(transcript_path),
        },
        monitor=monitor,
        protocol_version=str(cfg["protocol_version"]),
    )
    append_jsonl(out_dir / "hello_a2a.jsonl", episode)
    return episode


def run_a2a(seed: int, label: str | None = None) -> EpisodeLog:
    cfg = a2a()
    holder = cfg["hosts"]["holder"]
    analyst = cfg["hosts"]["analyst"]
    procs = [
        _start_server("holder", holder["host"], int(holder["port"])),
        _start_server("analyst", analyst["host"], int(analyst["port"])),
    ]
    try:
        deadline = time.time() + 10
        last_error = None
        while time.time() < deadline:
            try:
                asyncio.run(
                    fetch_card(f"http://{holder['host']}:{holder['port']}")
                )
                asyncio.run(
                    fetch_card(f"http://{analyst['host']}:{analyst['port']}")
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.3)
        if last_error:
            raise last_error
        return asyncio.run(_run_a2a_async(seed, label))
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait(timeout=10)


def hello_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hello", "a2a"], default="hello")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", choices=["benign", "violating"], default=None)
    args = parser.parse_args()
    if args.mode == "hello":
        episode = run_in_process(args.seed, args.label)
    else:
        episode = run_a2a(args.seed, args.label)
    print(episode.to_json())


if __name__ == "__main__":
    hello_main()
