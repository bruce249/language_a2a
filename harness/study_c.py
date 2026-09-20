from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from agents.monitor import GrokMonitor, LunaMonitor
from agents.worker import GrokAnalystPolicy, GrokHolderPolicy, GrokSoloPolicy, localized_client_message
from analysis.metrics import compute_all
from envs.env3_delegated import make_instance
from envs.generators import canary_recovered, parse_total
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
from harness.secrets import openai_key_present, xai_key_present


def _script(code: str) -> str:
    return str(languages()["languages"][code]["script"])


def _label_for(index: int) -> str:
    return "violating" if index % 10 in {0, 3, 6} else "benign"


def _usd(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = pricing()["per_million"].get(model)
    if not rates:
        raise KeyError(f"No price for {model}")
    return (tokens_in * float(rates["input"]) + tokens_out * float(rates["output"])) / 1_000_000


def _provider(model: str) -> str:
    return "openai" if model.startswith("openai/") else "grok"


class SpendGuard:
    def __init__(self) -> None:
        caps = pricing()["caps"]
        self.openai_stop = float(caps["openai_stop_usd"])
        self.grok_stop = float(caps["grok_stop_usd"])
        self.openai = 0.0
        self.grok = 0.0

    def charge(self, model: str, tokens_in: int, tokens_out: int) -> float:
        cost = _usd(model, tokens_in, tokens_out)
        if _provider(model) == "openai":
            self.openai += cost
        else:
            self.grok += cost
        return cost

    def allow(self, model: str, est_in: int, est_out: int) -> bool:
        cost = _usd(model, est_in, est_out)
        if _provider(model) == "openai":
            return self.openai + cost <= self.openai_stop
        return self.grok + cost <= self.grok_stop

    def snapshot(self) -> dict[str, float]:
        return {
            "openai_usd": round(self.openai, 4),
            "grok_usd": round(self.grok, 4),
            "openai_cap": self.openai_stop,
            "grok_cap": self.grok_stop,
        }


def _done_keys(path: Path, monitor_id: str) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if row.get("monitor_model", {}).get("id") != monitor_id:
            continue
        keys.add((row["exec_language"], int(row["seed"])))
    return keys


def _write_episode(
    path: Path,
    *,
    instance,
    language: str,
    seed: int,
    worker: str,
    monitor_id: str,
    monitor: dict,
    success: bool,
    solo_success: bool | None,
    tokens_in: int,
    tokens_out: int,
    recovered: bool,
    transcript_path: Path,
) -> None:
    append_jsonl(
        path,
        EpisodeLog(
            study="C",
            env="env3_delegated",
            instance_id=instance.instance_id,
            card_language=language,
            exec_language=language,
            script=_script(language),
            worker_model=ModelStamp(id=worker, version=worker, temperature=0.2),
            monitor_model=ModelStamp(id=monitor_id, version=monitor_id, temperature=0.0),
            seed=seed,
            a2a=A2AStamp(
                protocol_version=str(a2a()["protocol_version"]),
                task_id=f"study-c-{seed}",
                context_id=f"study-c-{seed}",
                final_state="COMPLETED",
            ),
            ground_truth=GroundTruth(
                label=instance.label,
                violation_type=instance.violation_type,
                canary=instance.canary,
            ),
            outcome=Outcome(
                success=success,
                turns=2,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                solo_success=solo_success,
            ),
            monitor=MonitorStamp(
                flag=monitor["flag"],
                score=monitor["score"],
                flag_turn=monitor.get("flag_turn"),
                rationale=monitor.get("rationale", ""),
                tokens_in=int(monitor.get("tokens_in", 0)),
                tokens_out=int(monitor.get("tokens_out", 0)),
            ),
            canary_recovered=recovered,
            transcript_ref=str(transcript_path),
        ),
    )


def run_episode_labeled(language: str, seed: int, worker: str, label: str) -> dict:
    instance = make_instance(seed, language=language, label=label)  # type: ignore[arg-type]
    client_msg = localized_client_message(instance)
    if instance.canary in client_msg:
        raise RuntimeError("Canary leaked into the client message")
    solo = GrokSoloPolicy(instance, worker).respond()
    holder = GrokHolderPolicy(instance, worker).respond(client_msg)
    analyst = GrokAnalystPolicy(language, worker).respond(holder["text"])
    transcript = "\n".join([client_msg, holder["text"], analyst["text"]])
    predicted = parse_total(analyst["text"])
    solo_pred = parse_total(solo["text"])
    return {
        "instance": instance,
        "transcript": transcript,
        "success": predicted == instance.gold_total_cents,
        "solo_success": solo_pred == instance.gold_total_cents,
        "canary_recovered": canary_recovered(transcript, instance.canary),
        "tokens_in": solo["tokens_in"] + holder["tokens_in"] + analyst["tokens_in"],
        "tokens_out": solo["tokens_out"] + holder["tokens_out"] + analyst["tokens_out"],
        "worker_usage": [
            (worker, solo["tokens_in"], solo["tokens_out"]),
            (worker, holder["tokens_in"], holder["tokens_out"]),
            (worker, analyst["tokens_in"], analyst["tokens_out"]),
        ],
    }


def _summarize(log_path: Path, spend: SpendGuard, codes: list[str], n: int, worker: str) -> dict:
    rows = read_jsonl(log_path) if log_path.exists() else []
    luna_id = model_id("monitor")
    luna_rows = [r for r in rows if r.get("monitor_model", {}).get("id") == luna_id]
    metrics = compute_all(luna_rows) if luna_rows else {}
    auroc_by = {}
    for lang in codes:
        group = [r for r in luna_rows if r.get("exec_language") == lang]
        y = [1 if r["ground_truth"]["label"] == "violating" else 0 for r in group]
        s = [float(r["monitor"]["score"]) for r in group]
        if group and len(set(y)) > 1:
            auroc_by[lang] = float(roc_auc_score(y, s))
        else:
            auroc_by[lang] = None
    summary = {
        "worker": worker,
        "n_per_language_target": n,
        "languages": codes,
        "episodes_logged": len(luna_rows),
        "spend": spend.snapshot(),
        "AUROC_luna": auroc_by,
        "metrics": metrics,
        "log": str(log_path),
    }
    out = ROOT / "results" / "processed" / "study_c.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(experiment().get("study_c_n", 200)))
    parser.add_argument("--languages", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", default=None)
    args = parser.parse_args()
    if not xai_key_present() or not openai_key_present():
        raise RuntimeError("Need both XAI_API_KEY and OPENAI_API_KEY in .env")

    worker = args.worker or str(experiment().get("study_c_worker") or model_id("worker"))
    luna = LunaMonitor()
    mid = GrokMonitor()
    codes = (
        [c.strip() for c in args.languages.split(",")]
        if args.languages
        else list(languages()["collab_codes"])
    )
    n = 2 if args.smoke else args.n
    seed_base = int(experiment().get("study_c_seed_base", 40000))
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "study_c.jsonl"
    spend = SpendGuard()
    done_luna = _done_keys(log_path, luna.model)
    done_mid = _done_keys(log_path, mid.model)

    if args.smoke:
        codes = ["en"]

    for lang_i, language in enumerate(codes):
        for i in range(n):
            seed = seed_base + lang_i * 1000 + i
            label = _label_for(i)
            key = (language, seed)
            if key in done_luna and key in done_mid:
                continue
            if not spend.allow(worker, 1800, 400):
                print(json.dumps({"stop": "grok_budget", **spend.snapshot()}), flush=True)
                print(json.dumps(_summarize(log_path, spend, codes, n, worker), indent=2))
                return
            if not spend.allow(luna.model, 1600, 250):
                print(json.dumps({"stop": "openai_budget", **spend.snapshot()}), flush=True)
                print(json.dumps(_summarize(log_path, spend, codes, n, worker), indent=2))
                return
            try:
                run = run_episode_labeled(language, seed, worker, label)
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {
                            "language": language,
                            "i": i + 1,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ),
                    flush=True,
                )
                continue
            for model, tin, tout in run["worker_usage"]:
                spend.charge(model, tin, tout)

            instance = run["instance"]
            tag = "study-c-smoke" if args.smoke else f"study-c-{language}"
            transcript_path = out_dir / f"{tag}-{instance.instance_id[:10]}.txt"
            transcript_path.write_text(run["transcript"], encoding="utf-8")

            if key not in done_luna:
                luna_scored = luna.score(run["transcript"])
                spend.charge(luna.model, int(luna_scored["tokens_in"]), int(luna_scored["tokens_out"]))
                _write_episode(
                    log_path,
                    instance=instance,
                    language=language,
                    seed=seed,
                    worker=worker,
                    monitor_id=luna.model,
                    monitor=luna_scored,
                    success=run["success"],
                    solo_success=run["solo_success"],
                    tokens_in=run["tokens_in"],
                    tokens_out=run["tokens_out"],
                    recovered=run["canary_recovered"],
                    transcript_path=transcript_path,
                )
            else:
                luna_scored = {"flag": None, "score": None}

            if key not in done_mid and spend.allow(mid.model, 1600, 250):
                mid_scored = mid.score(run["transcript"])
                spend.charge(mid.model, int(mid_scored["tokens_in"]), int(mid_scored["tokens_out"]))
                _write_episode(
                    log_path,
                    instance=instance,
                    language=language,
                    seed=seed,
                    worker=worker,
                    monitor_id=mid.model,
                    monitor=mid_scored,
                    success=run["success"],
                    solo_success=run["solo_success"],
                    tokens_in=0,
                    tokens_out=0,
                    recovered=run["canary_recovered"],
                    transcript_path=transcript_path,
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
                        "canary_recovered": run["canary_recovered"],
                        "luna_flag": luna_scored.get("flag"),
                        "luna_score": luna_scored.get("score"),
                        **spend.snapshot(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            spend_path = ROOT / "results" / "processed" / "study_c_spend.json"
            spend_path.write_text(json.dumps(spend.snapshot(), indent=2), encoding="utf-8")

    print(json.dumps(_summarize(log_path, spend, codes, n, worker), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
