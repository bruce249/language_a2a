from __future__ import annotations

from harness.config import experiment, model_id, model_record, models


def resolve_episode_models(worker_id: str | None = None, monitor_id: str | None = None) -> dict:
    worker = worker_id or model_id("worker")
    monitor = monitor_id or model_id("monitor")
    if models()["constraints"].get("workers_must_differ_from_monitor") and worker == monitor:
        raise ValueError(f"Worker and monitor must differ; both are {worker}")
    worker_rec = model_record(worker)
    monitor_rec = model_record(monitor)
    return {
        "worker_model": {
            "id": worker,
            "version": worker_rec.get("inspect_id", worker),
            "temperature": worker_rec.get(
                "temperature", experiment()["decode"]["worker_temperature"]
            ),
        },
        "monitor_model": {
            "id": monitor,
            "version": monitor_rec.get("inspect_id", monitor),
            "temperature": monitor_rec.get(
                "temperature", experiment()["decode"]["monitor_temperature"]
            ),
        },
    }
