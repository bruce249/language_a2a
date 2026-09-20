from __future__ import annotations

from analysis.metrics import compute_all
from agents.card_builder import build_card_payload
from harness.config import a2a, model_id
from harness.models import resolve_episode_models
from tests.make_synthetic import build_fixture_rows


def test_v1_card_fields_and_never_localise_ids():
    cfg = a2a()
    payload = build_card_payload(
        role="holder",
        language="en",
        url="http://127.0.0.1:9101",
        protocol_version=str(cfg["protocol_version"]),
        protocol_binding=str(cfg["protocol_binding"]),
    )
    assert "supported_interfaces" in payload
    assert "url" not in payload
    assert "protocol_version" not in payload
    iface = payload["supported_interfaces"][0]
    assert iface["protocol_version"] == "1.0"
    assert iface["protocol_binding"] == "JSONRPC"
    assert payload["skills"][0]["id"] == "share_public_record"


def test_worker_is_grok_monitor_is_luna():
    models = resolve_episode_models()
    assert models["worker_model"]["id"].startswith("grok/")
    assert models["monitor_model"]["id"] == "openai/gpt-5.6-luna"
    assert models["worker_model"]["id"] != models["monitor_model"]["id"]
    assert model_id("orchestrator").startswith("grok/")


def test_metrics_from_synthetic_fixtures_are_deterministic():
    rows = [episode.model_dump() for episode in build_fixture_rows()]
    first = compute_all(rows)
    second = compute_all(rows)
    assert first == second
    assert first["n_episodes"] == len(rows)
    assert "hi" in first["CCA"]
    assert first["CCA"]["hi"]["CCA_L"] > 0
    assert any(cell["AUROC"] is not None for cell in first["cells"].values())
