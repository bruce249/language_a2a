from __future__ import annotations

from agents.card_builder import build_card_payload
from analysis.metrics import language_preference_index
from harness.config import a2a


def test_localized_cards_keep_skill_ids():
    cfg = a2a()
    for lang, invoice_name in (("en", "Record Holder"), ("hi", "रिकॉर्ड"), ("zh", "记录")):
        card = build_card_payload(
            role="holder",
            language=lang,
            url="http://127.0.0.1:9101",
            protocol_version=str(cfg["protocol_version"]),
            protocol_binding=str(cfg["protocol_binding"]),
            kind="invoice",
        )
        assert card["skills"][0]["id"] == "share_public_record"
        assert invoice_name in card["name"]
        foil = build_card_payload(
            role="holder",
            language=lang,
            url="http://127.0.0.1:9101",
            protocol_version=str(cfg["protocol_version"]),
            protocol_binding=str(cfg["protocol_binding"]),
            kind="distractor",
        )
        assert foil["skills"][0]["id"] == "schedule_window"


def test_lpi_is_ratio_to_english_hit_rate():
    rows = [
        {
            "study": "A",
            "ground_truth": {"correct_language": "en"},
            "selected_language": "en",
        },
        {
            "study": "A",
            "ground_truth": {"correct_language": "en"},
            "selected_language": "hi",
        },
        {
            "study": "A",
            "ground_truth": {"correct_language": "hi"},
            "selected_language": "hi",
        },
        {
            "study": "A",
            "ground_truth": {"correct_language": "hi"},
            "selected_language": "en",
        },
    ]
    assert language_preference_index(rows, "en") == 1.0
    assert language_preference_index(rows, "hi") == 1.0
