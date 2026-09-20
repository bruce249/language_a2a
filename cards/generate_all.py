from __future__ import annotations

import json
from pathlib import Path

from agents.card_builder import build_card_payload
from harness.config import a2a, languages

OUT = Path(__file__).resolve().parents[1] / "cards" / "generated"
CFG = a2a()
HOSTS = CFG["hosts"]


def main() -> None:
    for lang in languages()["collab_codes"]:
        dest = OUT / lang
        dest.mkdir(parents=True, exist_ok=True)
        for kind in ("invoice", "distractor"):
            spec = HOSTS["holder"]
            payload = build_card_payload(
                role="holder",
                language=lang,
                url=f"http://{spec['host']}:{spec['port']}",
                protocol_version=str(CFG["protocol_version"]),
                protocol_binding=str(CFG["protocol_binding"]),
                kind=kind,  # type: ignore[arg-type]
            )
            path = dest / f"{kind}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
