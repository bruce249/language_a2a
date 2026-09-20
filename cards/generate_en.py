from pathlib import Path

from agents.card_builder import build_card_payload
from harness.config import a2a

OUT = Path(__file__).resolve().parents[1] / "cards" / "generated" / "en"
CFG = a2a()
HOSTS = CFG["hosts"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for role, spec in HOSTS.items():
        url = f"http://{spec['host']}:{spec['port']}"
        payload = build_card_payload(
            role=role,
            language="en",
            url=url,
            protocol_version=str(CFG["protocol_version"]),
            protocol_binding=str(CFG["protocol_binding"]),
        )
        path = OUT / f"{role}.json"
        import json

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
