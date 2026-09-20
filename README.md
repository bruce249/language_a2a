# Cross-lingual A2A oversight

Reproducible harness for *Language as an Oversight Surface in A2A Multi-Agent Systems*. Spec: `cross-lingual-a2a-oversight-spec.md`.

**Current models (keys in `.env`, never in git):**

| Role | Model |
|---|---|
| Worker / orchestrator | `grok/grok-4.6` (also 4.5, 4.3) |
| Weak monitor | `openai/gpt-5.6-luna` |
| Mid monitor | `grok/grok-4.3` |

Qwen is deferred. Deterministic ENV-3 and the A2A hello-world do not call any API.

## Setup

```powershell
copy .env.example .env
# fill OPENAI_API_KEY and XAI_API_KEY when you reach LLM monitor calibration
python -m pip install -e ".[dev]"
python -m pytest
python -m harness.runner --mode hello --seed 1
python -m analysis.metrics --input tests/fixtures/synthetic_episodes.jsonl
```

## Build order

1. ENV-3 generators + uniqueness tests (this repo)
2. English A2A two-agent hello-world
3. Metrics from JSONL fixtures
4. Luna monitor on English ENV-3 (gate: AUROC > 0.9)
5. hi/ru cards, then Study A

A2A protocol pinned to **v1.0**. Cards are served at `/.well-known/agent-card.json`.
