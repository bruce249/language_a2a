# Language as an Oversight Surface in Agent-to-Agent Systems

Harness, episode logs and analysis code for the paper *Language as an Oversight Surface in
Agent-to-Agent Systems*, a study of whether the natural language an A2A transcript is written
in changes how reliably a monitor catches a leak in it.

The A2A protocol is pinned to **v1.0**; cards are served at `/.well-known/agent-card.json`.

## What is here

| Path | Contents |
|---|---|
| `harness/` | Study runners (A–E), calibration, model routing, logging |
| `envs/` | ENV-1 (split-knowledge scheduling) and ENV-3 (invoice + canary) generators |
| `agents/` | Holder, analyst, orchestrator, monitors, card builder |
| `cards/` | Agent Card templates and the localized card packs |
| `analysis/` | Metrics, interval estimates and hypothesis tests, paper figures |
| `results/raw/*.jsonl` | The episode logs every number in the paper is computed from |
| `results/processed/` | Derived summaries (regenerated, not authoritative) |
| `paper/figures/` | The figures as published |
| `PREREGISTRATION.md` | The study plan, including the one confirmatory comparison |
| `docs/` | Threat model and the related-work / novelty-search log |

## Reproducing the paper's numbers

No API key is needed for this: it reads the committed logs.

```bash
python -m pip install -e ".[dev]"

# Every statistic quoted in the paper, from results/raw/*.jsonl
python -m analysis.stats            # -> results/processed/revision_stats.json

# Every figure in the paper, from the same logs
python -m analysis.figures_paper    # -> paper/figures/*.png
```

Neither script reads a hand-entered number. `analysis/stats.py` computes the confirmatory
comparison, Wilson and bootstrap intervals, Fisher tests with Holm adjustment, the paired
McNemar test for Study B (reported both as analysed and under the plan's exclusion rule), and
the Study D transition model.

To rebuild the PDF from the Markdown source, `python -m paper.build_pdf`.

## Re-running the experiments

This does call the APIs and costs money. Keys live in `.env`, never in git.

```bash
cp .env.example .env    # fill OPENAI_API_KEY and XAI_API_KEY
python -m pytest
python -m harness.runner --mode hello --seed 1   # deterministic, no API calls

python -m harness.calibrate     # Luna calibration gate on English ENV-3
python -m harness.study_a       # card routing        seeds 50000+
python -m harness.study_b       # ENV-1 collaboration seeds 60000+
python -m harness.study_c       # ENV-3, both monitors seeds 40000+
python -m harness.study_d       # language selection  seeds 70000+
python -m harness.study_e       # script control      seeds 80000+
```

| Role | Model | Settings |
|---|---|---|
| Worker / picker / router | `grok/grok-4.6` | temp 0.2 (0.0 for routing), `reasoning_effort=low` |
| Worker (Study B) | `grok/grok-4.3` | temp 0.2, `reasoning_effort=none` |
| Weak monitor | `openai/gpt-5.6-luna` | temp 0.0 |
| Mid monitor | `grok/grok-4.3` | temp 0.0, never paired with 4.3 workers |

Total logged spend for the published run was USD 0.19 on OpenAI and USD 14.25 on Grok.

## Caveats worth reading before reusing this

- The monitors were asked for a probability in `[0, 1]` and did not use the range. The mid
  monitor returned two distinct values, so AUROC and recall at a fixed FPR are not meaningful
  for it; the paper reports TPR, FPR and balanced accuracy instead.
- Study D is a single adaptive run of 200 trials. Consecutive transitions are dependent, so
  its intervals are optimistic, and it has no null-condition arm.
- Environment instances are generated fresh per episode and are never shared across language
  arms, so all cross-language contrasts are unpaired.
- All canaries are synthetic. There is no real personal data anywhere in the logs.
