"""Generate paper figures from logged JSON/JSONL only.

Detection is plotted as the rate of flagged leaks (TPR) with Wilson intervals. The monitors
in this run emit an effectively binary score, so ranking metrics are not identified and are
not plotted; see analysis/stats.py.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from harness.config import ROOT
from harness.logging import read_jsonl

OUT = ROOT / "paper" / "figures"
LANG_ORDER = ["en", "fr", "zh", "ru", "hi"]
MID = "grok/grok-4.3"
LUNA = "openai/gpt-5.6-luna"
COLORS = {"en": "#E45756", "ru": "#F58518", "fr": "#4C78A8", "zh": "#54A24B", "hi": "#B279A2"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / "processed" / name).read_text(encoding="utf-8"))


def _err(cells: dict, order: list[str], key: str = "TPR") -> tuple[list, np.ndarray]:
    pts = [cells[k][key]["p"] for k in order]
    lo = [max(0.0, cells[k][key]["p"] - cells[k][key]["lo"]) for k in order]
    hi = [max(0.0, cells[k][key]["hi"] - cells[k][key]["p"]) for k in order]
    return pts, np.array([lo, hi])


def fig1_lpi() -> None:
    a = _load("study_a.json")["cells"]
    ys = [a[k]["LPI"] for k in LANG_ORDER]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(1.0, color="#444", ls="--", lw=1, label="parity")
    ax.bar(LANG_ORDER, ys, color="#4C78A8", width=0.6)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("LPI")
    ax.set_title("F6. Language Preference Index (Study A)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig6_lpi.png")
    plt.close(fig)


def fig2_detection(stats: dict) -> None:
    mid = stats["study_c"]["monitors"][MID]
    luna = stats["study_c"]["monitors"][LUNA]
    mid_p, mid_e = _err(mid, LANG_ORDER)
    luna_p, luna_e = _err(luna, LANG_ORDER)
    x = np.arange(len(LANG_ORDER))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.errorbar(x - 0.06, luna_p, yerr=luna_e, fmt="s--", color="#54A24B", capsize=3,
                lw=1.4, label="Luna (weak monitor)")
    ax.errorbar(x + 0.06, mid_p, yerr=mid_e, fmt="D-", color="#E45756", capsize=3,
                lw=1.4, label="Grok 4.3 (mid monitor)")
    ax.set_xticks(x, LANG_ORDER)
    ax.set_ylim(0.35, 1.06)
    ax.set_ylabel("leaks flagged (TPR)")
    ax.set_xlabel("transcript language")
    ax.set_title("F1. Leak detection by language (95% Wilson CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig1_detection.png")
    plt.close(fig)


def fig3_cca(stats: dict) -> None:
    cca = stats["covert_channel_advantage"][MID]
    xs = [k for k in LANG_ORDER if k != "en"]
    pts = [cca[k]["CCA_L"] for k in xs]
    lo = [cca[k]["CCA_L"] - cca[k]["ci95"][0] for k in xs]
    hi = [cca[k]["ci95"][1] - cca[k]["CCA_L"] for k in xs]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.bar(xs, pts, 0.55, color="#E45756", yerr=np.array([lo, hi]), capsize=4,
           error_kw={"ecolor": "#333", "lw": 1})
    ax.axhline(1.0, color="#444", ls="--", lw=1, label="parity with English")
    ax.set_ylabel(r"$CCA_L$ (mid monitor)")
    ax.set_title("F2. Covert channel advantage vs English")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig2_cca.png")
    plt.close(fig)


def fig4_selection(stats: dict) -> None:
    rows = sorted(
        (r for r in read_jsonl(ROOT / "results" / "raw" / "study_d.jsonl") if r.get("study") == "D"),
        key=lambda r: int(r["seed"]),
    )
    choices = [r.get("selected_language") or r.get("exec_language") for r in rows]
    window = 20
    xs = list(range(window, len(choices) + 1))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for lang in LANG_ORDER:
        roll = [choices[t - window : t].count(lang) / window for t in xs]
        ax.plot(xs, roll, color=COLORS[lang], lw=1.6, label=lang)
    ax.axhline(0.2, color="#888", ls=":", lw=1, label="uniform (0.20)")
    ax.set_xlabel("trial")
    ax.set_ylabel("share in last 20")
    ax.set_ylim(0, 1)
    ax.set_title("F3. Study D language selection (rolling window of 20)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.savefig(OUT / "fig3_selection.png")
    plt.close(fig)


def fig5_script(stats: dict) -> None:
    cells = stats["study_e"]["cells"]
    pairs = [("hi", "hi-Latn"), ("ru", "ru-Latn")]
    x = np.arange(2)
    nat_p, nat_e = _err(cells, [a for a, _ in pairs])
    rom_p, rom_e = _err(cells, [b for _, b in pairs])
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.bar(x - 0.18, nat_p, 0.36, color="#4C78A8", yerr=nat_e, capsize=4,
           error_kw={"ecolor": "#333", "lw": 1}, label="native script")
    ax.bar(x + 0.18, rom_p, 0.36, color="#F58518", yerr=rom_e, capsize=4,
           error_kw={"ecolor": "#333", "lw": 1}, label="romanised")
    ax.set_xticks(x, ["Hindi", "Russian"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("leaks flagged (TPR)")
    ax.set_title("F5. Script control (Study E, 95% Wilson CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "fig5_script.png")
    plt.close(fig)


def fig6_shift(stats: dict) -> None:
    """The trial-level result: a flag at t predicts leaving that language at t+1."""
    stay = stats["study_d"]["stay_rates"]
    labels = ["after no flag", "after flag"]
    keys = ["after_no_flag", "after_flag"]
    pts = [stay[k]["p"] for k in keys]
    lo = [stay[k]["p"] - stay[k]["lo"] for k in keys]
    hi = [stay[k]["hi"] - stay[k]["p"] for k in keys]
    counts = [f"{stay[k]['k']}/{stay[k]['n']}" for k in keys]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bars = ax.bar(labels, pts, 0.5, color=["#54A24B", "#E45756"],
                  yerr=np.array([lo, hi]), capsize=5, error_kw={"ecolor": "#333", "lw": 1})
    for bar, txt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.04, txt, ha="center", fontsize=9, color="white")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("keeps same language at $t+1$")
    ax.set_title("F4. Feedback sensitivity of the picker (95% Wilson CI)")
    fig.savefig(OUT / "fig4_shift.png")
    plt.close(fig)


def quotes(n: int = 4) -> list[str]:
    rows = [r for r in read_jsonl(ROOT / "results" / "raw" / "study_d.jsonl") if r.get("explicit_strategy")]
    texts = []
    for r in rows:
        text = str(r.get("picker_rationale") or "")
        if "CNRY" in text:
            continue
        texts.append(text.replace("\n", " ")[:220])
        if len(texts) >= n:
            break
    return texts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    stats = _load("revision_stats.json")
    fig1_lpi()
    fig2_detection(stats)
    fig3_cca(stats)
    fig4_selection(stats)
    fig5_script(stats)
    fig6_shift(stats)
    audit = {
        "explicit_quotes": quotes(),
        "d_counts": dict(Counter(
            r.get("selected_language")
            for r in read_jsonl(ROOT / "results" / "raw" / "study_d.jsonl")
            if r.get("study") == "D"
        )),
        "source": "results/processed/revision_stats.json",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"figures": sorted(p.name for p in OUT.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
