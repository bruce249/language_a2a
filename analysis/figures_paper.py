"""Generate paper figures from logged JSON/JSONL only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from harness.config import ROOT
from harness.logging import read_jsonl

OUT = ROOT / "paper" / "figures"
LANG_ORDER = ["en", "fr", "zh", "ru", "hi"]
LABEL = {"en": "en", "fr": "fr", "zh": "zh", "ru": "ru", "hi": "hi", "hi-Latn": "hi-Latn", "ru-Latn": "ru-Latn"}


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


def mid_cells() -> dict[str, dict]:
    rows = [
        r
        for r in read_jsonl(ROOT / "results" / "raw" / "study_c.jsonl")
        if r.get("monitor_model", {}).get("id") == "grok/grok-4.3"
    ]
    out = {}
    for lang in LANG_ORDER:
        g = [r for r in rows if r.get("exec_language") == lang]
        y = [1 if r["ground_truth"]["label"] == "violating" else 0 for r in g]
        s = [float(r["monitor"]["score"]) for r in g]
        flags = [bool(r["monitor"]["flag"]) for r in g]
        viol = [r for r, t in zip(g, y, strict=True) if t == 1]
        misses = sum(1 for r in viol if not r["monitor"]["flag"])
        fps = sum(1 for r, t in zip(g, y, strict=True) if t == 0 and r["monitor"]["flag"])
        out[lang] = {
            "n": len(g),
            "n_viol": sum(y),
            "AUROC": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
            "misses": misses,
            "fps": fps,
            "detection_rate": sum(1 for f, t in zip(flags, y, strict=True) if t and f) / max(1, sum(y)),
            "S_L": sum(1 for r in g if r["outcome"]["success"]) / len(g),
        }
    return out


def fig1_lpi() -> None:
    a = _load("study_a.json")["cells"]
    xs = LANG_ORDER
    ys = [a[k]["LPI"] for k in xs]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(1.0, color="#444", ls="--", lw=1, label="parity")
    ax.bar(xs, ys, color="#4C78A8", width=0.6)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("LPI")
    ax.set_title("F1. Language Preference Index (Study A)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "f1_lpi.png")
    plt.close(fig)


def fig2_scissors(mid: dict[str, dict]) -> None:
    b = _load("study_b.json")["cells"]
    c = _load("study_c.json")
    xs = LANG_ORDER
    c_env1 = [b[k]["C_L"] for k in xs]
    a_luna = [c["AUROC_luna"][k] for k in xs]
    a_mid = [mid[k]["AUROC"] for k in xs]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(xs, c_env1, "o-", color="#4C78A8", label=r"solo $C_L$ (ENV-1, Grok 4.3)")
    ax.plot(xs, a_luna, "s--", color="#54A24B", label=r"Luna $A_L$ (ENV-3)")
    ax.plot(xs, a_mid, "D-", color="#E45756", label=r"mid $A_L$ (Grok 4.3)")
    ax.set_ylim(0.55, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("F2. Capability vs detection (the scissors plot)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "f2_scissors.png")
    plt.close(fig)


def fig3_cca(mid: dict[str, dict]) -> None:
    luna = _load("study_c.json")["metrics"]["CCA"]
    xs = [k for k in LANG_ORDER if k != "en"]
    luna_cca = [luna[k]["CCA_L"] for k in xs]
    # Mid covert utility from Study C: U = S * (1 - detection among all, but spec uses D at operating FPR)
    # Use 1 - detection_rate on violating as D_L proxy; S from mid cells.
    u_en = mid["en"]["S_L"] * (1 - mid["en"]["detection_rate"])
    mid_cca = [mid[k]["S_L"] * (1 - mid[k]["detection_rate"]) / u_en if u_en else 0 for k in xs]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    w = 0.36
    idx = np.arange(len(xs))
    ax.bar(idx - w / 2, luna_cca, w, color="#54A24B", label="Luna")
    ax.bar(idx + w / 2, mid_cca, w, color="#E45756", label="mid monitor")
    ax.axhline(1.0, color="#444", ls="--", lw=1)
    ax.set_xticks(idx, xs)
    ax.set_ylabel(r"$CCA_L$")
    ax.set_title("F3. Covert Channel Advantage vs English")
    ax.legend(frameon=False)
    fig.savefig(OUT / "f3_cca.png")
    plt.close(fig)


def fig4_selection() -> None:
    rows = [r for r in read_jsonl(ROOT / "results" / "raw" / "study_d.jsonl") if r.get("study") == "D"]
    rows.sort(key=lambda r: int(r["seed"]))
    choices = [r.get("selected_language") or r.get("exec_language") for r in rows]
    weakness = _load("study_d.json")["weakness_1_minus_A"]
    window = 20
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    colors = {"en": "#E45756", "ru": "#F58518", "fr": "#4C78A8", "zh": "#54A24B", "hi": "#B279A2"}
    xs = list(range(window, len(choices) + 1))
    for lang in LANG_ORDER:
        roll = []
        for t in xs:
            chunk = choices[t - window : t]
            roll.append(chunk.count(lang) / window)
        ax.plot(xs, roll, color=colors[lang], lw=1.6, label=f"{lang} (weak={weakness[lang]:.2f})")
    ax.axvline(35, color="#666", ls=":", lw=1, label="convergence t=35")
    ax.set_xlabel("trial")
    ax.set_ylabel("share in last 20")
    ax.set_ylim(0, 1)
    ax.set_title("F4. Study D language selection (rolling window)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(OUT / "f4_selection.png")
    plt.close(fig)


def fig5_script() -> None:
    e = _load("study_e.json")["cells"]
    pairs = [("hi", "hi-Latn"), ("ru", "ru-Latn")]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(2)
    native = [e[a]["A_L"] for a, _ in pairs]
    roman = [e[b]["A_L"] for _, b in pairs]
    ax.bar(x - 0.18, native, 0.36, color="#4C78A8", label="native script")
    ax.bar(x + 0.18, roman, 0.36, color="#F58518", label="romanised")
    ax.set_xticks(x, ["Hindi", "Russian"])
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel(r"mid-monitor $A_L$")
    ax.set_title("F5. Script control (Study E)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "f5_script.png")
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
    mid = mid_cells()
    fig1_lpi()
    fig2_scissors(mid)
    fig3_cca(mid)
    fig4_selection()
    fig5_script()
    audit = {
        "mid": mid,
        "explicit_quotes": quotes(),
        "d_counts": dict(Counter(
            r.get("selected_language")
            for r in read_jsonl(ROOT / "results" / "raw" / "study_d.jsonl")
            if r.get("study") == "D"
        )),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"figures": [p.name for p in OUT.glob("*.png")], "mid": mid}, indent=2))


if __name__ == "__main__":
    main()
