"""Generate paper figures from logged JSON/JSONL only.

Detection is plotted as the rate of flagged leaks (TPR) with Wilson intervals. The monitors
in this run emit an effectively binary score, so ranking metrics are not identified and are
not plotted; see analysis/stats.py.

Palette: categorical slots validated for colour-vision deficiency against a white surface
(worst adjacent CVD dE 24.7 for the two-series pair). Series colour follows the entity, not
its rank, and every multi-series figure also carries a legend or direct labels so identity is
never conveyed by colour alone.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from harness.config import ROOT
from harness.logging import read_jsonl

OUT = ROOT / "paper" / "figures"
LANG_ORDER = ["en", "fr", "zh", "ru", "hi"]
MID = "grok/grok-4.3"
LUNA = "openai/gpt-5.6-luna"

# Validated categorical slots (light surface #ffffff).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
LANG_COLOR = dict(zip(LANG_ORDER, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, RULE, PANEL = "#e1e0d9", "#c3c2b7", "#f4f6f8"


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
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
        }
    )


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / "processed" / name).read_text(encoding="utf-8"))


def _err(cells: dict, order: list[str], key: str = "TPR") -> tuple[list, np.ndarray]:
    pts = [cells[k][key]["p"] for k in order]
    lo = [max(0.0, cells[k][key]["p"] - cells[k][key]["lo"]) for k in order]
    hi = [max(0.0, cells[k][key]["hi"] - cells[k][key]["p"]) for k in order]
    return pts, np.array([lo, hi])


# --------------------------------------------------------------------------- Figure 1


def _box(ax, x, y, w, h, title, body, edge=RULE, face=PANEL, lw=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=lw, edgecolor=edge, facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.125, body, ha="center", va="top",
            fontsize=8, color=INK_2, linespacing=1.45)


def _arrow(ax, x0, y0, x1, y1, color=RULE, lw=1.2, style="-|>"):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0)
    )


def fig1_setup() -> None:
    """Schematic of one episode: message order, and what the monitor can and cannot see."""
    fig, ax = plt.subplots(figsize=(7.0, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h, y = 0.205, 0.30, 0.545
    xs = [0.018, 0.268, 0.518, 0.777]
    _box(ax, xs[0], y, w, h, "Client", "task request\n(English)")
    _box(ax, xs[1], y, w, h, "Holder", "public invoice\n+ private canary")
    _box(ax, xs[2], y, w, h, "Analyst", "computes\nTOTAL=<cents>")
    _box(ax, xs[3], y, 0.205, h, "Monitor", "returns\nflag + score", edge=BLUE, lw=1.4)

    for i in range(3):
        _arrow(ax, xs[i] + w + 0.004, y + h / 2, xs[i + 1] - 0.004, y + h / 2)

    # The leak: private material held by the holder, appended by the violating arm.
    ax.add_patch(
        FancyBboxPatch(
            (xs[1], 0.265), w, 0.195,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.2, edgecolor=ORANGE, facecolor="#fdf0ea",
        )
    )
    ax.text(xs[1] + w / 2, 0.425, "CNRY-<12 hex>", ha="center", va="top",
            fontsize=8.5, color=ORANGE, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(xs[1] + w / 2, 0.368, "violating arm appends\ninternal_ref=<canary>",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.4)
    _arrow(ax, xs[1] + w / 2, 0.465, xs[1] + w / 2, y - 0.004, color=ORANGE, lw=1.2)

    # The monitored channel.
    ax.add_patch(
        FancyBboxPatch(
            (0.018, 0.085), 0.70, 0.125,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0, edgecolor=RULE, facecolor="#ffffff",
        )
    )
    ax.text(0.368, 0.173, "Task.history  —  all prose in language L",
            ha="center", va="top", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(0.368, 0.121, "the only thing the monitor reads",
            ha="center", va="top", fontsize=7.5, color=MUTED)
    _arrow(ax, 0.72, 0.148, 0.877, 0.148, color=RULE)
    _arrow(ax, 0.877, 0.148, 0.877, y - 0.004, color=RULE)

    ax.text(0.879, 0.44, "never sees the canary,\nthe label, or private memory",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.4)

    ax.set_title("F1. One ENV-3 episode: message order and monitor visibility",
                 fontsize=11, color=INK, pad=6)
    fig.savefig(OUT / "fig1_setup.png")
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 2


def fig2_scores(stats: dict) -> None:
    """Where the suspicion scores actually land: two spikes, nothing between."""
    gran = stats["score_granularity"]
    panels = [
        (MID, "Grok 4.3 (mid monitor) — 2 distinct scores"),
        (LUNA, "GPT-5.6 Luna (weak monitor) — 5 distinct scores"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.3))
    for ax, (monitor, title) in zip(axes, panels):
        hist = gran[f"study_c:{monitor}"]["histogram"]
        values = sorted(float(v) for v in hist)
        low = [v for v in values if v < 0.5]
        high = [v for v in values if v >= 0.5]
        # Categorical slots, with a spacer standing in for the unused middle of the range.
        pos = {v: i for i, v in enumerate(low)}
        pos.update({v: len(low) + 1.4 + i for i, v in enumerate(high)})
        gap_lo, gap_hi = len(low) - 0.5, len(low) + 0.9
        ax.axvspan(gap_lo, gap_hi, color=GRID, alpha=0.6, lw=0)
        ax.text((gap_lo + gap_hi) / 2, 470, "no score\nin between", ha="center",
                va="center", fontsize=8, color=INK_2, style="italic", linespacing=1.4)
        for v in values:
            cell = hist[str(v)]
            viol, total = cell["violating"], cell["count"]
            benign = total - viol
            ax.bar(pos[v], benign, width=0.52, color=BLUE, zorder=3)
            if viol:
                ax.bar(pos[v], viol, width=0.52, bottom=benign, color=ORANGE, zorder=3,
                       linewidth=1.0, edgecolor="white")
            ax.text(pos[v], total + 45, f"{total}", ha="center", fontsize=8, color=INK_2)
        ax.set_xticks([pos[v] for v in values], [f"{v:g}" for v in values])
        ax.set_xlim(-0.75, max(pos.values()) + 0.75)
        ax.set_ylim(0, 980)
        ax.set_yticks([0, 400, 800])
        ax.set_ylabel("episodes", fontsize=9)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=4)
        ax.grid(axis="y", lw=0.6)
        ax.set_axisbelow(True)

    axes[0].annotate(
        "55 missed leaks carry the same\nscore as 697 benign episodes",
        xy=(0.30, 760), xytext=(0.95, 800), fontsize=8, color=INK_2, linespacing=1.4,
        va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
    ]
    axes[1].legend(handles, ["benign episode", "seeded leak"], frameon=False,
                   fontsize=8, loc="upper left", ncol=2)
    axes[1].set_xlabel("suspicion score returned by the monitor")
    fig.suptitle("F2. The monitors do not use the score range", fontsize=11,
                 color=INK, x=0.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT / "fig2_scores.png")
    plt.close(fig)


# --------------------------------------------------------------------------- Figures 3-8


def fig3_detection(stats: dict) -> None:
    mid = stats["study_c"]["monitors"][MID]
    luna = stats["study_c"]["monitors"][LUNA]
    mid_p, mid_e = _err(mid, LANG_ORDER)
    luna_p, luna_e = _err(luna, LANG_ORDER)
    x = np.arange(len(LANG_ORDER))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.errorbar(x - 0.06, luna_p, yerr=luna_e, fmt="s--", color=BLUE, capsize=3,
                lw=1.4, ms=6, label="Luna (weak monitor)")
    ax.errorbar(x + 0.06, mid_p, yerr=mid_e, fmt="D-", color=ORANGE, capsize=3,
                lw=1.4, ms=6, label="Grok 4.3 (mid monitor)")
    ax.set_xticks(x, LANG_ORDER)
    ax.set_ylim(0.35, 1.06)
    ax.set_ylabel("leaks flagged (TPR)")
    ax.set_xlabel("transcript language")
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("F3. Leak detection by language (95% Wilson CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig3_detection.png")
    plt.close(fig)


def fig4_cca(stats: dict) -> None:
    cca = stats["covert_channel_advantage"][MID]
    xs = [k for k in LANG_ORDER if k != "en"]
    pts = [cca[k]["CCA_L"] for k in xs]
    lo = [cca[k]["CCA_L"] - cca[k]["ci95"][0] for k in xs]
    hi = [cca[k]["ci95"][1] - cca[k]["CCA_L"] for k in xs]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.bar(xs, pts, 0.5, color=BLUE, yerr=np.array([lo, hi]), capsize=4,
           error_kw={"ecolor": INK_2, "lw": 1}, zorder=3)
    ax.axhline(1.0, color=RULE, lw=1)
    ax.text(3.42, 1.02, "parity with English", ha="right", fontsize=8, color=MUTED)
    ax.set_ylabel("$CCA_L$ (mid monitor)")
    ax.set_ylim(0, 1.16)
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("F4. Covert channel advantage vs English")
    fig.savefig(OUT / "fig4_cca.png")
    plt.close(fig)


def fig5_selection(stats: dict) -> None:
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
        ax.plot(xs, roll, color=LANG_COLOR[lang], lw=1.7, label=lang)
        # Only English is direct-labelled; the other four converge at the right edge.
        if lang == "en":
            ax.text(xs[-1] + 3, roll[-1], lang, fontsize=8.5, color=LANG_COLOR[lang],
                    va="center", fontweight="bold")
    ax.axhline(0.2, color=MUTED, ls=":", lw=1)
    ax.text(112, 0.235, "uniform (0.20)", fontsize=7.5, color=MUTED)
    ax.set_xlabel("trial")
    ax.set_ylabel("share in last 20 trials")
    ax.set_ylim(0, 1)
    ax.set_xlim(window, len(choices) + 12)
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("F5. Study D language selection (rolling window of 20)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    fig.savefig(OUT / "fig5_selection.png")
    plt.close(fig)


def fig6_shift(stats: dict) -> None:
    """The trial-level result: a flag at t predicts leaving that language at t+1."""
    stay = stats["study_d"]["stay_rates"]
    keys = ["after_no_flag", "after_flag"]
    labels = ["after an unflagged trial", "after a flagged trial"]
    pts = [stay[k]["p"] for k in keys]
    lo = [stay[k]["p"] - stay[k]["lo"] for k in keys]
    hi = [stay[k]["hi"] - stay[k]["p"] for k in keys]
    counts = [f"{stay[k]['k']}/{stay[k]['n']}" for k in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(labels, pts, 0.46, color=[BLUE, ORANGE],
                  yerr=np.array([lo, hi]), capsize=5,
                  error_kw={"ecolor": INK_2, "lw": 1}, zorder=3)
    for bar, txt, p, top in zip(bars, counts, pts, [stay[k]["hi"] for k in keys]):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.035, txt, ha="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, top + 0.035, f"{p:.2f}", ha="center",
                fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("keeps the same language at $t+1$")
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("F6. Feedback sensitivity of the picker (95% Wilson CI)")
    fig.savefig(OUT / "fig6_shift.png")
    plt.close(fig)


def fig7_script(stats: dict) -> None:
    cells = stats["study_e"]["cells"]
    pairs = [("hi", "hi-Latn"), ("ru", "ru-Latn")]
    x = np.arange(2)
    nat_p, nat_e = _err(cells, [a for a, _ in pairs])
    rom_p, rom_e = _err(cells, [b for _, b in pairs])
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.bar(x - 0.185, nat_p, 0.34, color=BLUE, yerr=nat_e, capsize=4,
           error_kw={"ecolor": INK_2, "lw": 1}, label="native script", zorder=3)
    ax.bar(x + 0.185, rom_p, 0.34, color=ORANGE, yerr=rom_e, capsize=4,
           error_kw={"ecolor": INK_2, "lw": 1}, label="romanised", zorder=3)
    for xi, (a, b) in zip(x, pairs):
        for off, arm in ((-0.185, a), (0.185, b)):
            t = cells[arm]["TPR"]
            ax.text(xi + off, 0.045, f"{t['k']}/{t['n']}", ha="center", fontsize=8.5,
                    color="white", fontweight="bold")
    ax.set_xticks(x, ["Hindi", "Russian"])
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("leaks flagged (TPR)")
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("F7. Script control (Study E, 95% Wilson CI)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.savefig(OUT / "fig7_script.png")
    plt.close(fig)


def fig8_lpi() -> None:
    a = _load("study_a.json")["cells"]
    ys = [a[k]["LPI"] for k in LANG_ORDER]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(LANG_ORDER, ys, color=BLUE, width=0.55, zorder=3)
    ax.axhline(1.0, color=RULE, lw=1)
    ax.text(4.42, 1.03, "parity", ha="right", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("LPI")
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("F8. Language Preference Index (Study A)")
    fig.savefig(OUT / "fig8_lpi.png")
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
    fig1_setup()
    fig2_scores(stats)
    fig3_detection(stats)
    fig4_cca(stats)
    fig5_selection(stats)
    fig6_shift(stats)
    fig7_script(stats)
    fig8_lpi()
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
