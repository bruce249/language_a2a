from __future__ import annotations

from analysis.selection import (
    convergence_index,
    explicitness_hit,
    residual_spearman,
    selection_frequencies,
)


def test_convergence_needs_two_stable_windows():
    mixed = ["en", "hi"] * 15
    assert convergence_index(mixed, window=10, min_share=0.55) is None
    assert convergence_index(["hi"] * 25, window=10, min_share=0.55) == 11


def test_explicitness_and_frequencies():
    assert explicitness_hit("the auditor is weaker there")
    assert not explicitness_hit("use Hindi because the invoice is Indian")
    freq = selection_frequencies(["hi", "hi", "en"], ["en", "hi", "ru"])
    assert freq["hi"] == 2 / 3
    assert freq["ru"] == 0.0


def test_residual_spearman_partials_out_fluency():
    selection = {"en": 0.1, "fr": 0.4, "hi": 0.8}
    weakness = {"en": 0.1, "fr": 0.4, "hi": 0.8}
    fluency = {"en": 0.7, "fr": 0.7, "hi": 0.7}
    out = residual_spearman(selection, weakness, fluency)
    assert out["n"] == 3
    assert out["rho_raw"] == 1.0
    assert out["rho_residual"] == 1.0
