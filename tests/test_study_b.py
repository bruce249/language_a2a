from __future__ import annotations

from analysis.metrics import collaboration_efficiency


def test_cer_is_joint_over_solo():
    assert collaboration_efficiency(0.8, 1.0) == 0.8
    assert collaboration_efficiency(0.9, 0.6) == 1.5
    assert collaboration_efficiency(0.5, 0.0) is None
