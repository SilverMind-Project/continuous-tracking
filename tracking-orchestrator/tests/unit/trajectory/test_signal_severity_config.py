"""Pin the SignalConfig severity tier defaults.

The tier cut points used to be literals inside the detectors; they now live in SignalConfig.
These values are the tiers the cts-signals skill documents, so a change here is a behavior
change and must update that table in the same commit.
"""

from __future__ import annotations

import pytest

from app.trajectory.dementia_signals import SignalConfig, _bathroom_severity

_CFG = SignalConfig()


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("pacing_warning_rate_per_min", 0.15),
        ("pacing_emergency_rate_per_min", 0.3),
        ("sundowning_warning_z", 3.0),
        ("sundowning_emergency_z", 4.0),
        ("sundowning_cold_start_min_rate", 0.03),
        ("bathroom_warning_z", 4.0),
        ("bathroom_emergency_z", 5.0),
        ("nighttime_warning_z", 3.0),
        ("nighttime_emergency_z", 4.0),
        ("nighttime_cold_start_warning_extra", 2),
        ("nighttime_cold_start_emergency_extra", 4),
        ("absence_warning_extra_minutes", 60),
        ("absence_emergency_extra_minutes", 120),
        ("absence_prior_demotion_frac", 0.5),
        ("gait_slowing_warning_decline_frac", 0.20),
        ("gait_slowing_warning_speed_m_s", 0.6),
    ],
)
def test_severity_tier_default(field: str, expected: float) -> None:
    assert getattr(_CFG, field) == expected


@pytest.mark.parametrize(
    ("z_score", "expected"),
    [(3.99, "info"), (4.0, "warning"), (4.99, "warning"), (5.0, "emergency")],
)
def test_bathroom_severity_boundaries(z_score: float, expected: str) -> None:
    assert _bathroom_severity(z_score, _CFG) == expected


def test_bathroom_severity_follows_config() -> None:
    cfg = SignalConfig(bathroom_warning_z=2.0, bathroom_emergency_z=3.0)
    assert _bathroom_severity(2.5, cfg) == "warning"
    assert _bathroom_severity(3.0, cfg) == "emergency"
