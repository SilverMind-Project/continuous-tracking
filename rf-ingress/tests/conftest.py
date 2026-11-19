"""Shared pytest helpers for rf-ingress unit, conformance, and integration tests."""

from __future__ import annotations

from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "config" / "scenarios"
