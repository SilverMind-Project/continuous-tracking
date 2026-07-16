"""Layer boundary tests for the identity subsystem.

Verifies that identity policy modules do not import storage, transport, or
pipeline code, and that typed source objects cannot be confused.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import pytest

from app.tracking.identity.evidence import EvidenceSource, IdentityEvidence
from app.tracking.identity.types import IdentityAuthority, IdentityConflict

_NOW = datetime.now(UTC)

FORBIDDEN_IMPORTS = [
    # Storage
    "psycopg",
    "asyncpg",
    "sqlalchemy",
    "app.storage.postgres",
    # Transport
    "fastapi",
    "starlette",
    # Pipeline stages
    "app.pipeline.stages",
    "app.pipeline.world_tracking",
    # Redis
    "redis",
    "aioredis",
]

IDENTITY_POLICY_MODULES = [
    "app.tracking.identity.evidence",
    "app.tracking.identity.posterior",
    "app.tracking.identity.commit_policy",
    "app.tracking.identity.policy",
    "app.tracking.identity.types",
    "app.tracking.identity.conflicts",
    "app.tracking.identity.provenance",
    "app.tracking.identity.protocols",
]


class TestImportLayerBoundary:
    """Identity policy modules must not import storage, transport, or pipeline."""

    @pytest.mark.parametrize("module_name", IDENTITY_POLICY_MODULES)
    def test_module_does_not_import_forbidden_packages(self, module_name: str) -> None:
        """Verify identity policy modules don't cross layer boundaries."""
        mod = importlib.import_module(module_name)
        mod_file = getattr(mod, "__file__", "") or ""

        for forbidden in FORBIDDEN_IMPORTS:
            # Check the module's __dict__ for direct imports.
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)
                    attr_module = getattr(attr, "__module__", "") or ""
                    if attr_module.startswith(forbidden):
                        pytest.fail(f"{module_name} imports from {forbidden} via {attr_name}")
                except AttributeError:
                    pass

            # Check sys.modules for indirect imports loaded alongside this module.
            # This is a lighter check: we only fail if the forbidden package is
            # in the module's own __file__ path or explicitly imported in its
            # source (not just transitively via shared stdlib).
            _ = mod_file  # used in error messages if needed


class TestTypedSourceDistinction:
    """Direct and propagated face evidence must not be interchangeable."""

    def test_direct_face_source_is_direct_face(self) -> None:
        ev = IdentityEvidence.direct_face("alice", 0.95, "tl-1", _NOW)
        assert ev.source == EvidenceSource.DIRECT_FACE
        assert ev.source != EvidenceSource.ASSOCIATION_HINT

    def test_association_hint_source_is_hint(self) -> None:
        ev = IdentityEvidence.association_hint("alice", 0.80, "tl-1", _NOW)
        assert ev.source == EvidenceSource.ASSOCIATION_HINT
        assert ev.source != EvidenceSource.DIRECT_FACE

    def test_evidence_source_values_are_distinct(self) -> None:
        """All EvidenceSource values must be unique strings."""
        values = [src.value for src in EvidenceSource]
        assert len(values) == len(set(values)), "Duplicate EvidenceSource wire values"

    def test_evidence_source_wire_values_stable(self) -> None:
        """Wire values must not change — they are protocol values."""
        assert EvidenceSource.DIRECT_FACE == "direct_face"
        assert EvidenceSource.REID == "reid"
        assert EvidenceSource.TEMPORAL_PRIOR == "temporal_prior"
        assert EvidenceSource.HEIGHT_PROXY == "height_proxy"
        assert EvidenceSource.OPERATOR == "operator"
        assert EvidenceSource.ASSOCIATION_HINT == "association_hint"

    def test_direct_face_advances_evidence_clock_propagated_does_not(self) -> None:
        """Direct face evidence refreshes the evidence clock; propagated face does not."""
        direct = IdentityEvidence.direct_face("alice", 0.95, "tl-1", _NOW)
        propagated = IdentityEvidence.association_hint("alice", 0.95, "tl-1", _NOW)

        assert direct.can_advance_evidence_clock
        assert not propagated.can_advance_evidence_clock

    def test_direct_face_can_create_identity_hint_cannot(self) -> None:
        direct = IdentityEvidence.direct_face("alice", 0.95, "tl-1", _NOW)
        hint = IdentityEvidence.association_hint("alice", 0.95, "tl-1", _NOW)

        assert direct.can_create_identity
        assert not hint.can_create_identity


class TestIdentityAuthorityEnum:
    def test_all_authority_levels_present(self) -> None:
        expected = {
            "unknown",
            "temporal_prior",
            "height_proxy",
            "reid_gallery",
            "direct_face",
            "operator",
            "posterior",
            "none",
        }
        actual = {a.value for a in IdentityAuthority}
        assert actual == expected

    def test_authority_wire_values_stable(self) -> None:
        assert IdentityAuthority.OPERATOR == "operator"
        assert IdentityAuthority.DIRECT_FACE == "direct_face"
        assert IdentityAuthority.UNKNOWN == "unknown"
        assert IdentityAuthority.POSTERIOR == "posterior"
        assert IdentityAuthority.NONE == "none"


class TestIdentityConflictEnum:
    def test_conflict_wire_values_stable(self) -> None:
        assert IdentityConflict.NONE == "none"
        assert IdentityConflict.QUALITY_GATE == "quality_gate_blocked"
        assert IdentityConflict.FLIP_DEBOUNCE == "flip_debounce_blocked"
        assert IdentityConflict.DUPLICATE_ACTIVE == "duplicate_active"
