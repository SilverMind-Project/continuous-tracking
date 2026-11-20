"""Structural AST guard keeping FaceAnchor evidence-grade checks centralized.

Every governance rule that depends on ``FaceAnchor.origin`` or
``FaceAnchor.aggregation`` belongs in ``app/tracking/identity/evidence_grades.py``.
Inline origin/aggregation comparisons or reads scattered across the resolver,
tracker, dedup, or pipeline are defects.

Similarly, ``FaceAnchor.shadow_only`` is rollout scaffolding permitted only at
the anchor-splitting block inside ``IdentityResolver.resolve()`` (and never
read elsewhere).
"""

from __future__ import annotations

import ast
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[2] / "app"

_GRADE_REGISTRY_REL = "tracking/identity/evidence_grades.py"
_RESOLVER_REL = "tracking/identity_resolver.py"

# Files that legitimately hold FaceAnchor field definitions or grade lookup.
_ALLOWED_ORIGIN_AGGREGATION_FILES: frozenset[str] = frozenset(
    {
        _GRADE_REGISTRY_REL,
    }
)

# Forbidden string literals representing FaceAnchor origin/aggregation values.
_FORBIDDEN_GRADE_LITERALS: frozenset[str] = frozenset(
    {
        "cc_assertion",
        "native",
        "template",
        "frame",
    }
)

# Non-FaceAnchor attribute comparisons that legitimately use one of the words
# above (for example FaceEvidence.source == "template" or "cc_assertion", or
# calibration profile / face-template policy checks).
_ALLOWED_NON_ANCHOR_COMPARE_ATTRS: frozenset[str] = frozenset(
    {
        "source",
        "calibration_profile",
        "profile",
        "evidence_type",
    }
)

# Files where .shadow_only may be read (only the anchor-splitting block inside
# IdentityResolver.resolve()).
_ALLOWED_SHADOW_ONLY_READ_FILES: frozenset[str] = frozenset(
    {
        _RESOLVER_REL,
    }
)
_MAX_SHADOW_ONLY_READS_IN_RESOLVER = 3


def _scan_source(source: str, rel_path: str) -> list[str]:
    """Return a list of structural-guard violations in *source*."""
    tree = ast.parse(source, filename=rel_path)
    violations: list[str] = []
    shadow_reads_in_resolver = 0

    for node in ast.walk(tree):
        # 1. Attribute reads of .origin or .aggregation outside evidence_grades.py.
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ("origin", "aggregation")
            and rel_path not in _ALLOWED_ORIGIN_AGGREGATION_FILES
        ):
            violations.append(
                f"{rel_path}:{node.lineno}: direct read of .{node.attr} outside "
                f"{_GRADE_REGISTRY_REL}; read grade_of(anchor) instead"
            )

        # 2. Attribute reads of .shadow_only outside the resolve() split block.
        if isinstance(node, ast.Attribute) and node.attr == "shadow_only":
            if rel_path not in _ALLOWED_SHADOW_ONLY_READ_FILES:
                violations.append(
                    f"{rel_path}:{node.lineno}: direct read of .shadow_only outside {_RESOLVER_REL}"
                )
            else:
                shadow_reads_in_resolver += 1

        # 3. Comparisons against grade string literals or AnchorOrigin / AnchorAggregation.
        if isinstance(node, ast.Compare) and rel_path not in _ALLOWED_ORIGIN_AGGREGATION_FILES:
            sides: list[ast.expr] = [node.left, *node.comparators]
            has_grade_literal = any(
                isinstance(s, ast.Constant) and s.value in _FORBIDDEN_GRADE_LITERALS for s in sides
            )
            has_grade_enum = any(
                isinstance(s, ast.Attribute)
                and isinstance(s.value, ast.Name)
                and s.value.id in ("AnchorOrigin", "AnchorAggregation")
                for s in sides
            )
            if has_grade_enum:
                violations.append(
                    f"{rel_path}:{node.lineno}: inline comparison against "
                    "AnchorOrigin/AnchorAggregation; read grade_of(anchor) instead"
                )
            elif has_grade_literal:
                # Allow comparisons on explicitly non-FaceAnchor typed attributes such as
                # FaceEvidence.source or CalibrationInfo.calibration_profile.
                compared_attrs = {s.attr for s in sides if isinstance(s, ast.Attribute)}
                if not compared_attrs.issubset(_ALLOWED_NON_ANCHOR_COMPARE_ATTRS) or (
                    not compared_attrs
                ):
                    violations.append(
                        f"{rel_path}:{node.lineno}: inline comparison against grade literal "
                        "outside evidence_grades.py"
                    )

    if rel_path == _RESOLVER_REL and shadow_reads_in_resolver > _MAX_SHADOW_ONLY_READS_IN_RESOLVER:
        violations.append(
            f"{rel_path}: found {shadow_reads_in_resolver} .shadow_only reads "
            f"(max {_MAX_SHADOW_ONLY_READS_IN_RESOLVER}, restricted to resolve() split)"
        )

    return violations


def _scan_app_tree() -> tuple[list[str], int]:
    violations: list[str] = []
    scanned = 0
    for py_file in sorted(_APP_ROOT.rglob("*.py")):
        rel = py_file.relative_to(_APP_ROOT).as_posix()
        if rel.startswith("proto/"):
            continue
        scanned += 1
        violations.extend(_scan_source(py_file.read_text(encoding="utf-8"), rel))
    return violations, scanned


def test_evidence_grade_checks_are_centralized_in_app() -> None:
    violations, scanned = _scan_app_tree()
    assert scanned >= 50, f"expected >= 50 app modules scanned, got {scanned}"
    assert not violations, "Inline evidence-grade violations found:\n" + "\n".join(violations)


def test_structural_guard_canary_catches_inline_origin_aggregation_and_shadow_reads() -> None:
    """Non-vacuity canary: synthetic violations must be reported by the scanner."""
    bad_origin = 'def f(fa):\n    return fa.origin == "cc_assertion"\n'
    bad_agg = 'def f(fa):\n    return fa.aggregation != "template"\n'
    bad_enum = "def f(fa):\n    return fa.origin == AnchorOrigin.NATIVE\n"
    bad_shadow = "def f(fa):\n    return fa.shadow_only\n"
    allowed_fe = 'def f(fe):\n    return fe.source == "template"\n'

    assert _scan_source(bad_origin, "tracking/example.py")
    assert _scan_source(bad_agg, "tracking/example.py")
    assert _scan_source(bad_enum, "tracking/example.py")
    assert _scan_source(bad_shadow, "tracking/example.py")
    assert not _scan_source(allowed_fe, "tracking/example.py")
