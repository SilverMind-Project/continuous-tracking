"""Typed domain objects for identity decisions.

Provides enums and value objects that replace anonymous strings and dicts
across the identity subsystem. All types are frozen; no module-level
singletons. EvidenceSource is not redefined here — import it from
``evidence.py``.
"""

from __future__ import annotations

from enum import StrEnum


class IdentityAuthority(StrEnum):
    """Authority level that produced the effective identity.

    Values are stable wire strings; do not rename without a migration.
    Listed from weakest to strongest so comparisons work naturally.

    ``NONE`` and ``POSTERIOR`` were added by codebase-hardening M07 (F9):
    the producer previously left ordinary Bayesian commits with authority
    ``""`` and, worse, set the ArcFace-authority path's ``authority`` to the
    matched identity id rather than a level. ``UNKNOWN`` and ``HEIGHT_PROXY``
    are legacy/reserved members the current producer never emits.
    """

    NONE = "none"
    UNKNOWN = "unknown"
    TEMPORAL_PRIOR = "temporal_prior"
    POSTERIOR = "posterior"
    HEIGHT_PROXY = "height_proxy"
    REID_GALLERY = "reid_gallery"
    DIRECT_FACE = "direct_face"
    OPERATOR = "operator"
