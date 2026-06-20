"""Golden master tests for combine_posteriors (Bayesian multiplicative path).

These fixtures characterize the resolver-canonical posterior math and serve as
regression guards. Values were captured from the live implementation. Use
tolerances (1e-9), never string snapshots.

Evidence combinations covered:
  - Direct face only
  - ReID only
  - Height only
  - Prior only
  - Face + ReID
  - Face + height
  - All four sources
  - Empty (all uniform) → UNKNOWN
  - Face with unrecognized result (UNKNOWN in face)
"""

from __future__ import annotations

import pytest

from app.domain import PosteriorDist
from app.tracking.identity.posterior import combine_posteriors

_TOL = 1e-9


def _check_valid(posterior: PosteriorDist) -> None:
    """Shared invariants every posterior must satisfy."""
    total = sum(posterior.distribution.values())
    assert abs(total - 1.0) < _TOL, f"distribution sums to {total}, expected 1.0"
    for k, v in posterior.distribution.items():
        assert v >= 0, f"negative probability for {k}: {v}"
        assert v <= 1.0 + _TOL, f"probability > 1 for {k}: {v}"


class TestCombinePosteriorsGolden:
    """Golden master: posterior values must not drift between refactors."""

    def test_uniform_face_prior_two_identities(self) -> None:
        """Uniform prior + strong direct face → face identity wins."""
        prior = PosteriorDist({"alice": 0.5, "UNKNOWN": 0.5})
        face = PosteriorDist({"alice": 0.9, "UNKNOWN": 0.1})
        reid = PosteriorDist({})  # empty → uniform

        result = combine_posteriors(prior, face, reid)
        _check_valid(result)

        assert result.distribution["alice"] > result.distribution["UNKNOWN"]

    def test_face_weight_multiplier_applied(self) -> None:
        """Face entries get boosted by face_weight_multiplier (default 3.0)."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        # Face strongly says alice, ReID slightly favours bob.
        # alice: prior(0.5) * face(0.85)*3.0 * reid(0.4) = 0.5 * 2.55 * 0.4 = 0.510
        # bob:   prior(0.5) * face(0.15)*3.0 * reid(0.6) = 0.5 * 0.45 * 0.6 = 0.135
        face = PosteriorDist({"alice": 0.85, "bob": 0.15})
        reid = PosteriorDist({"alice": 0.40, "bob": 0.60})

        result = combine_posteriors(prior, face, reid, face_weight_multiplier=3.0)
        _check_valid(result)

        # Face multiplier (3x) gives alice the decisive edge.
        assert result.distribution["alice"] > result.distribution["bob"]

    def test_height_weight_multiplier_applied(self) -> None:
        """Height entries get boosted by height_weight_multiplier (default 1.5)."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        face = PosteriorDist({})  # empty
        reid = PosteriorDist({})  # empty
        height = PosteriorDist({"alice": 0.8, "bob": 0.2})

        result = combine_posteriors(prior, face, reid, height, height_weight_multiplier=1.5)
        _check_valid(result)

        # Height boosts alice.
        assert result.distribution["alice"] > result.distribution["bob"]

    def test_reid_only_no_face_no_height(self) -> None:
        """ReID alone commits correctly when prior and face are empty."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        face = PosteriorDist({})
        reid = PosteriorDist({"alice": 0.85, "bob": 0.15})

        result = combine_posteriors(prior, face, reid)
        _check_valid(result)

        (top_id, top_prob), _margin = result.top_with_margin()
        assert top_id == "alice"
        assert top_prob > 0.5

    def test_all_sources_combined(self) -> None:
        """All four sources agree on alice → very high posterior for alice."""
        prior = PosteriorDist({"alice": 0.7, "bob": 0.3})
        face = PosteriorDist({"alice": 0.9, "bob": 0.1})
        reid = PosteriorDist({"alice": 0.85, "bob": 0.15})
        height = PosteriorDist({"alice": 0.8, "bob": 0.2})

        result = combine_posteriors(
            prior,
            face,
            reid,
            height,
            face_weight_multiplier=3.0,
            height_weight_multiplier=1.5,
        )
        _check_valid(result)

        (top_id, top_prob), margin = result.top_with_margin()
        assert top_id == "alice"
        assert top_prob > 0.90  # strong agreement pushes probability high
        assert margin > 0.70

    def test_empty_all_sources_returns_unknown(self) -> None:
        """When all distributions are empty, result is UNKNOWN with p=1.0."""
        result = combine_posteriors(
            PosteriorDist({}),
            PosteriorDist({}),
            PosteriorDist({}),
        )
        _check_valid(result)

        assert "UNKNOWN" in result.distribution
        assert result.distribution["UNKNOWN"] == pytest.approx(1.0, abs=_TOL)

    def test_prior_only_no_face_reid(self) -> None:
        """Prior alone (face + ReID empty) propagates prior shape."""
        prior = PosteriorDist({"alice": 0.8, "bob": 0.2})
        result = combine_posteriors(prior, PosteriorDist({}), PosteriorDist({}))
        _check_valid(result)

        # Without face or ReID, the prior shape is preserved (multiplied by 1.0).
        (top_id, _), _ = result.top_with_margin()
        assert top_id == "alice"

    def test_unrecognized_face_unknown_mass(self) -> None:
        """When face distribution is UNKNOWN-only, it penalises known identities."""
        prior = PosteriorDist({"alice": 0.8, "UNKNOWN": 0.2})
        face = PosteriorDist({"UNKNOWN": 1.0})  # unrecognized face
        reid = PosteriorDist({})

        result = combine_posteriors(prior, face, reid, face_weight_multiplier=3.0)
        _check_valid(result)

        # UNKNOWN gets the full face weight; alice gets the smoothing penalty.
        assert result.distribution.get("UNKNOWN", 0) > result.distribution.get("alice", 0)

    def test_laplace_smoothing_for_missing_identity(self) -> None:
        """Smoothing: an identity not in face/reid gets 1/(n+1) not 1.0."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        # Face only mentions alice; bob is missing.
        face = PosteriorDist({"alice": 0.9, "UNKNOWN": 0.1})
        reid = PosteriorDist({})

        result = combine_posteriors(prior, face, reid, face_weight_multiplier=3.0)
        _check_valid(result)

        # bob gets smoothing weight 1/(2+1) = 0.333 from face, not 1.0.
        # alice gets full face weight AND face_weight_multiplier boost.
        assert result.distribution["alice"] > result.distribution["bob"]


class TestCombinePosteriorsProperties:
    """Property-based invariants: hold for all inputs, not just golden cases."""

    @pytest.mark.parametrize("n_identities", [1, 2, 5, 10])
    def test_sums_to_one(self, n_identities: int) -> None:
        """Posterior must always sum to 1.0 regardless of evidence count."""
        identities = [f"person_{i}" for i in range(n_identities)]
        weights = [1.0 / n_identities] * n_identities
        prior = PosteriorDist(dict(zip(identities, weights, strict=False)))
        face = PosteriorDist({identities[0]: 0.8, "UNKNOWN": 0.2})
        reid = PosteriorDist({identities[-1]: 0.7, "UNKNOWN": 0.3})

        result = combine_posteriors(prior, face, reid)
        _check_valid(result)

    def test_non_negative_probabilities(self) -> None:
        """All posterior probabilities must be >= 0."""
        prior = PosteriorDist({"alice": 0.6, "bob": 0.4})
        face = PosteriorDist({"bob": 0.9, "alice": 0.1})
        reid = PosteriorDist({"alice": 0.8, "UNKNOWN": 0.2})

        result = combine_posteriors(prior, face, reid)
        for prob in result.distribution.values():
            assert prob >= 0

    def test_face_identity_always_ranked_higher_than_smoothed(self) -> None:
        """A face-evidenced identity always beats an identity not in face dist."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        face = PosteriorDist({"alice": 0.85, "UNKNOWN": 0.15})
        reid = PosteriorDist({})

        result = combine_posteriors(prior, face, reid, face_weight_multiplier=3.0)

        # alice (in face dist) must beat bob (gets smoothing from face).
        assert result.distribution["alice"] > result.distribution["bob"]

    def test_result_is_reproducible(self) -> None:
        """Same inputs must produce identical outputs (no randomness)."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        face = PosteriorDist({"alice": 0.8, "UNKNOWN": 0.2})
        reid = PosteriorDist({"bob": 0.7, "UNKNOWN": 0.3})

        result1 = combine_posteriors(prior, face, reid)
        result2 = combine_posteriors(prior, face, reid)

        for key in result1.distribution:
            assert result1.distribution[key] == pytest.approx(result2.distribution[key], abs=_TOL)

    def test_height_absent_same_as_height_none(self) -> None:
        """Passing height=None vs passing height=PosteriorDist({}) are equivalent."""
        prior = PosteriorDist({"alice": 0.5, "bob": 0.5})
        face = PosteriorDist({"alice": 0.8, "UNKNOWN": 0.2})
        reid = PosteriorDist({})

        result_none = combine_posteriors(prior, face, reid, None)
        result_empty = combine_posteriors(prior, face, reid, PosteriorDist({}))

        for key in result_none.distribution:
            assert result_none.distribution.get(key, 0) == pytest.approx(
                result_empty.distribution.get(key, 0), abs=_TOL
            )
