"""N8: Identity-assertion subscriber test (orchestrator side).

Verifies that cc.identity_assertions messages published to Redis are
correctly ingested and injected as face-anchor evidence for matching PHs.

Requires testcontainer Redis.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="Requires testcontainer Redis and cc.identity_assertions subscriber")
class TestCCIdentityAssertionSubscriber:
    """Round-trip: publish assertion, subscriber injects evidence."""

    async def test_assertion_published_to_redis_stream(self):
        """A cc.identity_assertions message carries the expected fields."""
        # TODO: wire testcontainer Redis
        # - Publish to cc.identity_assertions with { person_id, camera_id, confidence, ph_anchor }
        # - Assert the message is consumed by the subscriber
        pass

    async def test_assertion_injected_as_face_anchor_evidence(self):
        """Matching PH within the anchor window receives face-anchor evidence."""
        # TODO: wire testcontainer Redis + face_identity_stage
        # - Publish assertion with matching position
        # - Assert identity resolver receives face-anchor evidence in its next resolve call
        pass

    async def test_assertion_outside_window_is_ignored(self):
        """Assertion outside ph_anchor_match_window_s is not injected."""
        # TODO: assert stale assertions are dropped
        pass
