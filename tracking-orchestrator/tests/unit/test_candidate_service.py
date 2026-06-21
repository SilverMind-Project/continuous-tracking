import pytest
from uuid import uuid4
from datetime import datetime, UTC, timedelta
from app.tracking.identity.candidate_service import ReIDCandidateService
from app.domain import GalleryEmbedding

@pytest.fixture
def mock_storage():
    class MockStorage:
        def __init__(self):
            self.objects = {}
        async def put_object(self, key, data):
            self.objects[key] = data
        async def delete_object(self, key):
            self.objects.pop(key, None)
    return MockStorage()

@pytest.fixture
def mock_gallery_repo():
    class MockConnection:
        def __init__(self, data):
            self.data = data
            self.events = []
        async def execute(self, sql, *args):
            if "INSERT INTO continuous_tracking.reid_gallery" in sql:
                self.data[args[0]] = {"state": "pending_review", "embedding": args[4], "audit_version": 1}
            elif "UPDATE continuous_tracking.reid_gallery" in sql:
                if "review_note = $4" in sql:
                    # from _transition_state
                    new_state = args[0]
                    candidate_id = args[4]
                elif "identity_id = $1" in sql:
                    # from relabel_candidate
                    new_state = "operator_verified"
                    candidate_id = args[3] # id = $4
                elif "review_reason = $3" in sql:
                    # from undo_review
                    new_state = args[0]
                    candidate_id = args[3]
                else:
                    return

                if candidate_id in self.data:
                    if new_state == "rejected":
                        self.data[candidate_id]["embedding"] = None
                    self.data[candidate_id]["state"] = new_state
                    self.data[candidate_id]["audit_version"] += 1
            elif "INSERT INTO continuous_tracking.gallery_review_events" in sql:
                self.events.append(args)
        
        async def fetchrow(self, sql, *args):
            if "SELECT state, audit_version" in sql:
                candidate_id = args[0]
                if candidate_id in self.data:
                    return {"state": self.data[candidate_id]["state"], "audit_version": self.data[candidate_id]["audit_version"], "identity_id": "test_id"}
                return None
            elif "SELECT crop_key" in sql:
                # Faithful: return the row's real deterministic crop_key so the
                # row-based delete in _transition_state removes the right object
                # (the hardcoded "v1" delete that masked this was removed).
                candidate_id = args[0]
                return {"crop_key": f"reid-candidates/v1/{candidate_id}.jpg"}
            elif "SELECT previous_state" in sql:
                return {"previous_state": "pending_review"}
            return None
        
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass

        def transaction(self):
            class MockTransaction:
                async def __aenter__(self):
                    pass
                async def __aexit__(self, exc_type, exc, tb):
                    pass
            return MockTransaction()

    class MockPool:
        def __init__(self):
            self.data = {}
        
        def acquire(self):
            return MockConnection(self.data)

    class MockRepo:
        def __init__(self):
            self._pool = MockPool()
            
    return MockRepo()

@pytest.mark.asyncio
async def test_candidate_eligibility(mock_gallery_repo, mock_storage):
    service = ReIDCandidateService(mock_gallery_repo, mock_storage)
    
    class MockEntity:
        entity_id = str(uuid4())
    
    # 1. Invalid embedding (not L2 normalized)
    with pytest.raises(ValueError, match="not L2 normalized"):
        await service.create_candidate(
            MockEntity(), "id1", [0.0]*768, b"crop", None
        )
        
    # 2. Face derived mismatch
    embedding = [1.0] + [0.0]*767
    with pytest.raises(ValueError, match="identity_mismatch"):
        await service.create_candidate(
            MockEntity(), "id1", embedding, b"crop", None,
            candidate_reason="face_derived", arcface_identity="id2", effective_identity="id1"
        )
        
    # 3. Successful creation
    cid = await service.create_candidate(
        MockEntity(), "id1", embedding, b"crop", None,
        candidate_reason="face_derived", arcface_identity="id1", effective_identity="id1"
    )
    assert cid is not None
    assert len(mock_storage.objects) == 1

@pytest.mark.asyncio
async def test_candidate_rejection(mock_gallery_repo, mock_storage):
    service = ReIDCandidateService(mock_gallery_repo, mock_storage)
    
    class MockEntity:
        entity_id = str(uuid4())
    
    embedding = [1.0] + [0.0]*767
    cid = await service.create_candidate(MockEntity(), "id1", embedding, b"crop", None)
    
    # Ensure crop exists
    assert "reid-candidates/v1/" + cid + ".jpg" in mock_storage.objects
    
    # Reject candidate
    await service.reject_candidate(cid, actor="operator1", reason="bad_crop")
    
    # Check DB state
    assert mock_gallery_repo._pool.data[cid]["state"] == "rejected"
    assert mock_gallery_repo._pool.data[cid]["embedding"] is None
    
    # Check crop is deleted
    assert "reid-candidates/v1/" + cid + ".jpg" not in mock_storage.objects

@pytest.mark.asyncio
async def test_candidate_approval_and_relabel(mock_gallery_repo, mock_storage):
    service = ReIDCandidateService(mock_gallery_repo, mock_storage)
    
    class MockEntity:
        entity_id = str(uuid4())
    
    embedding = [1.0] + [0.0]*767
    cid = await service.create_candidate(MockEntity(), "id1", embedding, b"crop", None)
    
    # Approve candidate
    await service.approve_candidate(cid, actor="operator1", reason="verified")
    assert mock_gallery_repo._pool.data[cid]["state"] == "operator_verified"
    
    # Relabel candidate
    await service.relabel_candidate(cid, "id2", actor="operator1")
    assert mock_gallery_repo._pool.data[cid]["state"] == "operator_verified"
    
    # Undo
    await service.undo_review(cid, actor="operator1")
    assert mock_gallery_repo._pool.data[cid]["state"] == "pending_review"
