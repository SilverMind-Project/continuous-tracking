"""Pure-function helpers for the world tracker.

No I/O, no DB, no datetime.now() calls. All time and state inputs are
passed in. Easy to unit test.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon


def update_gallery_mean(
    current_mean: list[float] | None,
    new_embedding: list[float] | None,
    observation_count: int,
) -> list[float] | None:
    """Online exponential moving average of ReID embeddings.

    Returns None when no embedding data is available.
    """
    if new_embedding is None:
        return current_mean

    emb = np.asarray(new_embedding, dtype=np.float32)
    if current_mean is None:
        result: list[float] = list(emb.tolist())
        return result

    alpha = 1.0 / min(observation_count + 1, 100.0)
    cur = np.asarray(current_mean, dtype=np.float32)
    updated = (1.0 - alpha) * cur + alpha * emb
    # Re-normalise to unit length.
    norm = np.linalg.norm(updated)
    if norm > 1e-8:
        updated = updated / norm
    result2: list[float] = list(updated.tolist())
    return result2


def update_height_ema(
    current_height_m: float | None,
    new_height_m: float | None,
    alpha: float = 0.1,
) -> float | None:
    """Exponential moving average of height estimates."""
    if new_height_m is None:
        return current_height_m
    if current_height_m is None:
        return new_height_m
    return alpha * new_height_m + (1.0 - alpha) * current_height_m


def is_in_any_room_polygon(
    floor_x_m: float,
    floor_y_m: float,
    room_polygons: dict[str, list[tuple[float, float]]],
) -> bool:
    """Check whether a floor point falls inside any room polygon.

    Uses shapely for robust point-in-polygon tests. Returns False when
    *room_polygons* is empty (no rooms configured).
    """
    if not room_polygons:
        return False

    point = Point(floor_x_m, floor_y_m)
    for _room_id, vertices in room_polygons.items():
        if len(vertices) < 3:
            continue
        poly = Polygon(vertices)
        if poly.contains(point):
            return True
    return False


def resolve_room(
    floor_x_m: float,
    floor_y_m: float,
    camera_id: str,
    room_polygons: dict[str, list[tuple[float, float]]],
    camera_room_map: dict[str, str],
    room_names: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return (room_id, room_name) for a floor point.

    Falls back to the camera→room map when no polygon contains the point.
    """
    point = Point(floor_x_m, floor_y_m)
    for room_id, vertices in room_polygons.items():
        if len(vertices) < 3:
            continue
        if Polygon(vertices).contains(point):
            return room_id, (room_names or {}).get(room_id, room_id)

    room_name = camera_room_map.get(camera_id, "")
    return room_name, room_name


def speed_m_s(state_mean: tuple[float, float, float, float]) -> float:
    """Compute speed in m/s from the state velocity components."""
    vx, vy = state_mean[2], state_mean[3]
    return float(np.sqrt(vx * vx + vy * vy))


def position_sigma_m(state_cov: tuple[float, ...]) -> float:
    """Compute position uncertainty sqrt(trace(position_block) / 2)."""
    cov = np.asarray(state_cov, dtype=np.float64).reshape(4, 4)
    return float(np.sqrt((cov[0, 0] + cov[1, 1]) / 2.0))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised embedding vectors."""
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
