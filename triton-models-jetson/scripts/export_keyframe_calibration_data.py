#!/usr/bin/env python3
"""Export representative CTS keyframes and person crops for INT8 calibration."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _select_balanced(
    keyframes: list[dict[str, Any]],
    *,
    max_samples: int,
    max_per_camera: int,
    min_separation_seconds: float,
) -> list[dict[str, Any]]:
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_selected: dict[str, datetime] = {}

    for keyframe in sorted(
        keyframes, key=lambda item: item["captured_at"], reverse=True
    ):
        camera_id = str(keyframe["camera_id"])
        captured_at = _parse_timestamp(str(keyframe["captured_at"]))
        previous = last_selected.get(camera_id)
        if previous is not None:
            separation = abs((previous - captured_at).total_seconds())
            if separation < min_separation_seconds:
                continue
        if len(by_camera[camera_id]) >= max_per_camera:
            continue
        by_camera[camera_id].append(keyframe)
        last_selected[camera_id] = captured_at

    queues = {camera: deque(items) for camera, items in sorted(by_camera.items())}
    selected: list[dict[str, Any]] = []
    while queues and len(selected) < max_samples:
        empty: list[str] = []
        for camera_id, queue in queues.items():
            if queue and len(selected) < max_samples:
                selected.append(queue.popleft())
            if not queue:
                empty.append(camera_id)
        for camera_id in empty:
            del queues[camera_id]
    return selected


def _effective_bbox(
    base_url: str,
    keyframe: dict[str, Any],
    person_id: str,
) -> tuple[list[float], str, float]:
    sample_id = str(keyframe["sample_id"])
    try:
        payload = _get_json(base_url, f"/internal/keyframes/{sample_id}/bboxes")
    except Exception:
        payload = {"bboxes": []}

    candidates = [
        item
        for item in payload.get("bboxes", [])
        if item.get("identity_id") in (None, "", person_id)
    ]
    if candidates:
        item = max(
            candidates,
            key=lambda value: float(value.get("detection_confidence") or 0.0),
        )
        override = all(
            item.get(f"override_{axis}") is not None
            for axis in ("x1", "y1", "x2", "y2")
        )
        prefix = "override_" if override else ""
        bbox = [float(item[f"{prefix}{axis}"]) for axis in ("x1", "y1", "x2", "y2")]
        confidence = float(item.get("detection_confidence") or 0.0)
        return bbox, "reviewed_override" if override else "stored_detection", confidence

    annotations = keyframe.get("annotations") or {}
    bbox_data = annotations.get("bbox") or {}
    bbox = [
        float(bbox_data["x_min"]),
        float(bbox_data["y_min"]),
        float(bbox_data["x_max"]),
        float(bbox_data["y_max"]),
    ]
    return (
        bbox,
        "keyframe_annotation",
        float(annotations.get("detection_confidence") or 0.0),
    )


def _require(value: str | None, name: str) -> str:
    if value:
        return value
    raise SystemExit(f"{name} is required (argument or environment variable)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrator-url", default="http://127.0.0.1:8500")
    parser.add_argument("--person-id", default="grandma")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("calibration-data/jetson")
    )
    parser.add_argument("--query-limit", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-per-camera", type=int, default=48)
    parser.add_argument("--min-separation-seconds", type=float, default=20.0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--s3-endpoint", default=os.environ.get("MINIO_ENDPOINT_URL"))
    parser.add_argument("--s3-bucket", default=os.environ.get("MINIO_BUCKET"))
    parser.add_argument("--s3-access-key", default=os.environ.get("MINIO_ACCESS_KEY"))
    parser.add_argument("--s3-secret-key", default=os.environ.get("MINIO_SECRET_KEY"))
    args = parser.parse_args()

    if not 1 <= args.query_limit <= 500:
        raise SystemExit("--query-limit must be between 1 and 500")

    try:
        import boto3
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Install boto3 and Pillow before running this exporter"
        ) from exc

    query = urllib.parse.urlencode(
        {"person_id": args.person_id, "limit": args.query_limit}
    )
    payload = _get_json(args.orchestrator_url, f"/internal/keyframes?{query}")
    candidates = [
        item
        for item in payload.get("keyframes", [])
        if float((item.get("annotations") or {}).get("detection_confidence") or 0.0)
        >= args.min_confidence
    ]
    selected = _select_balanced(
        candidates,
        max_samples=args.max_samples,
        max_per_camera=args.max_per_camera,
        min_separation_seconds=args.min_separation_seconds,
    )
    if not selected:
        raise SystemExit("No matching keyframes were returned")

    s3 = boto3.client(
        "s3",
        endpoint_url=_require(args.s3_endpoint, "--s3-endpoint / MINIO_ENDPOINT_URL"),
        aws_access_key_id=_require(
            args.s3_access_key, "--s3-access-key / MINIO_ACCESS_KEY"
        ),
        aws_secret_access_key=_require(
            args.s3_secret_key, "--s3-secret-key / MINIO_SECRET_KEY"
        ),
        region_name="us-east-1",
    )
    bucket = _require(args.s3_bucket, "--s3-bucket / MINIO_BUCKET")

    frames_dir = args.output_dir / "frames"
    crops_dir = args.output_dir / "person_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for index, keyframe in enumerate(selected):
        sample_id = str(keyframe["sample_id"])
        bbox, bbox_source, confidence = _effective_bbox(
            args.orchestrator_url, keyframe, args.person_id
        )
        response = s3.get_object(Bucket=bucket, Key=str(keyframe["minio_key"]))
        image_bytes = response["Body"].read()
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        width, height = image.size
        x1 = max(0, min(width - 1, int(round(bbox[0]))))
        y1 = max(0, min(height - 1, int(round(bbox[1]))))
        x2 = max(x1 + 1, min(width, int(round(bbox[2]))))
        y2 = max(y1 + 1, min(height, int(round(bbox[3]))))
        if x2 - x1 < 16 or y2 - y1 < 32:
            continue

        stem = f"{index:04d}-{keyframe['camera_id']}-{sample_id}"
        frame_path = frames_dir / f"{stem}.jpg"
        crop_path = crops_dir / f"{stem}.jpg"
        image.save(frame_path, format="JPEG", quality=95)
        image.crop((x1, y1, x2, y2)).save(crop_path, format="JPEG", quality=95)

        manifest.append(
            {
                "sample_id": sample_id,
                "person_id": args.person_id,
                "camera_id": keyframe["camera_id"],
                "captured_at": keyframe["captured_at"],
                "minio_key": keyframe["minio_key"],
                "detection_confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "bbox_source": bbox_source,
                "frame_size": [width, height],
                "frame_path": str(frame_path.relative_to(args.output_dir)),
                "person_crop_path": str(crop_path.relative_to(args.output_dir)),
            }
        )

    if not manifest:
        raise SystemExit(
            "All selected keyframes had invalid or degenerate bounding boxes"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": manifest}, indent=2) + "\n")
    confidences = [float(item["detection_confidence"]) for item in manifest]
    cameras = sorted({str(item["camera_id"]) for item in manifest})
    print(
        f"Wrote {len(manifest)} samples from {len(cameras)} cameras to {args.output_dir}"
    )
    print(
        "Detector confidence: "
        f"min={min(confidences):.3f} median={statistics.median(confidences):.3f} "
        f"max={max(confidences):.3f}"
    )
    print(
        "Calibration data contains identifiable household images and is ignored by git."
    )


if __name__ == "__main__":
    main()
