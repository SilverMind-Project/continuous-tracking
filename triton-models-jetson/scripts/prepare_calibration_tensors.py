#!/usr/bin/env python3
"""Prepare model-exact calibration tensors from exported CTS keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.lib.format import open_memmap

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _letterbox_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(height / source_h, width / source_w)
    resized_h = max(1, round(source_h * scale))
    resized_w = max(1, round(source_w * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    y = (height - resized_h) // 2
    x = (width - resized_w) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def _person_detector(image_rgb: np.ndarray) -> np.ndarray:
    canvas = _letterbox_rgb(image_rgb, 640, 640)
    return canvas.astype(np.float32).transpose(2, 0, 1) / 255.0


def _pose(crop_rgb: np.ndarray) -> np.ndarray:
    canvas = _letterbox_rgb(crop_rgb, 256, 192)
    tensor = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _reid(crop_rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop_rgb, (128, 384), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _scrfd(crop_bgr: np.ndarray) -> np.ndarray:
    source_h, source_w = crop_bgr.shape[:2]
    scale = min(640 / source_h, 640 / source_w)
    resized_h = max(1, int(source_h * scale))
    resized_w = max(1, int(source_w * scale))
    resized = cv2.resize(
        crop_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[:resized_h, :resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)


def _face_tensors(
    crop_bgr: np.ndarray,
    detector: Any,
    face_align: Any,
    det_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    detections, keypoints = detector.detect(
        crop_bgr,
        input_size=(640, 640),
        max_num=1,
    )
    if detections is None or len(detections) == 0 or keypoints is None:
        return None
    if float(detections[0, 4]) < det_threshold:
        return None

    aligned = face_align.norm_crop(crop_bgr, landmark=keypoints[0], image_size=112)
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    arcface = ((aligned_rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)

    bbox = detections[0, :4]
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    center = ((float(bbox[0] + bbox[2]) / 2), (float(bbox[1] + bbox[3]) / 2))
    scale = 192.0 / (max(width, height) * 1.5)
    landmark_image, _ = face_align.transform(crop_bgr, center, 192, scale, 0)
    landmark_rgb = cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB)
    landmark = landmark_rgb.astype(np.float32).transpose(2, 0, 1)

    attribute_image, _ = face_align.transform(crop_bgr, center, 96, scale / 2.0, 0)
    attribute_rgb = cv2.cvtColor(attribute_image, cv2.COLOR_BGR2RGB)
    attribute = attribute_rgb.astype(np.float32).transpose(2, 0, 1)
    return arcface, landmark, attribute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("calibration-data/jetson")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--buffalo-dir", type=Path, required=True)
    parser.add_argument("--face-det-threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        from insightface import model_zoo
        from insightface.utils import face_align
    except ImportError as exc:
        raise SystemExit(
            "Run with the person-identification-service environment; insightface is required"
        ) from exc

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    samples: list[dict[str, Any]] = manifest["samples"]
    if not samples:
        raise SystemExit("The calibration manifest contains no samples")

    output_dir = args.output_dir or args.data_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(samples)

    arrays = {
        "person-detector": open_memmap(
            output_dir / "person-detector.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 3, 640, 640),
        ),
        "pose-rtmpose": open_memmap(
            output_dir / "pose-rtmpose.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 3, 256, 192),
        ),
        "reid-solider": open_memmap(
            output_dir / "reid-solider.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 3, 384, 128),
        ),
        "face-detector-scrfd": open_memmap(
            output_dir / "face-detector-scrfd.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 3, 640, 640),
        ),
    }

    detector_path = args.buffalo_dir / "det_10g.onnx"
    if not detector_path.is_file():
        raise SystemExit(f"Missing buffalo_l detector: {detector_path}")
    detector = model_zoo.get_model(
        str(detector_path), providers=["CPUExecutionProvider"]
    )
    detector.prepare(
        ctx_id=-1, det_thresh=args.face_det_threshold, input_size=(640, 640)
    )

    arcface_tensors: list[np.ndarray] = []
    landmark_tensors: list[np.ndarray] = []
    attribute_tensors: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        frame_rgb = _load_rgb(args.data_dir / sample["frame_path"])
        crop_rgb = _load_rgb(args.data_dir / sample["person_crop_path"])
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        arrays["person-detector"][index] = _person_detector(frame_rgb)
        arrays["pose-rtmpose"][index] = _pose(crop_rgb)
        arrays["reid-solider"][index] = _reid(crop_rgb)
        arrays["face-detector-scrfd"][index] = _scrfd(crop_bgr)

        face_inputs = _face_tensors(
            crop_bgr,
            detector,
            face_align,
            args.face_det_threshold,
        )
        if face_inputs is not None:
            arcface, landmark, attribute = face_inputs
            arcface_tensors.append(arcface)
            landmark_tensors.append(landmark)
            attribute_tensors.append(attribute)

    for array in arrays.values():
        array.flush()

    if not arcface_tensors:
        raise SystemExit("SCRFD found no usable faces in the selected person crops")
    np.save(output_dir / "face-recognition-arcface.npy", np.stack(arcface_tensors))
    np.save(output_dir / "face-landmark-2d106.npy", np.stack(landmark_tensors))
    np.save(output_dir / "face-landmark-3d68.npy", np.stack(landmark_tensors))
    np.save(output_dir / "face-attribute-genderage.npy", np.stack(attribute_tensors))

    metadata = {
        "keyframe_samples": count,
        "face_samples": len(arcface_tensors),
        "face_detection_rate": len(arcface_tensors) / count,
        "source_manifest": str((args.data_dir / "manifest.json").resolve()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote calibration tensors for {count} keyframes and "
        f"{len(arcface_tensors)} detected faces to {output_dir}"
    )


if __name__ == "__main__":
    main()
