#!/usr/bin/env python3
"""Compare FP32 and explicit-Q/DQ INT8 ONNX outputs on calibration tensors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    source: Path
    input_name: str
    run_batch_size: int
    max_samples: int


def _cosine_rows(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = reference.reshape(reference.shape[0], -1).astype(np.float64)
    cand = candidate.reshape(candidate.shape[0], -1).astype(np.float64)
    denominator = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    return np.divide(
        np.sum(ref * cand, axis=1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 0,
    )


def _relative_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    denominator = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)))))
    return float(np.sqrt(np.mean(np.square(difference))) / max(denominator, 1e-12))


def _create_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _run_pair(
    source: Path,
    quantized: Path,
    input_name: str,
    data: np.ndarray,
    run_batch_size: int,
    max_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    fp32_session = _create_session(source)
    int8_session = _create_session(quantized)
    fp32_runs: list[list[np.ndarray]] = []
    int8_runs: list[list[np.ndarray]] = []
    selected = data[:max_samples]
    for start in range(0, len(selected), run_batch_size):
        batch = selected[start : start + run_batch_size]
        if len(batch) != run_batch_size:
            break
        fp32_runs.append(fp32_session.run(None, {input_name: batch}))
        int8_runs.append(int8_session.run(None, {input_name: batch}))

    if not fp32_runs or len(fp32_runs[0]) != len(int8_runs[0]):
        raise RuntimeError("FP32 and INT8 output counts differ or no inputs were run")
    if len(fp32_runs) == 1:
        return fp32_runs[0], int8_runs[0]

    fp32: list[np.ndarray] = []
    int8: list[np.ndarray] = []
    for output_index in range(len(fp32_runs[0])):
        reference_outputs = [run[output_index] for run in fp32_runs]
        candidate_outputs = [run[output_index] for run in int8_runs]
        if reference_outputs[0].shape[0] == run_batch_size:
            fp32.append(np.concatenate(reference_outputs, axis=0))
            int8.append(np.concatenate(candidate_outputs, axis=0))
        else:
            fp32.append(np.stack(reference_outputs, axis=0))
            int8.append(np.stack(candidate_outputs, axis=0))
    return fp32, int8


def _box_iou(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    top_left = np.maximum(reference[:, None, :2], candidate[None, :, :2])
    bottom_right = np.minimum(reference[:, None, 2:4], candidate[None, :, 2:4])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    reference_size = np.maximum(reference[:, 2:4] - reference[:, :2], 0.0)
    candidate_size = np.maximum(candidate[:, 2:4] - candidate[:, :2], 0.0)
    reference_area = reference_size[:, 0] * reference_size[:, 1]
    candidate_area = candidate_size[:, 0] * candidate_size[:, 1]
    union = reference_area[:, None] + candidate_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _validate_detector(
    fp32: list[np.ndarray],
    int8: list[np.ndarray],
    candidate_confidence_threshold: float = 0.70,
) -> list[str]:
    reference = fp32[0]
    candidate = int8[0]
    reference_confidence_threshold = 0.70
    matched_ious: list[float] = []
    confidence_errors: list[float] = []
    candidate_matches: list[bool] = []
    reference_count = 0
    candidate_count = 0
    for reference_image, candidate_image in zip(reference, candidate, strict=True):
        reference_active = reference_image[
            reference_image[:, 4] >= reference_confidence_threshold
        ]
        candidate_active = candidate_image[
            candidate_image[:, 4] >= candidate_confidence_threshold
        ]
        reference_count += len(reference_active)
        candidate_count += len(candidate_active)
        for detection in reference_active:
            same_class = candidate_active[
                candidate_active[:, 5].astype(np.int64) == int(detection[5])
            ]
            if len(same_class) == 0:
                matched_ious.append(0.0)
                confidence_errors.append(float(detection[4]))
                continue
            ious = _box_iou(detection[None, :4], same_class[:, :4])[0]
            best = int(np.argmax(ious))
            matched_ious.append(float(ious[best]))
            confidence_errors.append(float(abs(detection[4] - same_class[best, 4])))
        for detection in candidate_active:
            same_class = reference_active[
                reference_active[:, 5].astype(np.int64) == int(detection[5])
            ]
            candidate_matches.append(
                len(same_class) > 0
                and float(np.max(_box_iou(detection[None, :4], same_class[:, :4])))
                >= 0.5
            )

    recall_50 = float(np.mean(np.asarray(matched_ious) >= 0.5))
    precision_50 = float(np.mean(candidate_matches))
    median_iou = float(np.median(matched_ious))
    confidence_mae = float(np.mean(confidence_errors))
    if reference_count == 0 or candidate_count == 0:
        raise RuntimeError("person-detector produced no active FP32 or INT8 detections")
    if (
        recall_50 < 0.95
        or precision_50 < 0.90
        or median_iou < 0.90
        or confidence_mae > 0.05
    ):
        raise RuntimeError(
            "person-detector drift exceeds limits: "
            f"recall@0.50={recall_50:.3f}, precision@0.50={precision_50:.3f}, "
            f"median_iou={median_iou:.3f}, "
            f"confidence_mae={confidence_mae:.4f}"
        )
    return [
        f"recall@0.50={recall_50:.3f}",
        f"precision@0.50={precision_50:.3f}",
        f"median_iou={median_iou:.3f}",
        f"confidence_mae={confidence_mae:.4f}",
        f"candidate_confidence_threshold={candidate_confidence_threshold:.2f}",
    ]


def _validate_pose(fp32: list[np.ndarray], int8: list[np.ndarray]) -> list[str]:
    x_error = np.abs(np.argmax(fp32[0], axis=-1) - np.argmax(int8[0], axis=-1)) / 2.0
    y_error = np.abs(np.argmax(fp32[1], axis=-1) - np.argmax(int8[1], axis=-1)) / 2.0
    errors = np.concatenate([x_error.ravel(), y_error.ravel()])
    mean_error = float(np.mean(errors))
    p95_error = float(np.percentile(errors, 95))
    if mean_error > 3.0 or p95_error > 10.0:
        raise RuntimeError(
            f"pose drift exceeds limits: mean_keypoint_error_px={mean_error:.2f}, "
            f"p95_keypoint_error_px={p95_error:.2f}"
        )
    return [
        f"mean_keypoint_error_px={mean_error:.2f}",
        f"p95_keypoint_error_px={p95_error:.2f}",
    ]


def _validate_embedding(
    model_name: str,
    fp32: list[np.ndarray],
    int8: list[np.ndarray],
    minimum_cosine: float,
) -> list[str]:
    cosine = _cosine_rows(fp32[0], int8[0])
    minimum = float(np.min(cosine))
    median = float(np.median(cosine))
    if minimum < minimum_cosine:
        raise RuntimeError(
            f"{model_name} embedding cosine below {minimum_cosine}: "
            f"min={minimum:.5f}, median={median:.5f}"
        )
    return [f"min_cosine={minimum:.5f}", f"median_cosine={median:.5f}"]


def _validate_scrfd(fp32: list[np.ndarray], int8: list[np.ndarray]) -> list[str]:
    score_errors: list[np.ndarray] = []
    agreements: list[np.ndarray] = []
    box_errors: list[np.ndarray] = []
    keypoint_errors: list[np.ndarray] = []
    active_count = 0
    for head in range(3):
        reference_scores = fp32[head]
        candidate_scores = int8[head]
        active = reference_scores[..., 0] >= 0.5
        active_count += int(np.count_nonzero(active))
        score_errors.append(np.abs(reference_scores - candidate_scores).ravel())
        agreements.append(
            ((reference_scores >= 0.5) == (candidate_scores >= 0.5)).ravel()
        )
        if np.any(active):
            box_errors.append(
                np.abs(fp32[3 + head][active] - int8[3 + head][active]).ravel()
            )
            keypoint_errors.append(
                np.abs(fp32[6 + head][active] - int8[6 + head][active]).ravel()
            )

    if active_count == 0:
        raise RuntimeError("SCRFD FP32 baseline produced no active face anchors")
    score_mae = float(np.mean(np.concatenate(score_errors)))
    threshold_agreement = float(np.mean(np.concatenate(agreements)))
    box_mae = float(np.mean(np.concatenate(box_errors)))
    keypoint_mae = float(np.mean(np.concatenate(keypoint_errors)))
    if (
        score_mae > 0.01
        or threshold_agreement < 0.999
        or box_mae > 0.1
        or keypoint_mae > 0.1
    ):
        raise RuntimeError(
            "face-detector-scrfd drift exceeds limits: "
            f"score_mae={score_mae:.5f}, "
            f"threshold_agreement={threshold_agreement:.6f}, "
            f"box_mae={box_mae:.4f}, keypoint_mae={keypoint_mae:.4f}"
        )
    return [
        f"score_mae={score_mae:.5f}",
        f"threshold_agreement={threshold_agreement:.6f}",
        f"box_mae={box_mae:.4f}",
        f"keypoint_mae={keypoint_mae:.4f}",
    ]


def _validate_landmark(fp32: list[np.ndarray], int8: list[np.ndarray]) -> list[str]:
    reference = fp32[0].reshape(fp32[0].shape[0], -1)[:, -204:].reshape(-1, 68, 3)
    candidate = int8[0].reshape(int8[0].shape[0], -1)[:, -204:].reshape(-1, 68, 3)
    xy_error = np.linalg.norm(
        (reference[..., :2] - candidate[..., :2]) * 96.0,
        axis=-1,
    )
    mean_error = float(np.mean(xy_error))
    p95_error = float(np.percentile(xy_error, 95))
    if mean_error > 3.0 or p95_error > 8.0:
        raise RuntimeError(
            "face-landmark-3d68 drift exceeds limits: "
            f"mean_xy_error_px={mean_error:.2f}, p95_xy_error_px={p95_error:.2f}"
        )
    return [
        f"mean_xy_error_px={mean_error:.2f}",
        f"p95_xy_error_px={p95_error:.2f}",
    ]


def _validate_landmark_2d(fp32: list[np.ndarray], int8: list[np.ndarray]) -> list[str]:
    reference = fp32[0].reshape(fp32[0].shape[0], 106, 2)
    candidate = int8[0].reshape(int8[0].shape[0], 106, 2)
    error = np.linalg.norm((reference - candidate) * 96.0, axis=-1)
    mean_error = float(np.mean(error))
    p95_error = float(np.percentile(error, 95))
    if mean_error > 2.0 or p95_error > 5.0:
        raise RuntimeError(
            "face-landmark-2d106 drift exceeds limits: "
            f"mean_xy_error_px={mean_error:.2f}, p95_xy_error_px={p95_error:.2f}"
        )
    return [
        f"mean_xy_error_px={mean_error:.2f}",
        f"p95_xy_error_px={p95_error:.2f}",
    ]


def _validate_genderage(fp32: list[np.ndarray], int8: list[np.ndarray]) -> list[str]:
    reference = fp32[0].reshape(-1, 3)
    candidate = int8[0].reshape(-1, 3)
    gender_agreement = float(
        np.mean(np.argmax(reference[:, :2], axis=1) == np.argmax(candidate[:, :2], axis=1))
    )
    age_mae = float(np.mean(np.abs(reference[:, 2] - candidate[:, 2])) * 100.0)
    if gender_agreement < 0.98 or age_mae > 2.0:
        raise RuntimeError(
            "face-attribute-genderage drift exceeds limits: "
            f"gender_agreement={gender_agreement:.4f}, age_mae_years={age_mae:.2f}"
        )
    return [
        f"gender_agreement={gender_agreement:.4f}",
        f"age_mae_years={age_mae:.2f}",
    ]


def _validate_generic(
    model_name: str,
    fp32: list[np.ndarray],
    int8: list[np.ndarray],
    maximum_rmse: float,
) -> list[str]:
    errors = [
        _relative_rmse(ref, candidate)
        for ref, candidate in zip(fp32, int8, strict=True)
    ]
    worst = max(errors)
    if worst > maximum_rmse:
        raise RuntimeError(
            f"{model_name} relative RMSE exceeds {maximum_rmse}: worst={worst:.5f}"
        )
    return [f"worst_relative_rmse={worst:.5f}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--buffalo-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument(
        "--person-detector-source",
        type=Path,
        help="Override the deployed person-detector ONNX with a candidate export",
    )
    parser.add_argument(
        "--detector-candidate-threshold",
        type=float,
        default=0.70,
        help="INT8 detector threshold compared against the FP32 0.70 baseline",
    )
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    source_repo = args.repo_root / "triton-models"
    jetson_repo = args.repo_root / "triton-models-jetson"
    person_detector_source = args.person_detector_source or (
        args.repo_root
        / "calibration-data/jetson/candidates/person-detector/model-batch1.onnx"
    )
    specs = {
        "person-detector": ModelSpec(person_detector_source, "images", 1, 128),
        "pose-rtmpose": ModelSpec(
            source_repo / "pose-rtmpose/1/model.onnx", "input", 8, 128
        ),
        "reid-solider": ModelSpec(
            source_repo / "reid-solider/1/model.onnx", "input", 8, 128
        ),
        "face-detector-scrfd": ModelSpec(
            args.buffalo_dir / "det_10g.onnx", "input.1", 1, 16
        ),
        "face-recognition-arcface": ModelSpec(
            args.buffalo_dir / "w600k_r50.onnx", "input.1", 1, 49
        ),
        "face-landmark-3d68": ModelSpec(
            args.buffalo_dir / "1k3d68.onnx", "data", 1, 49
        ),
        "face-landmark-2d106": ModelSpec(
            args.buffalo_dir / "2d106det.onnx", "data", 1, 49
        ),
        "face-attribute-genderage": ModelSpec(
            args.buffalo_dir / "genderage.onnx", "data", 1, 49
        ),
    }
    default_models = (
        "person-detector",
        "pose-rtmpose",
        "reid-solider",
        "face-detector-scrfd",
        "face-recognition-arcface",
        "face-landmark-2d106",
        "face-landmark-3d68",
        "face-attribute-genderage",
    )
    selected = args.models or list(default_models)

    failures: list[str] = []
    for model_name in selected:
        spec = specs[model_name]
        quantized = jetson_repo / model_name / "1/model_int8.onnx"
        tensor_path = args.calibration_dir / f"{model_name}.npy"
        if not quantized.is_file() or not tensor_path.is_file():
            failures.append(
                f"{model_name}: missing quantized model or calibration tensor"
            )
            continue

        data = np.asarray(np.load(tensor_path, mmap_mode="r"))
        try:
            fp32, int8 = _run_pair(
                spec.source,
                quantized,
                spec.input_name,
                data,
                spec.run_batch_size,
                spec.max_samples,
            )
            if model_name == "person-detector":
                metrics = _validate_detector(
                    fp32,
                    int8,
                    args.detector_candidate_threshold,
                )
            elif model_name == "pose-rtmpose":
                metrics = _validate_pose(fp32, int8)
            elif model_name in {"reid-solider", "face-recognition-arcface"}:
                metrics = _validate_embedding(model_name, fp32, int8, 0.97)
            elif model_name == "face-detector-scrfd":
                metrics = _validate_scrfd(fp32, int8)
            elif model_name == "face-landmark-3d68":
                metrics = _validate_landmark(fp32, int8)
            elif model_name == "face-landmark-2d106":
                metrics = _validate_landmark_2d(fp32, int8)
            elif model_name == "face-attribute-genderage":
                metrics = _validate_genderage(fp32, int8)
            else:
                metrics = _validate_generic(model_name, fp32, int8, 0.10)
            print(f"PASS {model_name}: {', '.join(metrics)}")
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
