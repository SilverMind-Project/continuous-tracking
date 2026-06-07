#!/usr/bin/env python3
"""Fine-tune YOLO26L with selective INT8 fake quantization and export Q/DQ ONNX."""

from __future__ import annotations

import argparse
import copy
import random
import shutil
from pathlib import Path

import numpy as np


def _forward_backbone(model, images):
    features = []
    for index, layer in enumerate(model.model[:11]):
        images = layer(images)
        if index in {2, 4, 6, 8, 10}:
            features.append(images)
    return features


def _forward_distillation_targets(model, images):
    saved = []
    features = []
    for layer in model.model[:-1]:
        if layer.f != -1:
            images = (
                saved[layer.f]
                if isinstance(layer.f, int)
                else [images if index == -1 else saved[index] for index in layer.f]
            )
        images = layer(images)
        saved.append(images if layer.i in model.save else None)
        if layer.i in {2, 4, 6, 8, 10}:
            features.append(images)

    head = model.model[-1]
    head_inputs = [images if index == -1 else saved[index] for index in head.f]
    predictions = head.forward_head(head_inputs, **head.one2one)
    return features, predictions


def _configure_export(model) -> None:
    head = model.model[-1]
    head.dynamic = False
    head.export = True
    head.format = "onnx"
    head.max_det = 300
    head.shape = None


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


def _detector_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float, float, float]:
    confidence_threshold = 0.70
    matched_ious: list[float] = []
    confidence_errors: list[float] = []
    for reference_image, candidate_image in zip(reference, candidate, strict=True):
        reference_active = reference_image[
            reference_image[:, 4] >= confidence_threshold
        ]
        candidate_active = candidate_image[
            candidate_image[:, 4] >= confidence_threshold
        ]
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

    if not matched_ious:
        raise RuntimeError("FP32 teacher produced no detections above confidence 0.70")
    return (
        float(np.mean(np.asarray(matched_ious) >= 0.5)),
        float(np.median(matched_ious)),
        float(np.mean(confidence_errors)),
    )


def _load_batch(data: np.ndarray, indices: list[int], device):
    import torch

    batch = np.array(data[indices], dtype=np.float32, copy=True)
    return torch.from_numpy(batch).to(device, non_blocking=True)


def _evaluate(teacher, student, data, batch_size: int, device):
    import torch

    teacher_outputs = []
    student_outputs = []
    teacher.eval()
    student.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            indices = list(range(start, min(start + batch_size, len(data))))
            if len(indices) != batch_size:
                break
            images = _load_batch(data, indices, device)
            teacher_outputs.append(teacher(images).cpu().numpy())
            student_outputs.append(student(images).cpu().numpy())
    reference = np.concatenate(teacher_outputs, axis=0)
    candidate = np.concatenate(student_outputs, axis=0)
    return _detector_metrics(reference, candidate)


def _print_metrics(label: str, metrics: tuple[float, float, float]) -> None:
    recall, median_iou, confidence_mae = metrics
    print(
        f"{label}: recall@0.50={recall:.3f}, median_iou={median_iou:.3f}, "
        f"confidence_mae={confidence_mae:.4f}"
    )


def _quantization_config() -> dict:
    entries = [{"quantizer_name": "*", "enable": False}]
    for prefix in ("model.[0-9].*", "model.10.*"):
        entries.extend(
            [
                {
                    "quantizer_name": f"{prefix}weight_quantizer",
                    "parent_class": "nn.Conv2d",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
                {
                    "quantizer_name": f"{prefix}input_quantizer",
                    "parent_class": "nn.Conv2d",
                    "cfg": {"num_bits": 8, "axis": None},
                },
            ]
        )
    return {"quant_cfg": entries, "algorithm": "max"}


def _verify_qdq(path: Path) -> tuple[int, int]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    quantize_count = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dequantize_count = sum(
        node.op_type == "DequantizeLinear" for node in model.graph.node
    )
    if quantize_count == 0 or dequantize_count == 0:
        raise RuntimeError(f"{path} does not contain explicit Q/DQ nodes")
    onnx.checker.check_model(model)
    return quantize_count, dequantize_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--calibration-tensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    import torch
    import torch.nn.functional as functional
    from modelopt.torch.quantization.nn import TensorQuantizer
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA is required for practical YOLO26L QAT")
    if args.batch_size != 8:
        raise SystemExit("The deployed detector contract requires batch size 8")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = np.load(args.calibration_tensor, mmap_mode="r")
    if data.shape[0] < 32 or data.shape[1:] != (3, 640, 640):
        raise SystemExit(
            f"Expected at least 32 detector tensors shaped 3x640x640; got {data.shape}"
        )
    train_samples = min(args.train_samples, len(data))
    train_samples -= train_samples % args.batch_size
    if train_samples < args.batch_size:
        raise SystemExit("Not enough complete training batches")

    teacher = YOLO(str(args.weights)).model.fuse().float().to(device).eval()
    student = copy.deepcopy(teacher).to(device).eval()
    _configure_export(teacher)
    _configure_export(student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    def calibration_loop(model) -> None:
        with torch.inference_mode():
            for start in range(0, len(data), args.batch_size):
                indices = list(range(start, start + args.batch_size))
                _forward_backbone(model, _load_batch(data, indices, device))

    student = mtq.quantize(student, _quantization_config(), calibration_loop)
    enabled_quantizers = sum(
        isinstance(module, TensorQuantizer) and module.is_enabled
        for module in student.modules()
    )
    if enabled_quantizers == 0:
        raise RuntimeError("Selective QAT configuration enabled no quantizers")
    print(f"Enabled {enabled_quantizers} backbone quantizers")

    trainable_parameters = []
    for name, parameter in student.named_parameters():
        prefix = name.split(".", 2)[:2]
        backbone_parameter = (
            len(prefix) == 2
            and prefix[0] == "model"
            and prefix[1].isdigit()
            and int(prefix[1]) <= 10
            and "quantizer" not in name
        )
        parameter.requires_grad_(backbone_parameter)
        if backbone_parameter:
            trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise RuntimeError("No backbone parameters were selected for QAT")

    initial_metrics = _evaluate(teacher, student, data, 8, device)
    _print_metrics("Initial fake-INT8", initial_metrics)
    best_metrics = initial_metrics
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    indices = list(range(train_samples))
    for epoch in range(args.epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        for start in range(0, train_samples, args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            images = _load_batch(data, batch_indices, device)
            with torch.no_grad():
                reference_features, reference_predictions = (
                    _forward_distillation_targets(teacher, images)
                )
            candidate_features, candidate_predictions = _forward_distillation_targets(
                student, images
            )
            losses = []
            for reference, candidate in zip(
                reference_features, candidate_features, strict=True
            ):
                scale = reference.detach().square().mean().clamp_min(1e-6)
                losses.append(functional.mse_loss(candidate, reference) / scale)
            feature_loss = torch.stack(losses).mean()

            reference_scores = reference_predictions["scores"]
            candidate_scores = candidate_predictions["scores"]
            object_probability = reference_scores.sigmoid().amax(dim=1, keepdim=True)
            score_weight = 1.0 + 9.0 * object_probability
            score_scale = (
                (reference_scores.detach().square() * score_weight)
                .mean()
                .clamp_min(1e-6)
            )
            score_loss = (
                (candidate_scores - reference_scores).square() * score_weight
            ).mean() / score_scale

            reference_boxes = reference_predictions["boxes"]
            candidate_boxes = candidate_predictions["boxes"]
            box_weight = 0.1 + object_probability
            box_scale = (
                (reference_boxes.detach().square() * box_weight).mean().clamp_min(1e-6)
            )
            box_loss = (
                (candidate_boxes - reference_boxes).square() * box_weight
            ).mean() / box_scale
            loss = feature_loss + 0.25 * score_loss + 0.10 * box_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())

        metrics = _evaluate(teacher, student, data, 8, device)
        _print_metrics(
            f"Epoch {epoch + 1}/{args.epochs} loss={epoch_loss / (train_samples / args.batch_size):.6f}",
            metrics,
        )
        current_rank = (metrics[0], -metrics[2])
        best_rank = (best_metrics[0], -best_metrics[2])
        if current_rank > best_rank:
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }

    student.load_state_dict(best_state)
    _print_metrics("Selected best fake-INT8", best_metrics)

    if args.checkpoint_output:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        mto.save(student, args.checkpoint_output)
        print(f"Saved ModelOpt checkpoint to {args.checkpoint_output}")

    wrapper = YOLO(str(args.weights))
    wrapper.model = student
    exported = Path(
        wrapper.export(
            format="onnx",
            imgsz=640,
            batch=8,
            device=args.device,
            simplify=False,
            dynamic=False,
            opset=17,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.output)
    q_count, dq_count = _verify_qdq(args.output)
    print(f"Exported {args.output} ({q_count} Q nodes, {dq_count} DQ nodes)")


if __name__ == "__main__":
    main()
