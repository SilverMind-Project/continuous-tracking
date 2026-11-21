#!/usr/bin/env python3
"""Offline trainer for the closed-set household appearance classifier.

Reads frozen body-appearance training embeddings from ``appearance_training_samples``
(plus operator-approved room-grade wearable samples from ``reid_gallery``) over a
read-only database connection, enforces the association spot-check audit bar and
per-class day/sample minimums, selects regularization via leave-one-day-out CV,
calibrates probabilities by temperature scaling on a held-out validation day, gates
on the most recent 3 held-out days (including open-set false-match on negatives),
and exports a versioned JSON artifact (no pickle).

Requires the optional ``classifier-tools`` dependency group (``scikit-learn``),
which is never imported by ``app/``.

Usage:
    DATABASE_URL=postgresql://... uv run --extra classifier-tools \\
        python scripts/train_household_classifier.py \\
        --feature-version reid-solider \\
        --association-audit /path/to/wearable-audit.json \\
        --output /path/to/household-classifier.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import asyncpg
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from app.domain import AppearanceLabelSource, AppearanceTrainingSample, OrientationBin
from app.storage.postgres.appearance_sample_repo import (
    PostgresAppearanceSampleRepository,
    _pgvector_to_tuple,
)
from app.tracking.identity.household_classifier import (
    HouseholdClassifierArtifact,
    HouseholdClassifierArtifactMetrics,
)

TOOL_VERSION = "household-classifier-trainer-v1"
_Z_95 = 1.959963984540054
_NORM_EPS = 1e-8


class TrainingGateError(ValueError):
    """Raised when a pre-fit or post-evaluation acceptance gate fails."""


@dataclass(frozen=True)
class TrainerConfig:
    """Acceptance thresholds and time-split settings for offline classifier training."""

    min_classes: int = 2
    min_days_per_class: int = 3
    min_samples_per_class: int = 300
    eval_days_count: int = 3
    dedup_window_s: float = 10.0
    c_grid: tuple[float, ...] = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
    prob_operating_point: float = 0.8
    min_macro_top1_at_op: float = 0.85
    min_per_class_recall_at_op: float = 0.75
    max_ece: float = 0.05
    max_open_set_fmr_at_op: float = 0.02
    min_audit_samples: int = 100
    min_audit_wilson_lower: float = 0.99
    ece_bins: int = 10
    outlier_exposure_weight: float = 0.25
    seed: int = 42


@dataclass(frozen=True)
class TrainingResult:
    """Output of a successful gated training run."""

    artifact: HouseholdClassifierArtifact
    omitted_classes: tuple[str, ...]
    audit_wilson_lower: float


def wilson_lower_bound(correct: int, total: int, *, z: float = _Z_95) -> float:
    """Return the lower bound of the Wilson score interval for ``correct / total``."""
    if total <= 0 or correct < 0 or correct > total:
        raise TrainingGateError(
            f"Invalid association audit counts: correct={correct}, total={total}"
        )
    p = correct / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, (center - half) / denom)


def validate_association_audit(
    audit_data: Mapping[str, object],
    *,
    min_samples: int = 100,
    min_wilson_lower: float = 0.99,
) -> tuple[int, int, float]:
    """Verify the wearable association spot-check audit evidence file."""
    raw_total = audit_data.get("sample_size")
    raw_correct = audit_data.get("correct_count")
    if (
        isinstance(raw_total, bool)
        or not isinstance(raw_total, int)
        or isinstance(raw_correct, bool)
        or not isinstance(raw_correct, int)
    ):
        raise TrainingGateError(
            "Association audit JSON must contain integer 'sample_size' and 'correct_count'"
        )
    if raw_total < min_samples:
        raise TrainingGateError(
            f"Association audit sample_size={raw_total} is below minimum {min_samples}"
        )
    lower = wilson_lower_bound(raw_correct, raw_total)
    if lower < min_wilson_lower:
        raise TrainingGateError(
            f"Association audit Wilson 95% lower bound {lower:.4f} is below {min_wilson_lower:.4f}"
        )
    return raw_total, raw_correct, lower


def _l2_normalize_matrix(embeddings: Sequence[Sequence[float]]) -> npt.NDArray[np.float64]:
    mat = np.asarray(embeddings, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        raise TrainingGateError("Training embeddings matrix is empty or malformed")
    if not bool(np.all(np.isfinite(mat))):
        raise TrainingGateError("Training embeddings contain non-finite values")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if bool(np.any(norms <= _NORM_EPS)):
        raise TrainingGateError("Training embeddings contain zero-norm vectors")
    return mat / norms


def _softmax_rows(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _mean_log_loss(probs: npt.NDArray[np.float64], y_true: npt.NDArray[np.int64]) -> float:
    clipped = np.clip(probs[np.arange(y_true.size), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def deduplicate_samples(
    samples: Sequence[AppearanceTrainingSample],
    *,
    window_s: float = 10.0,
) -> list[AppearanceTrainingSample]:
    """Keep at most one non-revoked sample per (identity_or_camera, orientation) per window_s."""
    ordered = sorted(
        (s for s in samples if s.revoked_at is None),
        key=lambda s: (s.observed_at, s.sample_id),
    )
    last_seen: dict[tuple[str, str, OrientationBin], datetime] = {}
    kept: list[AppearanceTrainingSample] = []
    for sample in ordered:
        if sample.label_source == AppearanceLabelSource.NEGATIVE:
            key = ("neg", sample.camera_id, sample.orientation)
        else:
            assert sample.identity_id is not None
            key = ("pos", sample.identity_id, sample.orientation)
        prev = last_seen.get(key)
        if prev is not None and (sample.observed_at - prev).total_seconds() < window_s:
            continue
        last_seen[key] = sample.observed_at
        kept.append(sample)
    return kept


def expected_calibration_error(
    probs: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int64],
    *,
    n_bins: int = 10,
) -> float:
    """Compute top-label Expected Calibration Error across uniform probability bins."""
    if y_true.size == 0:
        return 1.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(y_true.size)
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_linear_head(
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    num_classes: int,
    *,
    c_value: float,
    seed: int,
    x_neg: npt.NDArray[np.float64] | None = None,
    outlier_exposure_weight: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit balanced L2-regularized multinomial logistic regression and export (W, b).

    When ``x_neg`` and a positive ``outlier_exposure_weight`` are provided, each
    negative embedding is replicated across all ``num_classes`` labels with equal
    weight (exact uniform-target cross-entropy / outlier exposure) so the linear
    head projects out visitor and daily-clothing directions while staying strictly
    convex inside standard scikit-learn ``LogisticRegression``.

    For 2-class households scikit-learn emits a single ``(1, D)`` binary coefficient
    vector; we export it symmetrically as ``[-0.5 * w, +0.5 * w]`` and
    ``[-0.5 * b, +0.5 * b]`` so 2-class softmax matches ``predict_proba`` identically.
    """
    n_pos = int(y_train.size)
    counts = np.bincount(y_train, minlength=num_classes)
    if bool(np.any(counts == 0)):
        missing = [idx for idx, cnt in enumerate(counts) if cnt == 0]
        raise TrainingGateError(f"Training split is missing classes: {missing}")
    sw_pos = (float(n_pos) / (float(num_classes) * counts[y_train])).astype(np.float64)

    if x_neg is not None and x_neg.shape[0] > 0 and outlier_exposure_weight > 0.0:
        n_neg = int(x_neg.shape[0])
        x_neg_rep = np.repeat(x_neg, num_classes, axis=0)
        y_neg_rep = np.tile(np.arange(num_classes, dtype=np.int64), n_neg)
        per_neg_weight = (outlier_exposure_weight * float(n_pos)) / float(num_classes * n_neg)
        sw_neg = np.full(y_neg_rep.size, per_neg_weight, dtype=np.float64)
        x_fit = np.vstack([x_train, x_neg_rep])
        y_fit = np.concatenate([y_train, y_neg_rep])
        sample_weight = np.concatenate([sw_pos, sw_neg])
    else:
        x_fit = x_train
        y_fit = y_train
        sample_weight = sw_pos

    model = LogisticRegression(
        C=c_value,
        solver="newton-cg",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(x_fit, y_fit, sample_weight=sample_weight)
    fitted_classes = [int(c) for c in model.classes_]
    if len(fitted_classes) != num_classes or fitted_classes != list(range(num_classes)):
        raise TrainingGateError(
            f"Fitted classes {fitted_classes} do not cover all {num_classes} target classes"
        )
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    if num_classes == 2 and coef.shape[0] == 1:
        w = np.vstack([-0.5 * coef[0], 0.5 * coef[0]])
        b = np.array([-0.5 * intercept[0], 0.5 * intercept[0]], dtype=np.float64)
        return w, b
    return coef, intercept


def _uniform_cross_entropy(logits: npt.NDArray[np.float64]) -> float:
    """Mean cross-entropy between predicted softmax probabilities and uniform (1/K, ..., 1/K)."""
    probs = np.clip(_softmax_rows(logits), 1e-12, 1.0)
    return float(-np.mean(np.log(probs)))


def fit_temperature(
    logits: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.int64],
    *,
    neg_logits: npt.NDArray[np.float64] | None = None,
) -> float:
    """Find positive temperature scaling ``T`` minimizing calibrated validation loss.

    When pre-evaluation negative logits are supplied, includes a uniform-target
    cross-entropy term on those negatives alongside a mild ``(log T)^2`` prior so
    linearly separable validation days cannot collapse ``T`` toward zero and cause
    overconfident false matches on open-set visitors.
    """

    def _objective(temp: float) -> float:
        pos_probs = _softmax_rows(logits / temp)
        loss = _mean_log_loss(pos_probs, y_val)
        if neg_logits is not None and neg_logits.shape[0] > 0:
            loss += 0.5 * _uniform_cross_entropy(neg_logits / temp)
        loss += 1e-3 * (math.log(temp) ** 2)
        return loss

    result = minimize_scalar(_objective, bounds=(0.25, 10.0), method="bounded")
    temp = float(result.x)
    if not math.isfinite(temp) or temp <= 0.0:
        return 1.0
    return temp


def _select_c_lodo(
    x_pre: npt.NDArray[np.float64],
    y_pre: npt.NDArray[np.int64],
    days_pre: Sequence[date],
    *,
    num_classes: int,
    c_grid: Sequence[float],
    seed: int,
    x_pre_neg: npt.NDArray[np.float64] | None = None,
    days_pre_neg: Sequence[date] | None = None,
    outlier_exposure_weight: float = 0.0,
) -> tuple[float, float]:
    """Choose ``C`` by leave-one-day-out cross-validation minimizing held-out loss.

    When pre-evaluation negatives are available, each fold trains strictly on
    ``day != held_day`` negatives and evaluates held-out ``day == held_day``
    negatives under uniform-target cross-entropy so high-``C`` models cannot
    memorize daily clothing dimensions that inflate open-set false matches.
    """
    unique_days = sorted(set(days_pre))
    day_arr = np.asarray(days_pre)
    neg_day_arr = np.asarray(days_pre_neg) if days_pre_neg is not None else None
    best_c: float | None = None
    best_loss = float("inf")

    for c_val in c_grid:
        fold_losses: list[float] = []
        for held_day in unique_days:
            val_mask = day_arr == held_day
            train_mask = ~val_mask
            y_tr = y_pre[train_mask]
            y_va = y_pre[val_mask]
            if y_va.size == 0 or len({int(v) for v in y_tr}) < num_classes:
                continue
            neg_tr: npt.NDArray[np.float64] | None = None
            neg_va: npt.NDArray[np.float64] | None = None
            if x_pre_neg is not None and x_pre_neg.shape[0] > 0 and neg_day_arr is not None:
                neg_tr_mask = neg_day_arr != held_day
                neg_va_mask = neg_day_arr == held_day
                if bool(np.any(neg_tr_mask)):
                    neg_tr = x_pre_neg[neg_tr_mask]
                if bool(np.any(neg_va_mask)):
                    neg_va = x_pre_neg[neg_va_mask]
            w, b = fit_linear_head(
                x_pre[train_mask],
                y_tr,
                num_classes,
                c_value=c_val,
                seed=seed,
                x_neg=neg_tr,
                outlier_exposure_weight=outlier_exposure_weight,
            )
            logits_va = x_pre[val_mask] @ w.T + b
            fold_loss = _mean_log_loss(_softmax_rows(logits_va), y_va)
            if neg_va is not None and neg_va.shape[0] > 0:
                fold_loss += _uniform_cross_entropy(neg_va @ w.T + b)
            fold_losses.append(fold_loss)
        if not fold_losses:
            raise TrainingGateError(
                "Leave-one-day-out CV had zero valid folds; need more overlapping class days"
            )
        mean_loss = float(np.mean(fold_losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_c = float(c_val)

    assert best_c is not None
    return best_c, best_loss


def _deterministic_artifact_id(
    *,
    feature_version: str,
    classes: Sequence[str],
    trained_days: Sequence[str],
    eval_days: Sequence[str],
    seed: int,
    weights: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "feature_version": feature_version,
            "classes": list(classes),
            "trained_days": list(trained_days),
            "eval_days": list(eval_days),
            "seed": seed,
            "weights": [[round(float(v), 8) for v in row] for row in weights],
            "bias": [round(float(v), 8) for v in bias],
            "temperature": round(float(temperature), 8),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"hc-v1-{digest}"


def train_household_classifier(
    samples: Sequence[AppearanceTrainingSample],
    *,
    feature_version: str,
    association_audit: Mapping[str, object],
    config: TrainerConfig = TrainerConfig(),
    created_at: str = "",
) -> TrainingResult:
    """Validate gates, train with time-split LODO CV + temperature scaling, and return artifact."""
    _, _, audit_lower = validate_association_audit(
        association_audit,
        min_samples=config.min_audit_samples,
        min_wilson_lower=config.min_audit_wilson_lower,
    )

    matching = [s for s in samples if s.feature_version == feature_version and s.revoked_at is None]
    deduped = deduplicate_samples(matching, window_s=config.dedup_window_s)

    positives = [s for s in deduped if s.label_source != AppearanceLabelSource.NEGATIVE]
    negatives = [s for s in deduped if s.label_source == AppearanceLabelSource.NEGATIVE]

    if not positives:
        raise TrainingGateError("No positive appearance training samples found")

    all_pos_days = sorted({s.day for s in positives})
    min_total_days = config.eval_days_count + config.min_days_per_class
    if len(all_pos_days) < min_total_days:
        raise TrainingGateError(
            f"Need at least {min_total_days} distinct positive days "
            f"({config.min_days_per_class} train/CV + {config.eval_days_count} eval), "
            f"found {len(all_pos_days)}"
        )

    eval_day_set = set(all_pos_days[-config.eval_days_count :])
    eval_start_day = min(eval_day_set)
    pre_eval_day_set = set(all_pos_days[: -config.eval_days_count])

    by_class_pre: dict[str, list[AppearanceTrainingSample]] = defaultdict(list)
    by_class_eval: dict[str, list[AppearanceTrainingSample]] = defaultdict(list)
    for sample in positives:
        assert sample.identity_id is not None
        if sample.day in eval_day_set:
            by_class_eval[sample.identity_id].append(sample)
        elif sample.day in pre_eval_day_set:
            by_class_pre[sample.identity_id].append(sample)

    retained_classes: list[str] = []
    omitted_classes: list[str] = []
    all_classes = sorted(set(by_class_pre.keys()) | set(by_class_eval.keys()))
    for cls_id in all_classes:
        pre_rows = by_class_pre.get(cls_id, [])
        eval_rows = by_class_eval.get(cls_id, [])
        pre_days = {r.day for r in pre_rows}
        if (
            len(pre_days) >= config.min_days_per_class
            and len(pre_rows) >= config.min_samples_per_class
            and len(eval_rows) >= 1
        ):
            retained_classes.append(cls_id)
        else:
            omitted_classes.append(cls_id)

    if len(retained_classes) < config.min_classes:
        raise TrainingGateError(
            f"Need at least {config.min_classes} qualifying classes "
            f"(>= {config.min_days_per_class} pre-eval days, "
            f">= {config.min_samples_per_class} deduplicated samples, >= 1 eval sample); "
            f"found {len(retained_classes)} ({retained_classes})"
        )

    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(retained_classes)}
    pre_samples = [
        s for s in positives if s.day in pre_eval_day_set and s.identity_id in class_to_idx
    ]
    eval_pos_samples = [
        s for s in positives if s.day in eval_day_set and s.identity_id in class_to_idx
    ]
    eval_neg_samples = [s for s in negatives if s.day >= eval_start_day]
    if not eval_neg_samples:
        raise TrainingGateError(
            "Open-set gate requires negative samples on the held-out evaluation days; found 0"
        )

    x_pre = _l2_normalize_matrix([s.embedding for s in pre_samples])
    y_pre = np.asarray(
        [class_to_idx[s.identity_id] for s in pre_samples if s.identity_id is not None],
        dtype=np.int64,
    )
    days_pre = [s.day for s in pre_samples]

    pre_neg_samples = [s for s in negatives if s.day < eval_start_day]
    x_pre_neg = (
        _l2_normalize_matrix([s.embedding for s in pre_neg_samples]) if pre_neg_samples else None
    )
    days_pre_neg = [s.day for s in pre_neg_samples] if pre_neg_samples else None

    best_c, cv_log_loss = _select_c_lodo(
        x_pre,
        y_pre,
        days_pre,
        num_classes=len(retained_classes),
        c_grid=config.c_grid,
        seed=config.seed,
        x_pre_neg=x_pre_neg,
        days_pre_neg=days_pre_neg,
        outlier_exposure_weight=config.outlier_exposure_weight,
    )

    # Temperature scaling on a held-out validation day (most recent pre-eval day).
    sorted_pre_days = sorted(set(days_pre))
    val_day = sorted_pre_days[-1]
    day_arr_pre = np.asarray(days_pre)
    val_mask = day_arr_pre == val_day
    fit_mask = ~val_mask
    if len({int(v) for v in y_pre[fit_mask]}) < len(retained_classes):
        raise TrainingGateError(
            f"Holding out validation day {val_day.isoformat()} leaves a class absent from fit days"
        )
    cal_neg_fit: npt.NDArray[np.float64] | None = None
    cal_neg_val: npt.NDArray[np.float64] | None = None
    if x_pre_neg is not None and days_pre_neg is not None:
        neg_day_arr_pre = np.asarray(days_pre_neg)
        neg_fit_mask = neg_day_arr_pre != val_day
        neg_val_mask = neg_day_arr_pre == val_day
        if bool(np.any(neg_fit_mask)):
            cal_neg_fit = x_pre_neg[neg_fit_mask]
        if bool(np.any(neg_val_mask)):
            cal_neg_val = x_pre_neg[neg_val_mask]
    w_cal, b_cal = fit_linear_head(
        x_pre[fit_mask],
        y_pre[fit_mask],
        len(retained_classes),
        c_value=best_c,
        seed=config.seed,
        x_neg=cal_neg_fit,
        outlier_exposure_weight=config.outlier_exposure_weight,
    )
    val_logits = x_pre[val_mask] @ w_cal.T + b_cal
    val_neg_logits = cal_neg_val @ w_cal.T + b_cal if cal_neg_val is not None else None
    temperature = fit_temperature(val_logits, y_pre[val_mask], neg_logits=val_neg_logits)

    # Refit final linear weights on all pre-eval days with the selected C.
    weights, bias = fit_linear_head(
        x_pre,
        y_pre,
        len(retained_classes),
        c_value=best_c,
        seed=config.seed,
        x_neg=x_pre_neg,
        outlier_exposure_weight=config.outlier_exposure_weight,
    )

    # Evaluate strictly on held-out eval_days.
    x_eval_pos = _l2_normalize_matrix([s.embedding for s in eval_pos_samples])
    y_eval_pos = np.asarray(
        [class_to_idx[s.identity_id] for s in eval_pos_samples if s.identity_id is not None],
        dtype=np.int64,
    )
    probs_eval_pos = _softmax_rows((x_eval_pos @ weights.T + bias) / temperature)
    max_probs_pos = np.max(probs_eval_pos, axis=1)
    preds_pos = np.argmax(probs_eval_pos, axis=1)

    op = config.prob_operating_point
    confident_mask = max_probs_pos >= op
    if not bool(np.any(confident_mask)):
        raise TrainingGateError(f"No evaluation samples reached probability >= {op:.2f}")

    per_class_top1_at_op: list[float] = []
    per_class_recall_at_op: dict[str, float] = {}
    for cls_idx, cls_id in enumerate(retained_classes):
        cls_mask = y_eval_pos == cls_idx
        cls_total = int(np.sum(cls_mask))
        if cls_total == 0:
            raise TrainingGateError(f"Trained class {cls_id!r} has zero evaluation samples")
        cls_correct_at_op = int(np.sum(cls_mask & (preds_pos == cls_idx) & (max_probs_pos >= op)))
        recall_at_op = cls_correct_at_op / cls_total
        per_class_recall_at_op[cls_id] = float(recall_at_op)

        cls_conf_mask = cls_mask & confident_mask
        cls_conf_count = int(np.sum(cls_conf_mask))
        cls_acc_at_op = (
            float(np.mean(preds_pos[cls_conf_mask] == cls_idx)) if cls_conf_count > 0 else 0.0
        )
        per_class_top1_at_op.append(cls_acc_at_op)

    macro_top1_at_op = float(np.mean(per_class_top1_at_op))
    ece = expected_calibration_error(probs_eval_pos, y_eval_pos, n_bins=config.ece_bins)

    x_eval_neg = _l2_normalize_matrix([s.embedding for s in eval_neg_samples])
    probs_eval_neg = _softmax_rows((x_eval_neg @ weights.T + bias) / temperature)
    neg_max_probs = np.max(probs_eval_neg, axis=1)
    open_set_fmr = float(np.mean(neg_max_probs >= op))

    failures: list[str] = []
    if macro_top1_at_op < config.min_macro_top1_at_op:
        failures.append(
            f"macro_top1_at_{op:.1f}={macro_top1_at_op:.4f} < {config.min_macro_top1_at_op:.2f}"
        )
    for cls_id, recall in per_class_recall_at_op.items():
        if recall < config.min_per_class_recall_at_op:
            failures.append(
                f"per_class_recall_at_{op:.1f}[{cls_id}]={recall:.4f} "
                f"< {config.min_per_class_recall_at_op:.2f}"
            )
    if ece > config.max_ece:
        failures.append(f"ece={ece:.4f} > {config.max_ece:.2f}")
    if open_set_fmr > config.max_open_set_fmr_at_op:
        failures.append(
            f"open_set_fmr_at_{op:.1f}={open_set_fmr:.4f} > {config.max_open_set_fmr_at_op:.2f}"
        )
    if failures:
        raise TrainingGateError("Acceptance gates failed: " + "; ".join(failures))

    trained_days_iso = [d.isoformat() for d in sorted_pre_days]
    eval_days_iso = [d.isoformat() for d in sorted(eval_day_set)]
    artifact_id = _deterministic_artifact_id(
        feature_version=feature_version,
        classes=retained_classes,
        trained_days=trained_days_iso,
        eval_days=eval_days_iso,
        seed=config.seed,
        weights=weights,
        bias=bias,
        temperature=temperature,
    )
    artifact = HouseholdClassifierArtifact(
        schema_version=1,
        artifact_id=artifact_id,
        feature_version=feature_version,
        classes=list(retained_classes),
        weights=[[float(v) for v in row] for row in weights],
        bias=[float(v) for v in bias],
        temperature=float(temperature),
        trained_days=trained_days_iso,
        eval_days=eval_days_iso,
        metrics=HouseholdClassifierArtifactMetrics(
            macro_top1_at_0_8=macro_top1_at_op,
            per_class_recall_at_0_8=per_class_recall_at_op,
            ece=ece,
            open_set_false_match_rate_at_0_8=open_set_fmr,
            selected_c=best_c,
            cv_log_loss=cv_log_loss,
            eval_positive_samples=len(eval_pos_samples),
            eval_negative_samples=len(eval_neg_samples),
        ),
        tool_version=TOOL_VERSION,
        seed=config.seed,
        created_at=created_at,
    )
    return TrainingResult(
        artifact=artifact,
        omitted_classes=tuple(omitted_classes),
        audit_wilson_lower=audit_lower,
    )


async def _load_samples_from_db(
    dsn: str,
    *,
    feature_version: str,
    tz_name: str,
    include_operator_room_gallery: bool = True,
) -> list[AppearanceTrainingSample]:
    """Load non-revoked training samples (and operator-approved room gallery rows) read-only."""
    pool = await asyncpg.create_pool(
        dsn,
        min_size=1,
        max_size=1,
        server_settings={"default_transaction_read_only": "on"},
    )
    tz = ZoneInfo(tz_name)
    try:
        repo = PostgresAppearanceSampleRepository(pool)
        samples = await repo.list_samples(
            feature_version=feature_version,
            include_revoked=False,
        )
        if include_operator_room_gallery:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, seen_at, identity_id, camera_id, orientation,
                           quality, model_version, embedding::text AS embedding,
                           seed_association_id
                    FROM continuous_tracking.reid_gallery
                    WHERE state = 'operator_verified'
                      AND seed_source = 'wearable_room_attested'
                      AND model_version = $1
                      AND embedding IS NOT NULL
                      AND identity_id IS NOT NULL
                    ORDER BY seen_at ASC, id ASC
                    """,
                    feature_version,
                )
            for row in rows:
                emb = _pgvector_to_tuple(row["embedding"])
                if not emb:
                    continue
                seen_at: datetime = row["seen_at"]
                assoc_raw = row["seed_association_id"]
                samples.append(
                    AppearanceTrainingSample(
                        sample_id=str(row["id"]),
                        observed_at=seen_at,
                        day=seen_at.astimezone(tz).date(),
                        identity_id=str(row["identity_id"]),
                        label_source=AppearanceLabelSource.OPERATOR_VERIFIED,
                        camera_id=str(row["camera_id"] or ""),
                        orientation=OrientationBin(int(row["orientation"])),
                        quality=min(1.0, max(0.0, float(row["quality"]))),
                        feature_version=str(row["model_version"]),
                        embedding=emb,
                        seed_association_id=str(assoc_raw) if assoc_raw is not None else None,
                    )
                )
        return samples
    finally:
        await pool.close()


def _dsn_from_env() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise SystemExit("DATABASE_URL environment variable is not set")
    return raw.replace("postgresql+asyncpg://", "postgresql://")


async def _run_cli(args: argparse.Namespace) -> int:
    audit_path = Path(args.association_audit)
    if not audit_path.is_file():
        raise SystemExit(f"Association audit file not found: {audit_path}")
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit_data, dict):
        raise SystemExit("Association audit file must be a JSON object")

    samples = await _load_samples_from_db(
        _dsn_from_env(),
        feature_version=args.feature_version,
        tz_name=args.timezone,
        include_operator_room_gallery=not args.no_operator_room_gallery,
    )
    cfg = TrainerConfig(
        min_days_per_class=args.min_days_per_class,
        min_samples_per_class=args.min_samples_per_class,
        eval_days_count=args.eval_days,
        seed=args.seed,
    )
    created_at = args.created_at or datetime.now(UTC).isoformat()
    try:
        result = train_household_classifier(
            samples,
            feature_version=args.feature_version,
            association_audit=audit_data,
            config=cfg,
            created_at=created_at,
        )
    except TrainingGateError as exc:
        sys.stderr.write(f"TRAINING REFUSED: {exc}\n")
        return 1

    output_json = json.dumps(result.artifact.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json + "\n", encoding="utf-8")
        sys.stdout.write(
            f"Wrote {out_path} (artifact_id={result.artifact.artifact_id}, "
            f"classes={len(result.artifact.classes)}, "
            f"macro_top1@0.8={result.artifact.metrics.macro_top1_at_0_8:.4f}, "
            f"ece={result.artifact.metrics.ece:.4f}, "
            f"open_set_fmr@0.8={result.artifact.metrics.open_set_false_match_rate_at_0_8:.4f})\n"
        )
    else:
        sys.stdout.write(output_json + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--feature-version", default="reid-solider")
    parser.add_argument(
        "--association-audit",
        required=True,
        help="Path to JSON file with wearable spot-check sample_size & correct_count",
    )
    parser.add_argument("--output", default=None, help="Output JSON artifact path")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--min-days-per-class", type=int, default=3)
    parser.add_argument("--min-samples-per-class", type=int, default=300)
    parser.add_argument("--eval-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--created-at", default="")
    parser.add_argument(
        "--no-operator-room-gallery",
        action="store_true",
        help="Do not import operator_verified wearable_room_attested rows from reid_gallery",
    )
    return asyncio.run(_run_cli(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
