# Agitation detection: model upgrade path

This document describes the planned upgrade from the heuristic `agitation_index`
signal (M4, experimental) to a skeleton-action recognition model. It defines
the candidate architecture, data collection plan, serving budget, and the
concrete promotion criteria that trigger the investment.

This is a decision-ready note for architect review. No new code is introduced
here; the code lives in the M4 detector and the Part A feedback loop.

---

## 1. Candidate: skeleton-based action recognition

The phase-2 model is a **skeleton-action recognition network** consuming the
COCO-17 keypoint stream already produced by RTMPose in the live pipeline. Two
well-validated architectures fit the constraints:

| Architecture | Parameters | Notes |
| --- | --- | --- |
| PoseC3D | ~3 M (S variant) | 3D-Conv over stacked heatmaps; strong on small-scale repetitive motions |
| ST-GCN | ~3 M | Spatial-temporal graph convolution over skeleton graphs |

Both are available under Apache-2.0 from
[MMAction2](https://github.com/open-mmlab/mmaction2). The input is a sequence
of COCO-17 `(x, y, confidence)` keypoint frames for a fixed window. The output
is a softmax over action classes; fine-tuning adds an `agitation` class.

**Export path.** The selected model is exported to ONNX (FP32 first) using
`mmdeploy` or `torch.onnx.export`. The ONNX graph is placed in
`triton-models/agitation-action/1/model.onnx` and served through the existing
Triton instance. INT8 quantization follows the methodology in
[`docs/hardware/model-quantization.md`](../silvermind-project.github.io/docs/hardware/model-quantization.md)
if the FP32 model exceeds the memory budget (see section 3).

**Integration point.** The model runs in the `DementiaSignalWorker` per 30-minute
window per identity, replacing the heuristic feature computation inside
`agitation_index_signal.py`. The COCO-17 keypoint archive already accumulates
in the orchestrator's trajectory store.

---

## 2. Data plan

### Collection

Keypoint sequences for windows that received caregiver feedback through the
Part A loop (M4.3) become the fine-tuning training set.

| Field | Detail |
| --- | --- |
| Source | `cts_dementia_signals` rows with `evidence_grade = 'experimental'` and `feedback IS NOT NULL` |
| Input shape | `[T, 17, 3]` where T = 30 min * ~2 frames/s = ~3600 frames, subsampled to 64 |
| Labels | `feedback = 'accurate'` -> positive; `feedback = 'inaccurate'` -> negative; `feedback = 'unsure'` -> excluded |
| Storage | Keypoint tensors saved to MinIO under `agitation-training/{signal_id}.npy` at acknowledgment time, retained for 18 months |
| Privacy | Keypoints only. No raw images, video, or RGB frames leave the device. Consistent with the privacy-protecting behaviours-of-risk detection approach (Padilla et al., Biomedical Engineering Online 2023). |

### Retention and deletion

Training data linked to a person is deleted when:
- the household member record is deleted from Cognitive Companion; or
- the operator runs the `cts-db purge-training-data` CLI command (planned, not yet implemented).

---

## 3. Serving budget

### Memory constraint

The Jetson Orin Nano Super has 8 GB of unified memory shared across all
inference engines, the OS, and application processes. The current worst-case
detector demand table (from
[`docs/hardware/jetson-cts.md`](../silvermind-project.github.io/docs/hardware/jetson-cts.md))
shows the existing engines consume:

| Model | Parameters | Loaded plan size |
| --- | --- | --- |
| YOLO26L (person detector) | 135 M | 99.8 MB |
| RTMPose-m | 46 M | 54.4 MB |
| SOLIDER ReID | 46 M | 113.8 MB |
| ArcFace + 5 Buffalo_L | ~10 M each | ~200 MB total |
| **Available headroom** | | **~400 MB** |

A 3 M-parameter skeleton-action model in FP32 ONNX is approximately **12 MB**.
The TensorRT plan will be larger (typically 2-4x) but stays well inside the
headroom. No INT8 quantization is required to fit the 8 GB budget.

### Throughput cost

The agitation signal runs once per 30-minute window per tracked identity, not
per-frame. At 8 cameras and 4 identities, this is at most 4 inferences per 30
minutes - a throughput cost below 0.01 inferences per second. Latency per
inference (1 window = 64 frames of 17 keypoints) is estimated at 2-10 ms on
the Jetson; acceptable for a batch-window job.

**Memory is the constraint, not throughput.**

---

## 4. Promotion criteria

The heuristic ships as `evidence_grade = "experimental"` and the feedback loop
accumulates labeled windows. The transition to the skeleton-action model is
triggered when **both** of the following conditions are met:

| Condition | Threshold |
| --- | --- |
| Labeled windows (feedback = accurate or inaccurate) | >= 200 |
| Heuristic precision proxy (accurate / (accurate + inaccurate), rolling 60 days) | < 0.60 |

The precision proxy query (run monthly by a maintainer):

```sql
SELECT
    DATE_TRUNC('month', acknowledged_at) AS month,
    COUNT(*) FILTER (WHERE feedback = 'accurate') AS accurate,
    COUNT(*) FILTER (WHERE feedback = 'inaccurate') AS inaccurate,
    ROUND(
        COUNT(*) FILTER (WHERE feedback = 'accurate')::numeric /
        NULLIF(
            COUNT(*) FILTER (WHERE feedback IN ('accurate', 'inaccurate')),
        0),
        3
    ) AS precision_proxy
FROM cts_dementia_signals
WHERE signal_type = 'agitation_index'
  AND feedback IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC;
```

If precision >= 0.60 with >= 200 labeled windows, the heuristic is performing
well enough to defer the model investment. If precision < 0.60 with < 200
windows, wait for more data before committing.

### Alternative trigger

If a caregiver or clinical partner supplies a pre-labeled dataset (>= 500
windows, mixed demographics) before the internal threshold is reached, the
fine-tuning may proceed immediately using that dataset in combination with
internal labels.

---

## 5. Non-goals

The following are explicitly out of scope regardless of precision results:

- **No raw-video action models.** All inference stays on skeleton keypoints. No
  RGB, depth, or video frames are used for agitation classification.
- **No cloud training on household data.** Fine-tuning runs fully on-device
  or on operator-controlled infrastructure. No household keypoint sequences are
  sent to external cloud training services.
- **No cross-household pre-training using real resident data.** A publicly
  available skeleton action dataset (NTU RGB+D or similar, Apache-2.0 or
  CC-BY compatible) may be used for initialisation; household data is used only
  for fine-tuning.
- **No new Triton model infrastructure.** The model slots into the existing
  Triton instance and the existing DementiaSignalWorker calling convention.
