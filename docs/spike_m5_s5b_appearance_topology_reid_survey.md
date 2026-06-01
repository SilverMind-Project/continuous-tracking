# Spike S5b: Appearance-Topology ReID Literature Scan

**Status**: Evaluation memo only — not a committed feature.

## Objective

Survey current multi-camera multi-target tracking (MCMT) and camera-topology
ReID approaches that fit on-prem, GPU-modest constraints, and map them to
whether they can be expressed with the approved stack (`numpy`, `scipy`,
`opencv-python-headless`, `shapely`, `asyncpg`, no new heavy models).

## Motivation

The current cross-camera association uses:
- Appearance: SOLIDER-REID embeddings with multi-view prototypes
- Topology: learned camera-adjacency transit-time distributions
- Identity: face-anchored Bayesian posterior

This spike surveys whether any additional computer-vision technique from the
literature could improve cross-camera accuracy without adding a heavy new model.

## Search Dimensions

1. **Camera-topology-aware ReID**: Methods that learn or exploit camera
   adjacency for ReID matching.
2. **Multi-camera multi-target tracking (MCMT)**: Global optimisation
   approaches for track-to-track association.
3. **Temporal attention / graph neural networks for handoff detection**:
   Learning-based approaches to detect cross-camera handoffs.
4. **Unsupervised domain adaptation for ReID**: Improving ReID across
   camera-specific appearance shifts.

## Evaluation Criteria

For each candidate:
- **Expressibility**: Can it be implemented with the approved stack? If not,
  what new dependency is required and what is its cost (GPU memory, inference
  latency, model size)?
- **Accuracy gain**: What improvement does the paper claim over a baseline
  comparable to ours (appearance + simple topology)?
- **On-prem feasibility**: Does it require a data centre GPU, cloud API, or
  large labelled training set?
- **Integration complexity**: How invasive would the change be to the current
  pipeline?

## Approved Stack

`numpy`, `scipy`, `opencv-python-headless`, `shapely`, `asyncpg`, `redis`,
`pydantic`, `fastapi`, `structlog`, `prometheus-client`, `protobuf`.

Any candidate requiring a new dependency must be flagged with its full
dependency cost so the product owner can decide explicitly. Do not adopt a
remembered SOTA model without explicit owner sign-off.

## Results (placeholder)

| Approach | Expressible? | Accuracy claim | New dep? | On-prem? | Recommendation |
|----------|-------------|----------------|----------|----------|----------------|
| *To be filled* | | | | | |

## Recommendation (placeholder)

To be filled after survey.
