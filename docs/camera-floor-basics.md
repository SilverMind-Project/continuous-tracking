# Camera to Floor Basics

CTS tracks people on a shared floor plane. Each camera detection starts as a pixel-space bounding
box, and the tracker uses the bbox bottom-centre as the footpoint:

```text
footpoint_px = ((x_min + x_max) / 2, y_max)
```

For calibrated cameras, the camera homography maps that raw pixel point to floor-plan metres. The
domain `FloorPoint` stores the same position in millimetres:

```text
[x_m, y_m, w] = H * [footpoint_x_px, footpoint_y_px, 1]
floor_x_mm = round((x_m / w) * 1000)
floor_y_mm = round((y_m / w) * 1000)
```

For uncalibrated cameras, the live pipeline creates a synthetic per-camera virtual tile so tracking
can still create and maintain a PersonHypothesis. Those points are explicitly marked
`calibrated=False` and do not get homography covariance.

Footpoint reliability is separate from calibration. A calibrated homography can still receive a
bad footpoint when the detector bbox is clipped by the image edge or pose says both ankles are not
visible. In that case CTS keeps the observation but marks `footpoint_reliable=False`, which inflates
the pixel covariance before projection into floor metres.
