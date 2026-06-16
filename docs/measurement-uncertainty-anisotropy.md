# Measurement Uncertainty & Anisotropy

CTS projects each detection's footpoint from image pixels into the shared floor-plan frame with a
camera homography. The detector's pixel error is roughly circular in the image, but the homography
can stretch that error unevenly on the floor. Oblique views therefore produce an elongated
floor-plane error ellipse.

The per-observation model is:

```text
R_obs = J * Sigma_px * J^T + R_cal + epsilon * I
```

Where:

- `R_obs` is the full single-camera measurement covariance in floor-plan m^2.
- `J` is the analytic homography Jacobian at the bbox bottom-centre footpoint, in m/px.
- `Sigma_px` is detector footpoint covariance in px^2.
- `R_cal = (k_cal * floor_residual_m)^2 * I` is the systematic calibration term in m^2.
- `epsilon * I` is a small numeric floor that keeps `R_obs` invertible.

Footpoint reliability controls the scale of `Sigma_px`. A bbox clipped against the bottom or sides
of the image means the bbox bottom-centre is probably the image boundary, not the person's floor
contact point. If pose is available and both ankles have low visibility, the feet are likely
occluded. CTS keeps those observations but marks `footpoint_reliable=False`, which steeply inflates
the pixel covariance before applying `J`.

The random term has the expected units:

```text
(m/px) * px^2 * (m/px) = m^2
```

The calibration term is kept separate because it is systematic. Cross-camera fusion can shrink
the random term with more independent camera observations, but it must not fuse away a fixed
camera calibration bias.

```text
Image pixels                         Floor-plan metres

    bbox
  +------+
  |      |
  |      |
  +--x---+  footpoint_px                camera
      \                                  o
       \                                  \
        \ homography                       \ viewing ray
         \                                  \
          v                                  \        major axis
     circular Sigma_px                        \      .--------.
          (px^2)                               \    /          \
                                                \   \          /
                                                 \   '--------'
                                                  x footpoint_m

                                             elongated R_obs ellipse (m^2)
```
