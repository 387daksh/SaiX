# Calibration

`camera.json` is intentionally a placeholder. Do not treat it as ground truth.

For serious metric accuracy, calibrate the actual dashcam intrinsics/distortion with a checkerboard, then estimate a road-plane homography using physically measured points.

Recommended coordinate system for the homography:
- X = lateral position in metres; right positive
- Z = forward distance in metres

At the end, the system samples both boundaries at `Z = measurement_distance` and computes `width = X_right - X_left`.
