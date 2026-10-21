import os
import cv2
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FRAME_DIR = os.path.join(ROOT, "data", "test_frames")
LANE_DIR = os.path.join(ROOT, "data", "outputs", "yolop", "lane")
OUT_DIR = os.path.join(ROOT, "data", "outputs", "lane_original")

os.makedirs(OUT_DIR, exist_ok=True)

for i in range(10):

    frame_path = os.path.join(FRAME_DIR, f"{i:04d}.jpg")
    lane_path = os.path.join(LANE_DIR, f"{i:04d}.png")

    if not os.path.exists(frame_path):
        print("Missing:", frame_path)
        continue

    if not os.path.exists(lane_path):
        print("Missing:", lane_path)
        continue

    frame = cv2.imread(frame_path)
    lane = np.array(Image.open(lane_path))

    # Binary YOLOP lane mask
    mask = lane > 0

    # Green overlay
    overlay = frame.copy()
    overlay[mask] = (0, 255, 0)

    # Blend with original
    result = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    # Draw actual detected pixels more clearly
    result[mask] = (
        0.5 * result[mask] +
        0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    out = os.path.join(OUT_DIR, f"{i:04d}.jpg")
    cv2.imwrite(out, result)

    print("Saved:", out)

print("Done.")