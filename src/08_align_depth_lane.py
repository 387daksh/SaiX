import os
import cv2
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEPTH_FILE = os.path.join(
    ROOT, "data", "outputs", "depth_test", "dashcam_depths.npz"
)

LANE_DIR = os.path.join(
    ROOT, "data", "outputs", "yolop", "lane"
)

OUT_DIR = os.path.join(
    ROOT, "data", "outputs", "depth_lane_alignment"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load VDA depth
# ---------------------------------------------------------

data = np.load(DEPTH_FILE)
depths = data["depths"]

print("VDA depth shape:", depths.shape)

N, H, W = depths.shape

# ---------------------------------------------------------
# Process each frame
# ---------------------------------------------------------

for i in range(N):

    depth = depths[i]

    # YOLOP filename
    lane_path = os.path.join(LANE_DIR, f"{i:04d}.png")

    if not os.path.exists(lane_path):
        print(f"[WARNING] Missing lane mask: {lane_path}")
        continue

    # Read YOLOP lane mask
    lane = np.array(Image.open(lane_path))

    # Resize 1920x1080 -> 960x540
    lane_small = cv2.resize(
        lane,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    # Binary lane mask
    lane_binary = lane_small > 0

    # -----------------------------------------------------
    # Extract depth values at lane pixels
    # -----------------------------------------------------

    lane_depth = depth[lane_binary]

    if len(lane_depth) == 0:
        print(f"[{i:04d}] No lane pixels")
        continue

    # Robust statistics
    d_min = float(np.percentile(lane_depth, 5))
    d_med = float(np.median(lane_depth))
    d_max = float(np.percentile(lane_depth, 95))

    print(
        f"[{i:04d}] "
        f"lane_pixels={len(lane_depth):6d} | "
        f"depth P5={d_min:7.2f} | "
        f"median={d_med:7.2f} | "
        f"P95={d_max:7.2f}"
    )

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    # Normalize depth only for visualization
    d_vis = np.clip(depth, 0, np.percentile(depth, 99))

    d_vis = (
        (d_vis - d_vis.min())
        / (d_vis.max() - d_vis.min() + 1e-8)
        * 255
    ).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        d_vis,
        cv2.COLORMAP_TURBO
    )

    # Highlight lane pixels
    depth_color[lane_binary] = (0, 255, 0)

    # Save
    out_path = os.path.join(
        OUT_DIR,
        f"{i:04d}.png"
    )

    cv2.imwrite(out_path, depth_color)

print()
print("Done.")
print("Output:", OUT_DIR)