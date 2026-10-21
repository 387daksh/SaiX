import os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DRIVE_DIR = os.path.join(ROOT, "data", "outputs", "yolop", "drive")
FRAME_DIR = os.path.join(ROOT, "data", "test_frames")
OUT_DIR = os.path.join(ROOT, "data", "outputs", "final", "road_pixel_width")

os.makedirs(OUT_DIR, exist_ok=True)


def find_width(mask, y):
    """
    Find the leftmost and rightmost drivable pixels
    on horizontal row y.
    """

    row = mask[y]

    xs = np.where(row > 0)[0]

    if len(xs) < 2:
        return None

    left = int(xs[0])
    right = int(xs[-1])

    return left, right, right - left


# Choose several image rows.
# These are image coordinates, not real-world distances.
Y_VALUES = [650, 700, 750, 800, 850, 900]


for filename in sorted(os.listdir(DRIVE_DIR)):

    if not filename.endswith(".png"):
        continue

    drive_path = os.path.join(DRIVE_DIR, filename)

    # Corresponding original frame
    frame_name = os.path.splitext(filename)[0] + ".jpg"
    frame_path = os.path.join(FRAME_DIR, frame_name)

    if not os.path.exists(frame_path):
        continue

    mask = cv2.imread(drive_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(frame_path)

    if mask is None or frame is None:
        continue

    h, w = mask.shape

    # Make sure frame and mask have same size
    if frame.shape[:2] != mask.shape:
        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    measurements = []

    for y in Y_VALUES:

        if y >= h:
            continue

        result = find_width(mask, y)

        if result is None:
            continue

        left, right, width = result

        measurements.append((y, left, right, width))

        # Draw measurement
        cv2.line(
            frame,
            (left, y),
            (right, y),
            (0, 255, 0),
            3
        )

        cv2.circle(
            frame,
            (left, y),
            6,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (right, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"{width}px",
            ((left + right) // 2 - 50, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # Print measurements
    print(f"\n{filename}")

    for y, left, right, width in measurements:
        print(
            f"  y={y:4d} | "
            f"left={left:4d} | "
            f"right={right:4d} | "
            f"width={width:4d} px"
        )

    # Save visualization
    out_path = os.path.join(
        OUT_DIR,
        filename
    )

    cv2.imwrite(out_path, frame)


print("\nDone.")
print("Results:", OUT_DIR)