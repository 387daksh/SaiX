import os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FRAME_DIR = os.path.join(ROOT, "data", "test_frames")
LANE_DIR = os.path.join(ROOT, "data", "outputs", "yolop", "lane")
DRIVE_DIR = os.path.join(ROOT, "data", "outputs", "yolop", "drive")

OUT_DIR = os.path.join(
    ROOT, "data", "outputs", "final", "lane_geometry"
)

CSV_PATH = os.path.join(
    OUT_DIR, "lane_width_pixels.csv"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

MIN_COMPONENT_AREA = 40

# Ignore sky / buildings.
# Lane geometry normally starts below this.
ROI_TOP = 0.45

# Don't use pixels extremely close to image edges.
EDGE_MARGIN = 30

# Fixed image row used for the first pixel-width measurement.
MEASURE_Y = 750

# Minimum number of points required for a curve.
MIN_POINTS = 80


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def clean_mask(mask):
    """
    Remove isolated noise and tiny connected components.
    """

    mask = (mask > 0).astype(np.uint8) * 255

    # Small morphological opening
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Connect nearby lane pixels
    kernel2 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel2,
        iterations=1
    )

    # Connected component filtering
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 255

    return cleaned


def extract_lane_points(lane_mask, drive_mask):
    """
    Extract lane pixels only inside the drivable road region.
    """

    h, w = lane_mask.shape

    # ROI
    roi = np.zeros_like(lane_mask)
    roi[int(h * ROI_TOP):, :] = 255

    # Drivable area
    if drive_mask is not None:
        drive = drive_mask > 0
    else:
        drive = np.ones_like(lane_mask, dtype=bool)

    lane = lane_mask > 0

    valid = lane & drive & (roi > 0)

    ys, xs = np.where(valid)

    # Remove extreme image edges
    keep = (
        (xs > EDGE_MARGIN) &
        (xs < w - EDGE_MARGIN)
    )

    return xs[keep], ys[keep]


def robust_polyfit(xs, ys, degree=2):
    """
    Fit x = f(y) using iterative residual filtering.
    """

    if len(xs) < MIN_POINTS:
        return None

    # Initial fit
    coeff = np.polyfit(ys, xs, degree)

    for _ in range(5):

        predicted = np.polyval(coeff, ys)

        residual = np.abs(xs - predicted)

        # Robust threshold using MAD
        median = np.median(residual)
        mad = np.median(np.abs(residual - median)) + 1e-6

        threshold = max(
            12.0,
            median + 3.5 * mad
        )

        keep = residual < threshold

        if keep.sum() < MIN_POINTS:
            break

        coeff_new = np.polyfit(
            ys[keep],
            xs[keep],
            degree
        )

        if np.max(np.abs(coeff_new - coeff)) < 1e-5:
            coeff = coeff_new
            break

        coeff = coeff_new

    return coeff


def evaluate_curve(coeff, y):
    return float(np.polyval(coeff, y))


def sanity_check(left_coeff, right_coeff, h, w):
    """
    Check whether the two fitted curves look geometrically valid.
    """

    if left_coeff is None or right_coeff is None:
        return False, "missing_boundary"

    sample_y = np.linspace(
        int(h * 0.55),
        min(h - 10, 950),
        8
    )

    widths = []

    for y in sample_y:

        xl = evaluate_curve(left_coeff, y)
        xr = evaluate_curve(right_coeff, y)

        width = xr - xl
        widths.append(width)

        if xl >= xr:
            return False, "boundaries_cross"

        if xl < -100 or xr > w + 100:
            return False, "outside_image"

    widths = np.asarray(widths)

    # Width should generally increase towards camera
    # because of perspective, but shouldn't explode.
    if np.any(widths < 40):
        return False, "width_too_small"

    if np.max(widths) > w * 1.25:
        return False, "width_too_large"

    # Avoid wildly unstable fits
    ratio = np.max(widths) / max(np.min(widths), 1)

    if ratio > 8:
        return False, "unstable_width"

    return True, "ok"


# ---------------------------------------------------------
# CSV
# ---------------------------------------------------------

csv_rows = []

files = sorted(
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith(".jpg")
)

print(f"Found {len(files)} frames.")


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

for frame_name in files:

    stem = os.path.splitext(frame_name)[0]

    frame_path = os.path.join(
        FRAME_DIR,
        frame_name
    )

    lane_path = os.path.join(
        LANE_DIR,
        stem + ".png"
    )

    drive_path = os.path.join(
        DRIVE_DIR,
        stem + ".png"
    )

    if not os.path.exists(lane_path):
        print(f"[{stem}] missing lane mask")
        continue

    frame = cv2.imread(frame_path)
    lane_mask = cv2.imread(
        lane_path,
        cv2.IMREAD_GRAYSCALE
    )

    drive_mask = None

    if os.path.exists(drive_path):
        drive_mask = cv2.imread(
            drive_path,
            cv2.IMREAD_GRAYSCALE
        )

    if frame is None or lane_mask is None:
        continue

    h, w = frame.shape[:2]

    # Ensure masks match frame
    if lane_mask.shape != (h, w):
        lane_mask = cv2.resize(
            lane_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    if drive_mask is not None:
        if drive_mask.shape != (h, w):
            drive_mask = cv2.resize(
                drive_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

    # -----------------------------------------------------
    # Clean masks
    # -----------------------------------------------------

    lane_mask = clean_mask(lane_mask)

    if drive_mask is not None:
        drive_mask = (drive_mask > 0).astype(np.uint8) * 255

    # -----------------------------------------------------
    # Extract candidate lane pixels
    # -----------------------------------------------------

    xs, ys = extract_lane_points(
        lane_mask,
        drive_mask
    )

    if len(xs) < MIN_POINTS:
        print(f"[{stem}] insufficient lane pixels")
        continue

    # -----------------------------------------------------
    # Split into left/right candidates
    # -----------------------------------------------------

    center_x = w / 2.0

    left_keep = xs < center_x
    right_keep = xs >= center_x

    left_x = xs[left_keep]
    left_y = ys[left_keep]

    right_x = xs[right_keep]
    right_y = ys[right_keep]

    left_coeff = robust_polyfit(
        left_x,
        left_y
    )

    right_coeff = robust_polyfit(
        right_x,
        right_y
    )

    valid, reason = sanity_check(
        left_coeff,
        right_coeff,
        h,
        w
    )

    # -----------------------------------------------------
    # Draw visualization
    # -----------------------------------------------------

    vis = frame.copy()

    # Draw ROI line
    roi_y = int(h * ROI_TOP)

    cv2.line(
        vis,
        (0, roi_y),
        (w, roi_y),
        (255, 255, 0),
        2
    )

    # -----------------------------------------------------
    # Draw left curve
    # -----------------------------------------------------

    if left_coeff is not None:

        curve_ys = np.arange(
            roi_y,
            h,
            3
        )

        curve_xs = [
            evaluate_curve(left_coeff, y)
            for y in curve_ys
        ]

        pts = []

        for x, y in zip(curve_xs, curve_ys):
            if 0 <= x < w:
                pts.append([int(x), int(y)])

        if len(pts) > 1:
            pts = np.asarray(pts)

            cv2.polylines(
                vis,
                [pts],
                False,
                (255, 0, 0),
                5
            )

    # -----------------------------------------------------
    # Draw right curve
    # -----------------------------------------------------

    if right_coeff is not None:

        curve_ys = np.arange(
            roi_y,
            h,
            3
        )

        curve_xs = [
            evaluate_curve(right_coeff, y)
            for y in curve_ys
        ]

        pts = []

        for x, y in zip(curve_xs, curve_ys):
            if 0 <= x < w:
                pts.append([int(x), int(y)])

        if len(pts) > 1:
            pts = np.asarray(pts)

            cv2.polylines(
                vis,
                [pts],
                False,
                (0, 0, 255),
                5
            )

    # -----------------------------------------------------
    # Pixel width at measurement row
    # -----------------------------------------------------

    lane_width_px = None

    if valid:

        xl = evaluate_curve(
            left_coeff,
            MEASURE_Y
        )

        xr = evaluate_curve(
            right_coeff,
            MEASURE_Y
        )

        lane_width_px = xr - xl

        # Draw measurement line
        if 0 <= xl < w and 0 <= xr < w:

            cv2.line(
                vis,
                (int(xl), MEASURE_Y),
                (int(xr), MEASURE_Y),
                (0, 255, 0),
                5
            )

            cv2.circle(
                vis,
                (int(xl), MEASURE_Y),
                8,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                vis,
                (int(xr), MEASURE_Y),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                vis,
                f"Lane width = {lane_width_px:.1f} px",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                vis,
                f"Measurement row = {MEASURE_Y}",
                (50, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    else:

        cv2.putText(
            vis,
            f"INVALID: {reason}",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = os.path.join(
        OUT_DIR,
        stem + ".jpg"
    )

    cv2.imwrite(
        output_path,
        vis
    )

    csv_rows.append(
        (
            stem,
            lane_width_px if lane_width_px is not None else "",
            reason
        )
    )

    print(
        f"[{stem}] "
        f"pixels={len(xs):5d} | "
        f"width={lane_width_px if lane_width_px is not None else 'N/A'} | "
        f"{reason}"
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

with open(
    CSV_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "frame,lane_width_px,status\n"
    )

    for frame, width, status in csv_rows:

        f.write(
            f"{frame},{width},{status}\n"
        )

print()
print("==============================================")
print("DONE")
print("Visualizations:")
print(OUT_DIR)
print()
print("CSV:")
print(CSV_PATH)
print("==============================================")