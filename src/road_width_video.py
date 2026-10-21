"""
road_width_video.py  v4  —  Ground-up rewrite
================================================

Core insight:
  YOLOP produces a solid FILLED drivable-area mask.
  The LEFT and RIGHT EDGES of that mask ARE the road boundaries.
  We extract them row-by-row (one boundary point per row = hundreds of
  clean, reliable points) and fit polynomials to those — not to sparse,
  noisy lane-line pixels.

  Lane lines are used only as an optional REFINEMENT when both sides are
  clearly detected and pass a sanity check.

Pipeline per sampled frame:
  1. YOLOP  →  drive_mask (filled road) + lane_mask (thin lines)
  2. Clean drive_mask  →  largest road contour / flood fill
  3. Row-by-row boundary extraction  →  left_pts, right_pts
  4. RANSAC polynomial fit on each boundary set
  5. Sanity check + multi-row median width
  6. Optional metric projection via homography
  7. Kalman-style temporal smoother + holdout (no flickering)
  8. Rich annotated visualization

Usage (YOLOP env):
  python src/road_width_video.py
  python src/road_width_video.py --fps 5 --clip-duration 300
  python src/road_width_video.py --no-drive-overlay
"""

import argparse, csv, json, os, time, warnings
import cv2, numpy as np

warnings.filterwarnings('ignore', message='.*Polyfit may be poorly conditioned.*')
try:
    warnings.simplefilter('ignore', np.exceptions.RankWarning)
except AttributeError:
    pass

# ────────────────────────────────────────────────────────────────
# 1.  YOLOP  (unchanged, proven working)
# ────────────────────────────────────────────────────────────────

def load_yolop(device):
    import torch
    model = torch.hub.load("hustvl/yolop", "yolop",
                           pretrained=True, trust_repo=True)
    return model.to(device).eval()


def run_yolop(model, frame, device):
    """Returns lane_mask, drive_mask — uint8 {0,255}, original resolution."""
    import torch
    oh, ow = frame.shape[:2]

    # letterbox
    r   = 640 / max(oh, ow)
    nw  = int(round(ow * r)); nh = int(round(oh * r))
    dw  = (640 - nw) // 2;   dh = (640 - nh) // 2
    pad = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad[dh:dh+nh, dw:dw+nw] = cv2.resize(frame, (nw, nh))

    x = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    x = (x - np.array([0.485,0.456,0.406],np.float32)) \
      / np.array([0.229,0.224,0.225],np.float32)
    t = torch.from_numpy(x.transpose(2,0,1)[None]).float().to(device)

    with torch.no_grad():
        out = model(t)

    def decode(seg):
        seg = seg[..., dh:dh+nh, dw:dw+nw]
        m   = (torch.argmax(seg,1)[0].cpu().numpy().astype(np.uint8)) * 255
        return cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)

    return decode(out[2]), decode(out[1])   # lane_mask, drive_mask


# ────────────────────────────────────────────────────────────────
# 2.  Drive-mask cleaning
# ────────────────────────────────────────────────────────────────

def clean_drive_mask(drive, h, w, roi_top=0.40):
    """
    Morphologically clean the drive mask, keep only the LARGEST connected
    component (the road the vehicle is on), ignore sky and upper image.
    """
    m = (drive > 0).astype(np.uint8) * 255

    # Ignore top of frame (sky / buildings / hood)
    m[:int(h * roi_top), :] = 0

    # Close small holes (shadows, road markings)
    k = np.ones((21, 21), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((7,7), np.uint8))

    # Keep only the largest blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return m
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == biggest).astype(np.uint8) * 255)


# ────────────────────────────────────────────────────────────────
# 3.  Road boundary extraction from the drive mask
# ────────────────────────────────────────────────────────────────

def extract_road_boundaries(drive_clean, h, w, roi_top=0.40):
    """
    For every row in the ROI, record the leftmost and rightmost
    drive-mask pixel as boundary points.
    Returns:
      (lxs, lys)  — left  boundary points (arrays of float)
      (rxs, rys)  — right boundary points
    """
    roi_y = int(h * roi_top)
    lxs, lys, rxs, rys = [], [], [], []

    for y in range(roi_y, h):
        row = drive_clean[y]
        xs  = np.where(row > 0)[0]
        if len(xs) < 40:          # row too sparse → likely noise
            continue
        span = int(xs[-1]) - int(xs[0])
        if span < w * 0.08:       # road narrower than 8 % of frame → skip
            continue
        lxs.append(float(xs[0]));  lys.append(float(y))
        rxs.append(float(xs[-1])); rys.append(float(y))

    return (np.array(lxs), np.array(lys)), (np.array(rxs), np.array(rys))


# ────────────────────────────────────────────────────────────────
# 4.  RANSAC robust polynomial fit  x = f(y)
# ────────────────────────────────────────────────────────────────

MIN_BND_PTS = 15     # minimum boundary points to attempt fit
RANSAC_ITER = 60     # RANSAC iterations


def ransac_polyfit(xs, ys, deg=2, n_iter=RANSAC_ITER, inlier_thresh=12.0):
    """
    Fit a polynomial x = poly(y) using RANSAC for robustness against
    outliers caused by pedestrians, parked cars, shadows, etc.
    Returns coefficient array or None.
    """
    n = len(xs)
    if n < MIN_BND_PTS:
        return None

    # Use degree 1 if very few points to avoid wild extrapolation
    deg = min(deg, 1 if n < 40 else 2)
    k   = deg + 2       # minimum sample size for RANSAC

    rng        = np.random.default_rng(0)
    best_c     = None
    best_count = 0

    for _ in range(n_iter):
        idx = rng.choice(n, k, replace=False)
        try:
            c = np.polyfit(ys[idx], xs[idx], deg)
        except np.linalg.LinAlgError:
            continue
        res = np.abs(xs - np.polyval(c, ys))
        cnt = int(np.sum(res < inlier_thresh))
        if cnt > best_count:
            best_count = cnt
            best_c     = c

    if best_c is None or best_count < MIN_BND_PTS:
        return None

    # Re-fit on all inliers
    mask = np.abs(xs - np.polyval(best_c, ys)) < inlier_thresh
    if mask.sum() < MIN_BND_PTS:
        return None
    try:
        return np.polyfit(ys[mask], xs[mask], deg)
    except np.linalg.LinAlgError:
        return None


def xv(c, y):
    """Evaluate polynomial at scalar y."""
    return float(np.polyval(c, y))


# ────────────────────────────────────────────────────────────────
# 5.  Sanity check
# ────────────────────────────────────────────────────────────────

def check_boundaries(lc, rc, h, w, roi_top=0.40):
    """
    Verify left/right boundaries make geometric sense.
    Returns (valid:bool, reason:str).
    """
    if lc is None or rc is None:
        return False, "fit_failed"

    ys = np.linspace(int(h * 0.55), min(h - 1, int(h * 0.93)), 10)
    widths = []
    for y in ys:
        xl, xr = xv(lc, y), xv(rc, y)
        if xl >= xr:
            return False, "crossed"
        if xl < -w * 0.15 or xr > w * 1.15:
            return False, "outside_frame"
        widths.append(xr - xl)

    ws = np.asarray(widths)
    if np.min(ws) < 20:
        return False, "too_narrow"
    if np.max(ws) > w * 1.2:
        return False, "too_wide"
    # Road shouldn't vary wildly in perspective-corrected width
    if np.max(ws) / max(np.min(ws), 1) > 12:
        return False, "unstable"
    return True, "ok"


# ────────────────────────────────────────────────────────────────
# 6.  Multi-row pixel width measurement
# ────────────────────────────────────────────────────────────────

MEAS_FRACS = [0.60, 0.65, 0.70, 0.75, 0.80]   # rows at these frame fractions


def measure_pixel_widths(lc, rc, h, w):
    """
    Measure the road boundary separation at multiple rows.
    Returns (median_width_px, left_x_at_main, right_x_at_main, main_y)
    where main_y is 0.70 * h.
    """
    main_y = int(h * 0.70)
    widths = []
    for frac in MEAS_FRACS:
        y  = int(h * frac)
        xl = xv(lc, y)
        xr = xv(rc, y)
        if 0 <= xl < xr <= w:
            widths.append(xr - xl)

    if not widths:
        return None, None, None, main_y

    med   = float(np.median(widths))
    lx_px = int(np.clip(xv(lc, main_y), 0, w - 1))
    rx_px = int(np.clip(xv(rc, main_y), 0, w - 1))
    return round(med, 1), lx_px, rx_px, main_y


# ────────────────────────────────────────────────────────────────
# 7.  Lane-line refinement  (optional, uses YOLOP lane mask)
# ────────────────────────────────────────────────────────────────

LANE_MIN_CC  = 50
LANE_MIN_PTS = 25


def refine_with_lane_lines(lane_mask, drive_clean, drive_lc, drive_rc, h, w):
    """
    Try to fit lane-line polynomials.  If both sides pass the sanity check
    AND the lane-derived boundaries are inside the drive-mask boundaries,
    return them as the refined result.  Otherwise return the drive boundaries.
    Returns (lc, rc, used_lanes:bool).
    """
    # Clean lane mask, restrict to drive area
    m = (lane_mask > 0).astype(np.uint8) * 255
    m = cv2.bitwise_and(m, drive_clean)
    m[:int(h * 0.40), :] = 0
    # Remove tiny blobs
    n, lbl, sts, _ = cv2.connectedComponentsWithStats(m, 8)
    cleaned = np.zeros_like(m)
    for i in range(1, n):
        if sts[i, cv2.CC_STAT_AREA] >= LANE_MIN_CC:
            cleaned[lbl == i] = 255

    ys_all, xs_all = np.where(cleaned > 0)
    if len(xs_all) < LANE_MIN_PTS * 2:
        return drive_lc, drive_rc, False

    # Ego centre from drive mask at 75 % height
    ref_row = drive_clean[int(h * 0.75)]
    cx_xs   = np.where(ref_row > 0)[0]
    cx      = float(cx_xs[0] + cx_xs[-1]) / 2 if len(cx_xs) > 20 else w / 2.

    l_m  = xs_all <  cx
    r_m  = xs_all >= cx
    ll_c = ransac_polyfit(xs_all[l_m].astype(float), ys_all[l_m].astype(float))
    rl_c = ransac_polyfit(xs_all[r_m].astype(float), ys_all[r_m].astype(float))

    valid, _ = check_boundaries(ll_c, rl_c, h, w)
    if not valid:
        return drive_lc, drive_rc, False

    # Accept only if lanes are INSIDE the drive-boundary (sanity)
    test_ys = np.linspace(int(h * 0.6), int(h * 0.9), 5)
    inside  = all(
        xv(drive_lc, y) - 20 <= xv(ll_c, y) <= xv(rl_c, y) <= xv(drive_rc, y) + 20
        for y in test_ys
    )
    if not inside:
        return drive_lc, drive_rc, False

    return ll_c, rl_c, True


# ────────────────────────────────────────────────────────────────
# 8.  Homography metric projection
# ────────────────────────────────────────────────────────────────

def load_homography(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return np.array(d["H_image_to_XZ"], dtype=np.float64)


def project_poly(c, H, h, roi_top=0.40):
    ys  = np.linspace(int(h * roi_top), h - 1, 80)
    xs  = np.array([xv(c, y) for y in ys])
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    q   = (H @ pts.T).T
    xz  = q[:, :2] / q[:, 2:3]
    return xz   # col0=X, col1=Z forward


def x_at_z(xz, z):
    xz = xz[np.isfinite(xz).all(axis=1)]
    xz = xz[np.argsort(xz[:, 1])]
    zmin, zmax = xz[:, 1].min(), xz[:, 1].max()
    if len(xz) < 3 or z < zmin or z > zmax:
        return None
    return float(np.interp(z, xz[:, 1], xz[:, 0]))


def metric_width_m(lc, rc, H, h, dist_m):
    xl = x_at_z(project_poly(lc, H, h), dist_m)
    xr = x_at_z(project_poly(rc, H, h), dist_m)
    if xl is None or xr is None:
        return None
    w = abs(xr - xl)
    return round(w, 3) if 0.5 <= w <= 12.0 else None


# ────────────────────────────────────────────────────────────────
# 9.  Temporal smoother  (EWA + holdout)
# ────────────────────────────────────────────────────────────────

class Smoother:
    """
    Exponentially weighted average with jump detection and holdout.
    - If a new value is within JUMP_THRESH of the current estimate, blend.
    - If it jumps too far, reset immediately (new road geometry).
    - On missing frames, hold last good value for up to HOLDOUT frames.
    """
    ALPHA       = 0.25     # EWA weight for new observation
    JUMP_THRESH = 150      # px jump that triggers a reset
    HOLDOUT     = 25       # frames to hold last value without new data

    def __init__(self):
        self.val   = None
        self.miss  = 0

    def update(self, new_val):
        """
        new_val: float or None.
        Returns (smoothed:float|None, is_live:bool).
        """
        if new_val is not None:
            if self.val is None:
                self.val = new_val
            elif abs(new_val - self.val) > self.JUMP_THRESH:
                self.val = new_val           # reset on large jump
            else:
                self.val = self.ALPHA * new_val + (1 - self.ALPHA) * self.val
            self.miss = 0
            return round(self.val, 1), True
        else:
            self.miss += 1
            if self.miss <= self.HOLDOUT and self.val is not None:
                return round(self.val, 1), False
            return None, False


# ────────────────────────────────────────────────────────────────
# 10.  Visualization
# ────────────────────────────────────────────────────────────────

def overlay_drive(vis, drive_clean, alpha=0.20):
    """Semi-transparent green fill over drivable area."""
    green = np.zeros_like(vis)
    green[drive_clean > 0] = (0, 170, 0)
    cv2.addWeighted(green, alpha, vis, 1.0 - 0, 0, vis)


def draw_poly_curve(vis, c, h, roi_y, color, thick=5):
    w   = vis.shape[1]
    pts = []
    for y in range(roi_y, h, 2):
        x = int(round(xv(c, y)))
        if 0 <= x < w:
            pts.append([x, y])
    if len(pts) > 1:
        cv2.polylines(vis, [np.asarray(pts, np.int32)], False, color, thick,
                      cv2.LINE_AA)


def draw_measurement_arrows(vis, lx, rx, my, color):
    for x in (lx, rx):
        cv2.line(vis, (x, my - 35), (x, my + 35), color, 6)
    cv2.arrowedLine(vis, (lx, my), (rx, my), color, 5, tipLength=0.02)
    cv2.arrowedLine(vis, (rx, my), (lx, my), color, 5, tipLength=0.02)


def draw_hud(vis, w_px, w_sm_px, is_live, has_H, w_m, conf, strategy, status,
             lx, rx, main_y, h, w):
    meas_y = main_y
    col_map = {
        "drive" : (0, 220, 255),    # yellow — drivable area edges
        "hybrid": (0, 230, 150),    # green-cyan — one lane + one edge
        "lanes" : (0, 255,  50),    # bright green — both lane lines
        "fail"  : (0,   0, 255),    # red
    }
    col = col_map.get(strategy, (0, 200, 200))

    # Measurement line
    cv2.line(vis, (0, meas_y), (w, meas_y), (100, 100, 100), 1)

    if lx is not None and rx is not None and status != "fail":
        draw_measurement_arrows(vis, lx, rx, meas_y, col)

    held_tag = "  [held]" if not is_live else ""

    if status == "fail":
        lines = ["ROAD NOT DETECTED", ""]
        col   = col_map["fail"]
    elif has_H and w_m is not None:
        lines = [
            f"Road width: {w_m:.2f} m{held_tag}",
            f"{w_px:.0f} px  |  {strategy}  |  conf {conf:.2f}",
        ]
    else:
        if w_sm_px is not None:
            lines = [
                f"Road width: {w_sm_px:.0f} px{held_tag}  (raw {w_px:.0f} px)",
                f"Strategy: {strategy}  |  Confidence: {conf:.2f}",
            ]
        else:
            lines = ["ROAD NOT DETECTED", ""]
            col   = col_map["fail"]

    for i, line in enumerate(lines):
        if not line:
            continue
        sz, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        ty    = 36 + i * 48
        cv2.rectangle(vis, (14, ty - sz[1] - 9), (14 + sz[0] + 16, ty + 10),
                      (0, 0, 0), -1)
        cv2.putText(vis, line, (22, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)


def draw_timestamp(vis, fn, fps, h):
    t   = fn / fps
    txt = f"  {int(t//60):02d}:{int(t%60):02d}  frame {fn}"
    cv2.putText(vis, txt, (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (190, 190, 190), 2, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────
# 11.  Per-frame processing
# ────────────────────────────────────────────────────────────────

def process_frame(lane_mask, drive_mask, H, h, w, dist_m, show_drive_overlay, frame):
    """
    Full processing for one sampled frame.
    Returns a result dict with all computed values.
    """
    roi_top = 0.40
    roi_y   = int(h * roi_top)

    # ── Clean drive mask ──────────────────────────────────────────
    drive_clean = clean_drive_mask(drive_mask, h, w, roi_top)

    if np.sum(drive_clean > 0) < h * w * 0.02:
        # Drive mask basically empty — can't do anything
        return dict(strategy="fail", lc=None, rc=None,
                    lx=None, rx=None, main_y=int(h*0.70),
                    w_px=None, w_m=None, conf=0.0, reason="empty_drive_mask")

    # ── Extract road boundaries from drive mask ───────────────────
    (lbx, lby), (rbx, rby) = extract_road_boundaries(drive_clean, h, w, roi_top)

    drive_lc = ransac_polyfit(lbx, lby)
    drive_rc = ransac_polyfit(rbx, rby)

    valid_drive, d_reason = check_boundaries(drive_lc, drive_rc, h, w, roi_top)

    if not valid_drive:
        return dict(strategy="fail", lc=None, rc=None,
                    lx=None, rx=None, main_y=int(h*0.70),
                    w_px=None, w_m=None, conf=0.0, reason=d_reason)

    # ── Try lane-line refinement ──────────────────────────────────
    lc, rc, used_lanes = refine_with_lane_lines(
        lane_mask, drive_clean, drive_lc, drive_rc, h, w
    )
    strategy = "lanes" if used_lanes else "drive"

    # ── Multi-row pixel width ─────────────────────────────────────
    w_px, lx, rx, main_y = measure_pixel_widths(lc, rc, h, w)

    if w_px is None:
        return dict(strategy="fail", lc=lc, rc=rc,
                    lx=None, rx=None, main_y=int(h*0.70),
                    w_px=None, w_m=None, conf=0.0, reason="meas_failed")

    # ── Confidence ────────────────────────────────────────────────
    drive_pix   = float(np.sum(drive_clean > 0))
    total_pix   = float(h * w)
    drive_ratio = drive_pix / total_pix           # how much of frame is road

    # Left/right drive-mask balance
    mid   = w // 2
    bal_l = float(np.sum(drive_clean[:, :mid] > 0))
    bal_r = float(np.sum(drive_clean[:, mid:] > 0))
    bal   = min(bal_l, bal_r) / max(max(bal_l, bal_r), 1)

    conf_base  = min(1.0, drive_ratio * 6)       # 0 → 1 as road fills ~17% frame
    conf_lanes = 0.15 if used_lanes else 0.0
    conf       = round(min(1.0, conf_base * 0.7 + bal * 0.15 + conf_lanes), 3)

    # ── Homography metric width ───────────────────────────────────
    w_m = None
    if H is not None:
        w_m = metric_width_m(lc, rc, H, h, dist_m)

    return dict(strategy=strategy, lc=lc, rc=rc,
                drive_lc=drive_lc, drive_rc=drive_rc,
                lx=lx, rx=rx, main_y=main_y,
                w_px=w_px, w_m=w_m, conf=conf, reason="ok",
                drive_clean=drive_clean, roi_y=roi_y)


# ────────────────────────────────────────────────────────────────
# 12.  Main video loop
# ────────────────────────────────────────────────────────────────

def create_clips(a):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOP on {device} …")
    model = load_yolop(device)
    print("YOLOP ready.\n")

    H = load_homography(a.homography)
    if H is not None:
        print(f"✓ Homography loaded  →  metric output at {a.distance:.0f} m")
    else:
        print(f"⚠  No homography at:  {a.homography}")
        print("   Pixel-width output only.  Run 03_calibrate_homography.py for metres.\n")

    os.makedirs(a.output, exist_ok=True)
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {a.input}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride  = max(1, round(src_fps / a.fps))
    out_fps = src_fps / stride

    print("═" * 70)
    print("  ROAD WIDTH  v4  —  drive-area boundary + RANSAC polynomial")
    print("═" * 70)
    print(f"  Input      : {a.input}")
    print(f"  Resolution : {vid_w} × {vid_h}  |  {src_fps:.1f} fps source")
    print(f"  Sampling   : {out_fps:.2f} fps  (stride {stride})")
    print(f"  Duration   : {total/src_fps/60:.1f} min  ({total} frames)")
    print("═" * 70)

    # CSV
    csv_path = os.path.join(a.output, "road_width_measurements.csv")
    csvf = open(csv_path, "w", newline="", encoding="utf-8")
    cw   = csv.DictWriter(csvf, fieldnames=[
        "sample","source_frame","time_s",
        "width_px","width_px_smooth","width_m",
        "confidence","strategy","status",
    ])
    cw.writeheader()

    # Clip writer factory
    clip_stride = int(src_fps * a.clip_duration)
    clip_no, writer = 1, None

    def new_clip(n):
        p  = os.path.join(a.output, f"clip_{n:02d}_road_width.mp4")
        wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (vid_w, vid_h))
        if not wr.isOpened():
            raise RuntimeError(f"Cannot create: {p}")
        print(f"\n  → {p}")
        return wr

    smoother_px = Smoother()
    smoother_m  = Smoother()
    smoother_m.JUMP_THRESH = 2.0    # metres

    fn   = 0; samp = 0; t0 = time.time()
    counts = {"drive": 0, "lanes": 0, "hybrid": 0, "fail": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fn % clip_stride == 0:
            if writer:
                writer.release()
            writer = new_clip(clip_no)
            clip_no += 1

        if fn % stride != 0:
            fn += 1
            continue

        vis = frame.copy()

        # ── YOLOP inference ──────────────────────────────────────
        try:
            lane_mask, drive_mask = run_yolop(model, frame, device)
        except Exception as e:
            print(f"\n[WARN] YOLOP frame {fn}: {e}")
            draw_timestamp(vis, fn, src_fps, vid_h)
            writer.write(vis); fn += 1; samp += 1
            continue

        # ── Process ──────────────────────────────────────────────
        res = process_frame(lane_mask, drive_mask, H,
                            vid_h, vid_w, a.distance,
                            not a.no_drive_overlay, frame)

        strategy = res["strategy"]
        counts[strategy] = counts.get(strategy, 0) + 1

        # ── Temporal smoothing ────────────────────────────────────
        w_sm_px, live_px = smoother_px.update(res["w_px"])
        w_sm_m,  live_m  = smoother_m.update(res["w_m"])
        is_live = live_px

        # ── Draw ─────────────────────────────────────────────────
        roi_y = res.get("roi_y", int(vid_h * 0.40))

        # Drive-area overlay
        if not a.no_drive_overlay and res.get("drive_clean") is not None:
            overlay_drive(vis, res["drive_clean"], alpha=0.18)

        # ROI line
        cv2.line(vis, (0, roi_y), (vid_w, roi_y), (50, 50, 0), 1)

        # Boundary curves
        if strategy != "fail":
            lc, rc = res["lc"], res["rc"]
            col = (0, 220, 255) if strategy == "drive" else (0, 255, 50)
            draw_poly_curve(vis, lc, vid_h, roi_y, col, thick=5)
            draw_poly_curve(vis, rc, vid_h, roi_y, col, thick=5)

        # HUD
        draw_hud(vis,
                 res["w_px"],
                 w_sm_px,
                 is_live,
                 H is not None,
                 w_sm_m,
                 res["conf"],
                 strategy,
                 strategy,
                 res["lx"], res["rx"], res.get("main_y", int(vid_h*0.70)),
                 vid_h, vid_w)
        draw_timestamp(vis, fn, src_fps, vid_h)
        writer.write(vis)

        # ── CSV ──────────────────────────────────────────────────
        status = "ok" if strategy != "fail" else ("held" if not is_live else "fail")
        if strategy == "fail" and not is_live:
            status = "no_detection:" + res.get("reason", "")
        cw.writerow({
            "sample"         : samp,
            "source_frame"   : fn,
            "time_s"         : round(fn / src_fps, 3),
            "width_px"       : res["w_px"]   if res["w_px"] is not None else "",
            "width_px_smooth": w_sm_px        if w_sm_px    is not None else "",
            "width_m"        : w_sm_m         if w_sm_m     is not None else "",
            "confidence"     : res["conf"],
            "strategy"       : strategy,
            "status"         : status,
        })

        fn += 1; samp += 1

        if samp % 20 == 0:
            pct = fn / total * 100
            el  = time.time() - t0
            sps = samp / max(el, 1)
            eta = (total / stride - samp) / max(sps, 0.01)
            d, l, f = counts["drive"], counts["lanes"], counts["fail"]
            print(f"\r  {pct:5.1f}%  drive={d} lanes={l} fail={f}"
                  f"  {sps:.1f} samp/s  ETA {eta/60:.1f} min", end="")

    if writer:
        writer.release()
    cap.release()
    csvf.close()

    tot = sum(counts.values())
    print(f"\n\n{'═'*70}")
    print("  DONE")
    print(f"  Output : {a.output}")
    print(f"  CSV    : {csv_path}")
    print()
    print("  Detection breakdown:")
    for s, n in counts.items():
        bar = "█" * int(n / max(tot, 1) * 30)
        print(f"    {s:<8}  {n:5d}  {n/max(tot,1)*100:5.1f}%  {bar}")

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        px = df["width_px_smooth"].dropna()
        if len(px):
            print(f"\n  Pixel width:  median={px.median():.0f} px  "
                  f"P10={px.quantile(0.10):.0f}  P90={px.quantile(0.90):.0f}")
        wm = df["width_m"].dropna()
        if len(wm):
            print(f"  Metric width: median={wm.median():.3f} m  "
                  f"P10={wm.quantile(0.10):.3f}  P90={wm.quantile(0.90):.3f}")
        elif H is None:
            print("\n  ⚠  No metric output — run 03_calibrate_homography.py")
    except ImportError:
        pass
    print("═" * 70)


# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Road width from dashcam — drive-area boundary + RANSAC polynomial."
    )
    p.add_argument("--input",
        default=r"E:\lane_width\testing\lane_width_estimation_v2\data\videos\dashcam.mp4")
    p.add_argument("--output",
        default=r"E:\lane_width\testing\lane_width_estimation_v2\data\outputs\road_width_clips")
    p.add_argument("--homography",
        default=r"E:\lane_width\testing\lane_width_estimation_v2\calibration\homography.json",
        help="calibration/homography.json (optional; pixel-only without it)")
    p.add_argument("--fps",           type=float, default=5.0,
        help="YOLOP sampling FPS (default 5)")
    p.add_argument("--distance",      type=float, default=15.0,
        help="Forward Z distance in metres for metric measurement (default 15)")
    p.add_argument("--clip-duration", type=int,   default=300,
        help="Seconds per output clip (default 300 = 5 min)")
    p.add_argument("--no-drive-overlay", action="store_true",
        help="Suppress semi-transparent drivable-area overlay")
    return p.parse_args()


if __name__ == "__main__":
    create_clips(parse_args())