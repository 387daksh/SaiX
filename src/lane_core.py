"""
Shared engine for metric EGO-LANE width and LEAD-VEHICLE width estimation.

Measurement model
-----------------
Everything is measured on the road plane, not in the image.

  image pixel --(pinhole ray)--> rotate by camera pitch/yaw --> intersect road plane
  --> (X lateral metres, Z forward metres)

The road-plane transform needs three things:
  * focal length f        -- recovered from the footage by `calibrate_focal`
  * vanishing point       -- measured per frame from the lane markings themselves
  * camera height h       -- the single physical number that sets absolute scale

Pitch and yaw are derived from the vanishing point each frame, so the mapping
tracks road pitch changes instead of assuming a fixed homography.
"""

import json
import math
import os
import queue
import subprocess
import threading

import cv2
import numpy as np

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ---------------------------------------------------------------------------
# Detected lane marking
# ---------------------------------------------------------------------------

class Marking:
    """One lane-marking instance in image space."""

    __slots__ = ("px", "py", "kind", "score")

    def __init__(self, px, py, kind="unknown", score=1.0):
        self.px = np.asarray(px, np.float32)
        self.py = np.asarray(py, np.float32)
        self.kind = kind
        self.score = float(score)

    def __len__(self):
        return len(self.px)


# ---------------------------------------------------------------------------
# Ground-plane projection
# ---------------------------------------------------------------------------

class GroundPlane:
    """
    Maps image pixels to road-plane (X right, Z forward) metres.

    Camera frame is standard CV: x right, y down, z forward. The camera sits
    `height` metres above the road and is pitched down by `pitch`.
    """

    def __init__(self, f, cx, cy, vp_u, vp_v, height, yaw_correct=True):
        self.f = float(f)
        self.cx = float(cx)
        self.cy = float(cy)
        self.height = float(height)
        # Pitching the camera down lifts the horizon, so the vanishing point
        # sits above the principal point: vp_v = cy - f*tan(pitch). Taking the
        # difference the other way round puts the computed horizon on the wrong
        # side of centre and the near road reads as sky.
        self.pitch = math.atan2(self.cy - float(vp_v), self.f)
        if yaw_correct:
            self.yaw = math.atan2(
                (float(vp_u) - self.cx) * math.cos(self.pitch), self.f
            )
        else:
            self.yaw = 0.0

    def to_ground(self, u, v):
        """
        Project image points to the road plane.

        Returns (X, Z) arrays in metres. Points on or above the horizon come
        back as NaN.
        """
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)

        dx = (u - self.cx) / self.f
        dy = (v - self.cy) / self.f
        dz = np.ones_like(dx)

        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        # Rotate the ray out of the pitched camera frame into a road-aligned
        # frame (Y stays "down", Z becomes horizontal-forward).
        y_w = cp * dy + sp * dz
        z_w = -sp * dy + cp * dz

        # Only downward rays ever meet the road.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y_w > 1e-6, self.height / y_w, np.nan)

        X = t * dx
        Z = t * z_w

        if self.yaw != 0.0:
            cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
            X, Z = cy_ * X - sy_ * Z, sy_ * X + cy_ * Z

        bad = ~np.isfinite(X) | ~np.isfinite(Z) | (Z <= 0)
        X = np.where(bad, np.nan, X)
        Z = np.where(bad, np.nan, Z)
        return X, Z

    def to_image(self, X, Z):
        """Inverse map: road-plane metres back to image pixels."""
        X = np.asarray(X, np.float64)
        Z = np.asarray(Z, np.float64)
        if self.yaw != 0.0:
            cy_, sy_ = math.cos(-self.yaw), math.sin(-self.yaw)
            X, Z = cy_ * X - sy_ * Z, sy_ * X + cy_ * Z

        Y = np.full_like(X, self.height)
        cp, sp = math.cos(-self.pitch), math.sin(-self.pitch)
        y_c = cp * Y + sp * Z
        z_c = -sp * Y + cp * Z

        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(z_c > 1e-6, self.cx + self.f * X / z_c, np.nan)
            v = np.where(z_c > 1e-6, self.cy + self.f * y_c / z_c, np.nan)
        return u, v


# ---------------------------------------------------------------------------
# Vanishing point
# ---------------------------------------------------------------------------

def _fit_image_line(mk, min_span=12.0):
    """
    Fit u = m*v + c to one marking. Returns a normalised line (a, b, c) with
    a*u + b*v + c = 0, or None if the marking is too short or too horizontal.
    """
    if len(mk) < 12:
        return None
    v, u = mk.py.astype(np.float64), mk.px.astype(np.float64)
    if v.max() - v.min() < min_span:
        return None
    # Reject near-horizontal paint (stop lines, arrows, crosswalks).
    if (u.max() - u.min()) > 6.0 * (v.max() - v.min()):
        return None
    m, c = np.polyfit(v, u, 1)
    a, b, cc = 1.0, -m, -c
    n = math.hypot(a, b)
    return a / n, b / n, cc / n


def vanishing_point(markings, w, h):
    """
    Least-squares intersection of all marking lines. Lane markings are parallel
    on the road, so in the image they share one vanishing point.
    """
    lines = [L for L in (_fit_image_line(m) for m in markings) if L is not None]
    if len(lines) < 2:
        return None

    A = np.array([[l[0], l[1]] for l in lines], np.float64)
    b = -np.array([l[2] for l in lines], np.float64)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    u, v = float(sol[0]), float(sol[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    # A vanishing point far outside this band means the fit collapsed.
    if not (0.15 * w < u < 0.85 * w and 0.20 * h < v < 0.70 * h):
        return None
    return u, v


class VPFilter:
    """Temporal smoother for the vanishing point."""

    def __init__(self, alpha=0.1, max_jump=140.0):
        self.alpha = alpha
        self.max_jump = max_jump
        self.vp = None

    def update(self, vp):
        if vp is None:
            return self.vp
        if self.vp is None:
            self.vp = vp
        else:
            du = vp[0] - self.vp[0]
            dv = vp[1] - self.vp[1]
            a = 0.5 if math.hypot(du, dv) > self.max_jump else self.alpha
            self.vp = (self.vp[0] + a * du, self.vp[1] + a * dv)
        return self.vp


# ---------------------------------------------------------------------------
# Lane boundaries on the road plane
# ---------------------------------------------------------------------------

class Boundary:
    """A lane boundary on the road plane, modelled as X = f(Z)."""

    __slots__ = ("coef", "zmin", "zmax", "n", "kind", "score", "X", "Z")

    def __init__(self, coef, zmin, zmax, n, kind, score, X=None, Z=None):
        self.coef = coef
        self.zmin = zmin
        self.zmax = zmax
        self.n = n
        self.kind = kind
        self.score = score
        # Kept so fragments of one dashed line can be regrouped and refitted
        # from their original points rather than from extrapolated fits.
        self.X = X
        self.Z = Z

    def x_at(self, z):
        return float(np.polyval(self.coef, z))

    def covers(self, z, slack=2.0):
        return (self.zmin - slack) <= z <= (self.zmax + slack)


def project_markings(markings, gp, z_near=3.0, z_far=55.0, min_pts=10):
    """
    Project each marking onto the road plane and fit X = f(Z).

    Markings that land as a short, wide blob (stop lines, arrows, glare) are
    dropped: on the road plane a real lane marking is long in Z and thin in X.
    """
    out = []
    for mk in markings:
        if len(mk) < min_pts:
            continue
        X, Z = gp.to_ground(mk.px, mk.py)
        ok = np.isfinite(X) & np.isfinite(Z) & (Z > z_near) & (Z < z_far)
        if ok.sum() < min_pts:
            continue
        X, Z = X[ok], Z[ok]

        z_span = float(Z.max() - Z.min())
        x_span = float(X.max() - X.min())
        if z_span < 1.2:
            continue
        if x_span > max(1.5, 0.9 * z_span):
            continue

        deg = 2 if (len(Z) >= 40 and z_span > 10.0) else 1
        try:
            coef = np.polyfit(Z, X, deg)
        except (np.linalg.LinAlgError, ValueError):
            continue
        resid = np.abs(X - np.polyval(coef, Z))
        keep = resid < max(0.25, 3.0 * float(np.median(resid)) + 1e-6)
        if keep.sum() >= min_pts:
            try:
                coef = np.polyfit(Z[keep], X[keep], deg)
            except (np.linalg.LinAlgError, ValueError):
                pass

        out.append(
            Boundary(coef, float(Z.min()), float(Z.max()), int(len(Z)),
                     mk.kind, mk.score, X=X, Z=Z)
        )
    return out


def merge_boundaries(bounds, tol=0.45):
    """
    Group fragments that describe the same physical line, then refit each group.

    A dashed lane line arrives as one detection per dash. Comparing the dashes
    by their own fitted curves does not work: each dash spans a metre or two of
    depth, so its slope is noise and joining two of them means extrapolating
    both far outside their support. When that comparison fails the dashes stay
    separate, get dropped for being too short, and the nearest surviving
    boundary on that side becomes the next line out -- reading one whole lane
    too wide.

    Lane lines on a road are parallel, so instead every fragment is measured as
    a lateral offset from one reference curve (the longest boundary, which is
    the best-conditioned). That offset is well determined even for a single
    short dash, and grouping happens in that 1-D space.
    """
    if not bounds:
        return []

    ref = max(bounds, key=lambda b: b.zmax - b.zmin)
    offsets = []
    for b in bounds:
        if b.Z is not None and len(b.Z):
            d = b.X - np.polyval(ref.coef, b.Z)
        else:
            zz = np.linspace(b.zmin, b.zmax, 8)
            d = np.polyval(b.coef, zz) - np.polyval(ref.coef, zz)
        offsets.append(float(np.median(d)))

    groups = []
    for i in np.argsort(offsets):
        off = offsets[i]
        if groups and abs(off - groups[-1][1]) < tol:
            groups[-1][0].append(bounds[i])
            groups[-1][2].append(off)
            groups[-1][1] = float(np.median(groups[-1][2]))
        else:
            groups.append([[bounds[i]], off, [off]])

    merged = []
    for members, _, _ in groups:
        if len(members) == 1:
            merged.append(members[0])
            continue

        Zs = [m.Z for m in members if m.Z is not None and len(m.Z)]
        Xs = [m.X for m in members if m.X is not None and len(m.X)]
        if not Zs:
            merged.append(max(members, key=lambda m: m.zmax - m.zmin))
            continue
        Z = np.concatenate(Zs)
        X = np.concatenate(Xs)
        zmin, zmax = float(Z.min()), float(Z.max())
        deg = 2 if (zmax - zmin) > 12.0 and len(Z) >= 30 else 1
        try:
            coef = np.polyfit(Z, X, deg)
        except (np.linalg.LinAlgError, ValueError):
            coef = max(members, key=lambda m: m.zmax - m.zmin).coef

        kinds = [m.kind for m in members]
        # Confidence of the regrouped line is its members' average weighted by
        # how much each contributes, not the best member's -- one strong dash
        # should not speak for a line made mostly of weak ones.
        wsum = float(sum(m.n for m in members)) or 1.0
        score = float(sum(m.score * m.n for m in members) / wsum)
        merged.append(Boundary(
            coef, zmin, zmax, int(wsum),
            max(set(kinds), key=kinds.count), score, X=X, Z=Z,
        ))
    return merged


def offset_boundary(b, dx):
    """Copy of a boundary shifted sideways, used to draw where a missing
    marking is expected to lie."""
    coef = np.array(b.coef, dtype=float).copy()
    coef[-1] += dx
    return Boundary(coef, b.zmin, b.zmax, b.n, b.kind, b.score)


def auto_z_ref(bounds, lo=4.0, hi=16.0, default=10.0):
    """
    Pick the depth to measure at from where the detector actually has support.

    Detectors differ a lot here: the segmentation model returns markings out to
    ~50 m, while CLRNet returns short near-field lanes covering ~3-5 m. A fixed
    reference depth silently discards everything one of them produces.
    """
    if not bounds:
        return default
    mids = [0.5 * (b.zmin + b.zmax) for b in bounds]
    return float(np.clip(np.median(mids), lo, hi))


def select_ego_lane(bounds, z_ref, max_half=6.0, min_span=3.5):
    """
    Pick the two boundaries straddling the camera (X = 0).

    The nearest boundary on each side is the ego lane, which also handles a
    double centre line correctly: the inner stripe is the nearer one.

    `min_span` drops boundaries built from one short far-field fragment. Those
    extrapolate badly and were the main source of two-lane-wide readings.
    """
    left = right = None
    lx = -1e9
    rx = 1e9
    for b in bounds:
        if (b.zmax - b.zmin) < min_span:
            continue
        # Evaluate inside the boundary's own depth range instead of demanding
        # it reach z_ref. Lane lines are parallel, so a near dash and a long
        # edge line are still comparable; requiring coverage at a single depth
        # silently drops the near dashes and promotes the next line out.
        z = float(np.clip(z_ref, b.zmin, b.zmax))
        if z_ref > b.zmax * 3.0 or z_ref < b.zmin / 3.0:
            continue
        x = b.x_at(z)
        if not np.isfinite(x) or abs(x) > max_half:
            continue
        if x < 0 and x > lx:
            lx, left = x, b
        elif x >= 0 and x < rx:
            rx, right = x, b
    return left, right


def boundary_offsets(bounds, z_ref, max_half, min_span):
    """Lateral offset of each usable boundary, evaluated inside its own support."""
    out = []
    for b in bounds:
        if (b.zmax - b.zmin) < min_span:
            continue
        if z_ref > b.zmax * 3.0 or z_ref < b.zmin / 3.0:
            continue
        x = b.x_at(float(np.clip(z_ref, b.zmin, b.zmax)))
        if np.isfinite(x) and abs(x) <= max_half:
            out.append((float(x), b))
    return sorted(out, key=lambda t: t[0])


def grid_phase(offsets, pitch, weights=None):
    """
    Phase of the lane grid, i.e. where the repeating boundary positions sit.

    Markings on a road repeat every lane width, so their offsets modulo the
    pitch cluster around one value. Averaging them on the unit circle handles
    the wrap-around and tolerates a stray line that is off-grid.
    """
    if not len(offsets) or pitch <= 1e-6:
        return None
    a = 2.0 * math.pi * np.asarray(offsets, float) / pitch
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    v = np.sum(w * np.exp(1j * a))
    if abs(v) < 1e-9:
        return None
    return float(pitch * np.angle(v) / (2.0 * math.pi))


LANE_NAMES = ("left", "middle", "right")


def boundary_confidence(hit, tol, height):
    """
    How much to trust one matched boundary, in 0..1.

    Combines the detector's own score with two geometric terms: how close the
    detection sits to its expected grid position, and how much depth it spans.
    A line seen over a long run of road is worth more than a single short dash
    that happens to land in the right place.
    """
    if hit is None:
        return 0.0
    _, b, d = hit
    # Mild modifiers: a well-placed, long line should report essentially the
    # detector's own confidence, not a heavily discounted version of it.
    pos = 1.0 - 0.35 * min(1.0, d / max(tol, 1e-6))
    span = float(np.clip((b.zmax - b.zmin) / max(1.5 * height, 1e-6),
                         0.55, 1.0))
    return float(np.clip(b.score * pos * span, 0.0, 1.0))


class Lane:
    """One lane of the road, named by its position relative to the camera."""

    __slots__ = ("name", "left_x", "right_x", "left", "right", "width", "conf")

    def __init__(self, name, left_x, right_x, left, right, width, conf):
        self.name = name
        self.left_x = left_x
        self.right_x = right_x
        self.left = left
        self.right = right
        self.width = width
        self.conf = conf


class EgoLane:
    """Which lane the camera is in, and which expected markings are absent."""

    __slots__ = ("left", "right", "left_x", "right_x", "width",
                 "missing_left", "missing_right", "slots", "pitch", "lanes")

    def __init__(self, left, right, left_x, right_x, width,
                 missing_left, missing_right, slots, pitch):
        self.left = left
        self.right = right
        self.left_x = left_x
        self.right_x = right_x
        self.width = width
        self.missing_left = missing_left
        self.missing_right = missing_right
        self.slots = slots
        self.pitch = pitch

    @property
    def missing(self):
        if self.missing_left and self.missing_right:
            return "both"
        if self.missing_left:
            return "left"
        if self.missing_right:
            return "right"
        return ""


def resolve_ego_lane(bounds, z_ref, pitch, phase, max_half, min_span,
                     height, tol_frac=0.28, n_side=2):
    """
    Place the expected lane grid, then report the ego lane against it.

    Taking simply the nearest detected boundary either side of the camera fails
    exactly when a marking is absent: the next line out gets promoted and the
    lane reads one whole lane too wide. Anchoring to a grid of expected
    positions instead means an absent marking is reported as absent, which is
    both the honest answer and the thing a reviewer wants flagged.
    """
    if pitch is None or pitch <= 1e-6 or phase is None:
        return None

    offs = boundary_offsets(bounds, z_ref, max_half, min_span)
    tol = tol_frac * pitch

    # Grid line immediately left of the camera, and the one above it.
    k_left = math.floor((0.0 - phase) / pitch)
    left_x = phase + k_left * pitch
    right_x = left_x + pitch

    def nearest(target):
        best = None
        bd = tol
        for x, b in offs:
            d = abs(x - target)
            if d < bd:
                bd, best = d, (x, b, d)
        return best

    hit_l = nearest(left_x)
    hit_r = nearest(right_x)

    slots = []
    for k in range(k_left - n_side, k_left + 2 + n_side):
        gx = phase + k * pitch
        if abs(gx) > max_half:
            continue
        slots.append((round(gx, 3), nearest(gx) is not None))

    left = hit_l[1] if hit_l else None
    right = hit_r[1] if hit_r else None
    width = None
    if left is not None and right is not None:
        width = hit_r[0] - hit_l[0]
        if width <= 0:
            width = None

    ego = EgoLane(left, right, left_x, right_x, width,
                  hit_l is None, hit_r is None, slots, pitch)

    # The ego lane plus its neighbours, so the readout covers the road rather
    # than only the lane being driven.
    ego.lanes = []
    for name, k0 in zip(LANE_NAMES, (k_left - 1, k_left, k_left + 1)):
        ax = phase + k0 * pitch
        bx = ax + pitch
        ha, hb = nearest(ax), nearest(bx)
        ca = boundary_confidence(ha, tol, height)
        cb = boundary_confidence(hb, tol, height)
        lw = (hb[0] - ha[0]) if (ha and hb) else None
        if lw is not None and lw <= 0:
            lw = None
        ego.lanes.append(Lane(name, ax, bx,
                              ha[1] if ha else None, hb[1] if hb else None,
                              lw, math.sqrt(ca * cb)))
    return ego


def lane_width(left, right, z_ref, z_lo=6.0, z_hi=22.0):
    """
    Median separation over a depth window.

    Sampling a span of Z instead of a single row keeps one bad fit at one
    distance from swinging the result.
    """
    if left is None or right is None:
        return None
    lo = max(z_lo, left.zmin, right.zmin)
    hi = min(z_hi, left.zmax, right.zmax)

    if hi > lo:
        zs = np.linspace(lo, hi, 12)
        ws = np.array([right.x_at(z) - left.x_at(z) for z in zs])
        ws = ws[np.isfinite(ws) & (ws > 0)]
        if len(ws) == 0:
            return None
        return float(np.median(ws))

    # No shared depth range -- a near dash paired with a long edge line. The
    # two are still parallel, so compare each inside its own support rather
    # than extrapolating one of them across the gap.
    xl = left.x_at(float(np.clip(z_ref, left.zmin, left.zmax)))
    xr = right.x_at(float(np.clip(z_ref, right.zmin, right.zmax)))
    w = xr - xl
    return float(w) if np.isfinite(w) and w > 0 else None


# ---------------------------------------------------------------------------
# Metric scale
# ---------------------------------------------------------------------------
#
# What is actually identifiable from this footage
# -----------------------------------------------
# Parallel lane lines plus their vanishing point do NOT determine focal length:
# any f maps lines through the vanishing point onto parallel road-plane lines,
# so there is nothing for a focal solver to minimise. Measured directly, the
# lane-width-vs-depth spread is bit-identical across f = 600 .. 3000.
#
# Working the algebra through, with pitch p = atan((vp_v - cy)/f):
#
#     ray_y  = ((v - cy) + (vp_v - cy)) / f        (small-angle)
#     t      = height / ray_y          ~  height * f / (...)
#     X      = t * (u - cx) / f        ~  height * (u - cx) / (...)
#
# f cancels. Lateral metric scale is set by camera height alone, and every
# width this module reports is exactly proportional to it. So there is one
# calibration constant, not two, and `focal` only has to be roughly right for
# the depth (Z) axis.
#
# Supply the constant either way:
#   * measure the camera height once with a tape, or
#   * pass a known lane width for this road and let `scale_from_lane_width`
#     back out the effective height.


def scale_from_lane_width(raw_widths, target_m, base_height=1.0):
    """
    Back out the effective camera height from a road whose lane width is known.

    `raw_widths` are lane widths measured with `base_height`. Width scales
    linearly with height, so one division fixes the whole clip.
    """
    vals = np.asarray([w for w in raw_widths if w and np.isfinite(w)], float)
    if len(vals) < 10:
        return None, None
    med = float(np.median(vals))
    if med <= 1e-6:
        return None, None
    return base_height * target_m / med, med


# ---------------------------------------------------------------------------
# Lead vehicle
# ---------------------------------------------------------------------------

class Vehicle:
    __slots__ = ("box", "cls", "score", "X", "Z", "width", "track_id",
                 "plate", "plate_conf")

    def __init__(self, box, cls, score, X, Z, width, track_id=None):
        self.box = box
        self.cls = cls
        self.score = score
        self.X = X
        self.Z = Z
        self.width = width
        self.track_id = track_id
        # Filled in by the plate reader once it knows which box a plate sits in.
        self.plate = None
        self.plate_conf = 0.0


# Where a box-bottom width measurement can be trusted, in metres. Closer than
# Z_LO the box spans the side of an obliquely seen vehicle rather than its
# front; past Z_HI the box is so few pixels tall that a pixel of slop at the
# tyre line swings Z by several metres. Beyond X_MAX the vehicle is oncoming
# or in a far lane and is again seen side-on.
#
# Tuned on this footage by trading coverage against the share of per-track
# widths that come out physically impossible: this band keeps 246 of 655
# tracks at 15% impossible, against 504 at 22% unfiltered. Tightening to
# 10-30 m and 3 m cuts that to 13% but keeps only 76 tracks, which is not
# worth losing two thirds of the vehicles for. Accuracy is flat across every
# band tried -- a car reads about 1.6 m against a true 1.75 -- so the band
# removes the catastrophic tail, not the residual bias.
WIDTH_OK_Z = (8.0, 40.0)
WIDTH_OK_X = 6.0


def width_is_reliable(Z, X):
    """Whether a box-bottom width at this position is worth reporting."""
    if Z is None or X is None:
        return False
    return WIDTH_OK_Z[0] <= Z <= WIDTH_OK_Z[1] and abs(X) <= WIDTH_OK_X


def measure_vehicles(boxes, classes, scores, gp, z_max=90.0, track_ids=None):
    """
    Width of each vehicle from its box's bottom edge.

    The bottom edge sits where the tyres meet the road, so both of its corners
    lie on the road plane and project to true metric positions.

    With gp None there is no road plane and nothing can be placed in metres.
    The boxes still come back, carrying no metric fields, because plate reading
    and track identity need no geometry -- and long stretches of dashcam
    footage have no lane markings to build a ground plane from.
    """
    out = []
    if track_ids is None:
        track_ids = [None] * len(boxes)
    for (x1, y1, x2, y2), c, s, tid in zip(boxes, classes, scores, track_ids):
        if gp is None:
            out.append(Vehicle((x1, y1, x2, y2), c, s, None, None, None, tid))
            continue
        X, Z = gp.to_ground(np.array([x1, x2]), np.array([y2, y2]))
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Z)):
            continue
        z = float(np.mean(Z))
        if not (2.0 < z < z_max):
            continue
        w = float(abs(X[1] - X[0]))
        out.append(
            Vehicle((x1, y1, x2, y2), c, s, float(np.mean(X)), z, w, tid)
        )
    return out


def pick_lead(vehicles, left, right, z_ref, fallback_half=1.9):
    """
    Nearest vehicle whose footprint centre falls inside the ego lane.

    Uses the measured lane edges where available so the gate follows the lane
    through curves instead of assuming the lane sits straight ahead.
    """
    best = None
    for v in vehicles:
        if left is not None and right is not None \
                and left.covers(v.Z, 12.0) and right.covers(v.Z, 12.0):
            xl, xr = left.x_at(v.Z), right.x_at(v.Z)
            if xr < xl:
                xl, xr = xr, xl
            pad = 0.15 * max(xr - xl, 0.5)
            inside = (xl - pad) <= v.X <= (xr + pad)
        else:
            inside = abs(v.X) <= fallback_half
        if not inside:
            continue
        if best is None or v.Z < best.Z:
            best = v
    return best


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

class Smoother:
    """
    Median-gated exponential smoother.

    Lane width and vehicle width are near-constant physically, so a value far
    from the recent median is treated as a bad frame rather than real change.

    The gate is relative (a fraction of the running median) because absolute
    scale here depends on camera height, which is a free constant -- a fixed
    metre threshold would tighten or loosen itself as that constant changes.
    """

    def __init__(self, alpha=0.18, window=15, reject_rel=0.20, holdout=20):
        self.alpha = alpha
        self.window = window
        self.reject_rel = reject_rel
        self.holdout = holdout
        self.hist = []
        self.val = None
        self.miss = 0

    def reset(self):
        self.hist.clear()
        self.val = None
        self.miss = 0

    def update(self, x):
        if x is not None and np.isfinite(x):
            if len(self.hist) >= 5:
                med = float(np.median(self.hist))
                if abs(x - med) > self.reject_rel * max(med, 1e-6):
                    # Outlier: remember it, but do not let it move the output.
                    self.hist.append(x)
                    del self.hist[:-self.window]
                    self.miss += 1
                    if self.miss <= self.holdout and self.val is not None:
                        return self.val, False
                    self.hist = [x]
                    self.val = x
                    self.miss = 0
                    return self.val, True
            self.hist.append(x)
            del self.hist[:-self.window]
            self.val = x if self.val is None else \
                self.alpha * x + (1 - self.alpha) * self.val
            self.miss = 0
            return self.val, True

        self.miss += 1
        if self.miss <= self.holdout and self.val is not None:
            return self.val, False
        if self.miss > self.holdout:
            self.val = None
            self.hist.clear()
        return None, False


# ---------------------------------------------------------------------------
# Minimal overlay
# ---------------------------------------------------------------------------

LANE_COLOR = (90, 230, 90)
VEH_COLOR = (80, 190, 255)
VEH_OTHER_COLOR = (40, 150, 255)
MISSING_COLOR = (70, 70, 245)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Overlay text and panel geometry were sized for a screen; the numbers are
# unreadable when the 1080p render is watched scaled down, so every font size
# and pixel offset in the readout is multiplied by this.
UI_SCALE = 1.6


def _px(v):
    return int(round(v * UI_SCALE))


def _thick(v):
    return max(1, int(round(v * UI_SCALE)))


def draw_missing(img, b, gp, pitch, color=MISSING_COLOR, z_lo=4.0, z_hi=40.0):
    """
    Dashed line where a marking was expected but none was found.

    Drawn so a reviewer can judge the fourth step of the workflow: whether the
    paint is worn away or the road never had a marking there.
    """
    if b is None:
        return
    lo = max(z_lo, b.zmin)
    hi = min(z_hi, b.zmax)
    if hi <= lo:
        return
    zs = np.linspace(lo, hi, 80)
    u, v = gp.to_image(np.array([b.x_at(z) for z in zs]), zs)
    pts = [
        (int(a), int(bb))
        for a, bb in zip(u, v)
        if np.isfinite(a) and np.isfinite(bb)
        and -5000 < a < 5000 and 0 <= bb < img.shape[0]
    ]
    for i in range(0, len(pts) - 3, 6):
        cv2.line(img, pts[i], pts[i + 3], color, 3, cv2.LINE_AA)


def draw_boundary(img, b, gp, color, z_lo=4.0, z_hi=40.0, thick=3):
    if b is None:
        return
    lo = max(z_lo, b.zmin)
    hi = min(z_hi, b.zmax)
    if hi <= lo:
        return
    zs = np.linspace(lo, hi, 60)
    xs = np.array([b.x_at(z) for z in zs])
    u, v = gp.to_image(xs, zs)
    pts = [
        [int(a), int(bb)]
        for a, bb in zip(u, v)
        if np.isfinite(a) and np.isfinite(bb)
        and -5000 < a < 5000 and 0 <= bb < img.shape[0]
    ]
    if len(pts) > 1:
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, color,
                      thick, cv2.LINE_AA)


def draw_confidence(img, lanes, x0, y0, w=300):
    """
    Per-lane confidence as a labelled meter row.

    Confidence is a magnitude, so it rides on bar length and the printed number
    together -- one hue throughout, no colour-coded good/bad. That keeps it
    readable for colour-vision deficiency and in greyscale, where a red/green
    scheme would collapse. The number is always present, so colour is never
    carrying the value on its own.
    """
    lh = _px(26)
    bx = x0 + _px(74)
    bw = w - _px(108)
    cv2.putText(img, "Lane marking confidence", (x0, y0 - _px(9)), _FONT,
                0.45 * UI_SCALE, (150, 160, 170), _thick(1), cv2.LINE_AA)
    for i, ln in enumerate(lanes):
        y = y0 + i * lh
        cv2.putText(img, ln.name.title(), (x0, y + _px(11)), _FONT,
                    0.46 * UI_SCALE, (170, 180, 190), _thick(1), cv2.LINE_AA)
        cv2.rectangle(img, (bx, y + _px(2)), (bx + bw, y + _px(12)),
                      (58, 58, 58), -1)
        fill = int(bw * float(np.clip(ln.conf, 0.0, 1.0)))
        if fill > 1:
            cv2.rectangle(img, (bx, y + _px(2)), (bx + fill, y + _px(12)),
                          LANE_COLOR, -1)
        cv2.putText(img, f"{ln.conf:.2f}", (bx + bw + _px(10), y + _px(12)),
                    _FONT, 0.5 * UI_SCALE, (235, 238, 242), _thick(1),
                    cv2.LINE_AA)


def draw_readout(img, lane_m, veh_m, veh_label, missing="", lanes=None):
    """Two numbers, plus a missing-marking line only when one is absent."""
    lines = [
        ("Lane width", f"{lane_m:.2f} m" if lane_m else "--"),
        (f"{veh_label} width" if veh_label else "Vehicle width",
         f"{veh_m:.2f} m" if veh_m else "--"),
    ]
    if missing:
        lines.append(("Marking missing", missing.upper()))
    pad, lh = _px(14), _px(42)
    n_conf = len(lanes) if lanes else 0
    bw = _px(366)
    bh = pad * 2 + lh * len(lines) + (_px(26) * n_conf + _px(40)
                                      if n_conf else 0)
    x0, y0 = 26, 26

    panel = img[y0:y0 + bh, x0:x0 + bw]
    if panel.size:
        cv2.addWeighted(np.full_like(panel, (16, 18, 20)), 0.72,
                        panel, 0.28, 0, panel)
        img[y0:y0 + bh, x0:x0 + bw] = panel

    for i, (k, val) in enumerate(lines):
        y = y0 + pad + lh * i + _px(28)
        col = MISSING_COLOR if k == "Marking missing" else (255, 255, 255)
        cv2.putText(img, k, (x0 + _px(16), y), _FONT, 0.6 * UI_SCALE,
                    (165, 175, 185), _thick(1), cv2.LINE_AA)
        cv2.putText(img, val, (x0 + _px(180), y), _FONT, 0.84 * UI_SCALE, col,
                    _thick(2), cv2.LINE_AA)

    if lanes:
        draw_confidence(img, lanes, x0 + _px(16),
                        y0 + pad + lh * len(lines) + _px(30),
                        w=bw - _px(32))


# ---------------------------------------------------------------------------
# H.264 writer
# ---------------------------------------------------------------------------

class VideoWriter:
    """
    Pipes BGR frames to ffmpeg.

    OpenCV's mp4v writer produces ~400 MB per minute at 1080p here; H.264 via
    ffmpeg is roughly two orders of magnitude smaller at better quality.
    """

    def __init__(self, path, w, h, fps, crf=23, preset="veryfast", depth=12):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", f"{fps}",
                "-i", "-",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Feeding the pipe from a thread lets x264 encode while the caller runs
        # the next frame's inference; a blocking write serializes the two. The
        # queue is bounded, so a slow encoder applies backpressure rather than
        # growing without limit.
        self.q = queue.Queue(maxsize=depth)
        self.err = None
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        while True:
            frame = self.q.get()
            if frame is None:
                return
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                err = self.proc.stderr.read().decode(errors="replace")[-800:]
                self.err = RuntimeError(
                    f"ffmpeg failed while writing {self.path}:\n{err}")
                return

    def write(self, frame):
        # The frame is encoded later, on the pump thread, so the caller must
        # not mutate it after handing it over.
        if self.err is not None:
            raise self.err
        if not self.thread.is_alive():
            raise RuntimeError(f"ffmpeg writer for {self.path} stopped early")
        self.q.put(frame)

    def release(self):
        self.q.put(None)
        self.thread.join()
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self.proc.wait()
        if self.err is not None:
            raise self.err


class FrameReader:
    """
    Decodes ahead of the consumer, on its own thread.

    Decode, inference and encode each idle while the others work when they run
    in one loop; reading ahead into a bounded queue overlaps decode with the
    rest. Takes ownership of `cap`: seek before constructing one, and do not
    read from `cap` directly afterwards.
    """

    def __init__(self, cap, depth=12):
        self.cap = cap
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            while not self.stop.is_set():
                try:
                    self.q.put(frame, timeout=0.2)
                    break
                except queue.Full:
                    continue
        self.q.put(None)

    def read(self):
        frame = self.q.get()
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self.stop.set()
        # Drain, so a blocked put sees room and the thread can observe the stop.
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------

def load_calibration(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_calibration(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
