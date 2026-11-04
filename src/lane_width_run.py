"""
Ego-lane width and lead-vehicle width from a dashcam clip.

    python src/lane_width_run.py --detector yoloseg --part 5

Measures the lane the car is actually in -- the two markings straddling the
camera -- not the full carriageway, and the metric width of the vehicle ahead
in that lane. Output is one H.264 clip with a two-line readout plus a CSV.

Absolute scale rests on one constant: camera height. Either pass the measured
mount height, or pass --anchor-lane-width with this road's known lane width and
the effective height is solved from the clip itself.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lane_core as lc
import lane_detectors as ld
from lane_detectors import VehicleDetector, build_lane_detector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def clip_range(cap, part, clip_seconds):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if part <= 0:
        return 0, total, fps, total
    n = int(round(clip_seconds * fps))
    start = (part - 1) * n
    if start >= total:
        raise SystemExit(
            f"part {part} starts at frame {start} but the video has {total}"
        )
    return start, min(total, part * n), fps, total


def measure_frame(frame, lane_det, veh_det, state, args, abs_bounds=True,
                  plate_reader=None):
    """
    One frame: markings -> road plane -> ego lane -> lead vehicle.

    Returns a dict of raw (unsmoothed) measurements plus what is needed to draw.
    """
    h, w = frame.shape[:2]
    markings = lane_det(frame)

    vp = state["vpf"].update(lc.vanishing_point(markings, w, h))
    if vp is None:
        # No vanishing point means no ground plane, so nothing can be placed
        # in metres. Detection still runs: plates and track identity need no
        # geometry, and whole stretches of this footage have no painted
        # markings at all. Returning early here cost every vehicle and every
        # plate over 26 minutes of the source video.
        vehicles, _ = _detect_vehicles(frame, veh_det, plate_reader,
                                       None, None, None, None, args)
        return {"lane": None, "lead": None, "vehicles": vehicles, "gp": None,
                "left": None, "right": None, "n_mark": len(markings)}

    H = state["height"]
    gp = lc.GroundPlane(args.focal, w / 2.0, h / 2.0, vp[0], vp[1], H)

    # Every depth gate is expressed in camera heights, not metres. Depth scales
    # with height just as width does, so a fixed metre threshold means something
    # different at the nominal height used for calibration than at the final
    # one -- at 1 m the whole near field fell outside a 3 m limit and the ego
    # lane's own markings were discarded before selection ever ran.
    bounds = lc.merge_boundaries(
        lc.project_markings(markings, gp, args.z_near * H, args.z_far * H),
        tol=args.merge_tol * H,
    )
    # z_ref = 0 means "measure where this detector actually has support".
    z_ref = args.z_ref if args.z_ref > 0 else \
        lc.auto_z_ref(bounds, lo=3.0 * H, hi=12.0 * H, default=7.0 * H)
    z_lo = args.z_lo if args.z_lo > 0 else max(2.2 * H, 0.6 * z_ref)
    z_hi = args.z_hi if args.z_hi > 0 else 2.0 * z_ref
    max_half = 4.5 * H
    min_span = args.min_span * H

    # Once the lane pitch is known, place the expected grid of markings and
    # read the ego lane off it. Until then fall back to nearest-straddling to
    # bootstrap that pitch.
    ego = None
    pitch = args.lane_pitch if args.lane_pitch > 0 else state.get("pitch")
    ready = args.lane_pitch > 0 or len(state.get("lane_hist", [])) >= 12
    if pitch and ready:
        offs = lc.boundary_offsets(bounds, z_ref, max_half, min_span)
        ph = lc.grid_phase([x for x, _ in offs], pitch,
                           [b.n for _, b in offs])
        ph = _update_phase(state, ph, pitch, args.phase_alpha)
        ego = lc.resolve_ego_lane(bounds, z_ref, pitch, ph, max_half,
                                  min_span, H, tol_frac=args.slot_tol)

    if ego is not None:
        # A single frame where the detector failed to fire is not a missing
        # marking. Only report one after it has been absent for several
        # consecutive frames, so the flag means worn or absent paint rather
        # than detector flicker.
        for side in ("left", "right"):
            key = f"miss_run_{side}"
            if getattr(ego, f"missing_{side}"):
                state[key] = state.get(key, 0) + 1
            else:
                state[key] = 0
            if state[key] < args.missing_persist:
                setattr(ego, f"missing_{side}", False)
        left, right, lane = ego.left, ego.right, ego.width
    else:
        left, right = lc.select_ego_lane(bounds, z_ref, max_half=max_half,
                                         min_span=min_span)
        lane = lc.lane_width(left, right, z_ref, z_lo, z_hi)
    # Absolute metre bounds are only meaningful once the scale is known. During
    # the calibration pass the height is still a placeholder, and applying them
    # there rejects true single lanes while letting two-lane spans through --
    # which then becomes the calibrated scale.
    if abs_bounds and lane is not None:
        if not (state["lane_lo"] <= lane <= state["lane_hi"]):
            lane = None

    # When one edge is missed the next marking over gets picked instead, which
    # reads as a lane one whole lane too wide. Reject against the running
    # median rather than publish a two-lane span.
    if lane is not None:
        hist = state.setdefault("lane_hist", [])
        ref = state.get("lane_ref")
        # Only gate once there is enough history to trust, otherwise an early
        # bad frame would lock the reference and reject everything correct.
        if ref and len(hist) >= 15 and abs(lane - ref) > args.lane_tol * ref:
            lane = None
            left = right = None
            # A long reject streak means the road changed (lane change, wider
            # carriageway), not that every frame is bad -- drop the reference
            # and relearn, otherwise the gate deadlocks for the rest of the clip.
            state["reject_run"] = state.get("reject_run", 0) + 1
            if state["reject_run"] >= args.relearn_after:
                hist.clear()
                state["lane_ref"] = None
                state["reject_run"] = 0
        else:
            state["reject_run"] = 0
            hist.append(lane)
            del hist[:-45]
            state["lane_ref"] = float(np.median(hist))
            state["pitch"] = state["lane_ref"]

    vehicles, lead = _detect_vehicles(frame, veh_det, plate_reader, gp,
                                      left, right, z_ref, args)

    return {"lane": lane, "lead": lead, "vehicles": vehicles, "gp": gp,
            "left": left, "right": right, "n_mark": len(markings), "ego": ego,
            "pitch": state.get("pitch")}


def _detect_vehicles(frame, veh_det, plate_reader, gp, left, right, z_ref,
                     args):
    """
    Boxes, plates and the lead vehicle for one frame.

    gp None means no ground plane: boxes and plates still come back, but
    nothing is placed in metres and there is no lead vehicle to pick.
    """
    if veh_det is None:
        return [], None
    boxes, cls, scores, track_ids = veh_det(frame)
    if not len(boxes):
        return [], None

    vehicles = lc.measure_vehicles(boxes, cls, scores, gp,
                                   track_ids=track_ids)
    if plate_reader is not None and vehicles:
        # Plates are only legible close up, so skip the read entirely on
        # frames whose vehicles are all far away. That is most of an open-road
        # drive, and it costs nothing measurable. With no ground plane there
        # is no distance to gate on, so every such frame is read.
        zs = [v.Z for v in vehicles if v.Z is not None]
        if not zs or min(zs) <= args.plate_z_max:
            if args.plate_mode == "crops":
                plates = plate_reader.read_crops(
                    frame, vehicles, min_px=args.plate_min_px)
            else:
                plates = plate_reader(frame)
            ld.attach_plates(vehicles, plates)

    if gp is None:
        return vehicles, None
    lead = lc.pick_lead(vehicles, left, right, z_ref)
    if lead is not None and not (args.min_veh <= lead.width <= args.max_veh):
        lead = None
    return vehicles, lead


def _update_phase(state, ph, pitch, alpha=0.25):
    """
    Smooth the grid phase across frames.

    It shifts as the car drifts within its lane, so it is tracked rather than
    trusted per frame; the update wraps through the pitch so a crossing near
    the modulo boundary does not snap the grid a whole lane sideways.
    """
    if ph is None:
        return state.get("phase")
    prev = state.get("phase")
    if prev is None:
        state["phase"] = ph
        return ph
    d = (ph - prev + 0.5 * pitch) % pitch - 0.5 * pitch
    state["phase"] = (prev + alpha * d) % pitch
    return state["phase"]


def calibrate_from_vehicles(cap, start, end, lane_det, veh_det, state, args):
    """
    Solve camera height from the width of detected cars.

    Anchoring on a known lane width makes lane width true by construction,
    which is circular when lane width is the thing being asked about. Car
    bodies are a scale reference that is independent of the road, so anchoring
    on them leaves lane width as a real measurement.

    Only cars are used -- trucks and buses vary far more -- and only ones close
    enough, central enough and untruncated enough for their box edges to be
    trustworthy.
    """
    n = min(args.anchor_samples, max(1, (end - start) // 2))
    frames = np.linspace(start, end - 1, n).astype(int)
    prev_h = state["height"]
    state["height"] = 1.0
    state["vpf"] = lc.VPFilter()

    widths = []
    print(f"  calibrating scale from vehicles on {len(frames)} frames ...")
    for i, fi in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        markings = lane_det(frame)
        vp = state["vpf"].update(lc.vanishing_point(markings, w, h))
        if vp is None:
            continue
        gp = lc.GroundPlane(args.focal, w / 2.0, h / 2.0, vp[0], vp[1], 1.0)

        boxes, cls, scores, _ = veh_det(frame)
        for (x1, y1, x2, y2), c, s in zip(boxes, cls, scores):
            if int(c) != 2:                       # cars only
                continue
            if (x2 - x1) < 40:                    # too few pixels to trust
                continue
            if x1 < 8 or x2 > w - 8 or y2 > h - 8:  # truncated by the frame
                continue
            X, Z = gp.to_ground(np.array([x1, x2]), np.array([y2, y2]))
            if not np.all(np.isfinite(X)):
                continue
            z = float(np.mean(Z))
            if not (2.5 <= z <= 12.0):            # in camera heights
                continue
            if abs(float(np.mean(X))) > 3.0:
                continue
            widths.append(float(abs(X[1] - X[0])))
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(frames)}  cars={len(widths)}")

    state["vpf"] = lc.VPFilter()
    state["lane_ref"], state["lane_hist"] = None, []
    state["pitch"], state["phase"] = None, None

    if len(widths) < 15:
        print(f"  too few usable cars ({len(widths)}); keeping "
              f"camera-height={prev_h}")
        state["height"] = prev_h
        return prev_h, len(widths)

    med = float(np.median(widths))
    height = args.anchor_vehicle_width / med
    print(f"  median raw car width {med:.3f} (at 1 m) from {len(widths)} cars"
          f" -> camera height {height:.3f} m for a "
          f"{args.anchor_vehicle_width} m car")
    state["height"] = height
    return height, len(widths)


def calibrate_height(cap, start, end, lane_det, state, args):
    """
    Solve the effective camera height from a road of known lane width.

    Samples across the clip, measures lane width at a nominal height of 1 m,
    then rescales so the median lands on the supplied value.
    """
    n = min(args.anchor_samples, max(1, (end - start) // 2))
    frames = np.linspace(start, end - 1, n).astype(int)
    prev_h = state["height"]
    state["height"] = 1.0
    state["vpf"] = lc.VPFilter()
    state["lane_ref"], state["lane_hist"] = None, []
    state["pitch"], state["phase"] = None, None

    raw = []
    print(f"  calibrating scale on {len(frames)} frames ...")
    for i, fi in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        r = measure_frame(frame, lane_det, None, state, args,
                          abs_bounds=False)
        if r["lane"] is not None:
            raw.append(r["lane"])
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(frames)}  usable={len(raw)}")

    height, med = lc.scale_from_lane_width(raw, args.anchor_lane_width)
    state["vpf"] = lc.VPFilter()
    state["lane_ref"], state["lane_hist"] = None, []
    state["pitch"], state["phase"] = None, None
    if height is None:
        print("  calibration failed (too few usable frames); keeping "
              f"camera-height={prev_h}")
        state["height"] = prev_h
        return prev_h, len(raw)

    print(f"  median raw width {med:.3f} (at 1 m) -> "
          f"camera height {height:.3f} m for a {args.anchor_lane_width} m lane")
    state["height"] = height
    return height, len(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(ROOT, "data/videos/dashcam.mp4"))
    ap.add_argument("--output", default=os.path.join(ROOT, "data/outputs/lane_width"))
    ap.add_argument("--detector", default="yoloseg",
                    choices=["yoloseg", "laneseg", "clrnet", "yolopv2"])
    ap.add_argument("--part", type=int, default=5,
                    help="which clip to process (5 = last); 0 = whole video")
    ap.add_argument("--clip-seconds", type=float, default=300.0)
    ap.add_argument("--min-track-frames", type=int, default=3,
                    help="drop plate-less tracks seen fewer times than this; "
                         "they are nearly all ID switches")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="resume at this absolute frame instead of the part "
                         "boundary; temporal state restarts here")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="debug: stop after N frames")
    ap.add_argument("--stride", type=int, default=1,
                    help="run detection every Nth frame (held in between)")

    ap.add_argument("--lane-weights",
                    default=os.path.join(ROOT, "models/checkpoints/yolov8_lane_seg.pt"))
    ap.add_argument("--yolopv2-weights",
                    default=os.path.join(ROOT, "models/checkpoints/yolopv2.pt"))
    ap.add_argument("--laneseg-weights",
                    default=os.path.join(ROOT, "third_party/LaneSegACCESS/models/"
                                               "lane_seg_model_torchjit_serialized.pth"))
    ap.add_argument("--laneseg-thresh", type=float, default=0.30)
    ap.add_argument("--clrnet-weights",
                    default=os.path.join(ROOT, "models/checkpoints/clrnet_r18/culane_r18.pth"))
    ap.add_argument("--clrnet-root",
                    default=os.path.join(ROOT, "third_party/CLRNet"))
    ap.add_argument("--clrnet-conf", type=float, default=0.15,
                    help="CULane-trained model is under-confident on this "
                         "footage; 0.4 (its CULane default) yields nothing")
    ap.add_argument("--clrnet-max-lanes", type=int, default=8)
    ap.add_argument("--clrnet-nms", type=float, default=25.0)
    ap.add_argument("--clrnet-top", type=int, default=400,
                    help="top row of the road band fed to CLRNet")
    ap.add_argument("--vehicle-weights",
                    default=os.path.join(ROOT, "models/checkpoints/yolo11s.pt"))
    ap.add_argument("--no-vehicle", action="store_true")
    ap.add_argument("--device", default=0)
    ap.add_argument("--lane-conf", type=float, default=0.12)
    ap.add_argument("--lane-imgsz", type=int, default=640)
    ap.add_argument("--veh-conf", type=float, default=0.35)

    ap.add_argument("--camera-height", type=float, default=1.35,
                    help="camera height above the road in metres; every width "
                         "scales linearly with this")
    ap.add_argument("--focal", type=float, default=1000.0,
                    help="only affects the depth axis; width is insensitive")
    ap.add_argument("--anchor-lane-width", type=float, default=0.0,
                    help="known lane width of this road; solves camera height "
                         "from the clip instead of trusting --camera-height")
    ap.add_argument("--anchor-samples", type=int, default=200)

    ap.add_argument("--z-ref", type=float, default=0.0,
                    help="depth (m) to measure lane width at; 0 = adapt to "
                         "where the detector has support")
    ap.add_argument("--z-lo", type=float, default=0.0,
                    help="0 = derive from z-ref")
    ap.add_argument("--z-hi", type=float, default=0.0,
                    help="0 = derive from z-ref")
    ap.add_argument("--z-near", type=float, default=2.2,
                    help="near depth limit, in camera heights")
    ap.add_argument("--z-far", type=float, default=40.0,
                    help="far depth limit, in camera heights")
    ap.add_argument("--merge-tol", type=float, default=0.33,
                    help="lateral tolerance for grouping fragments of one "
                         "line, in camera heights")
    ap.add_argument("--min-span", type=float, default=0.9,
                    help="shortest depth run a boundary needs to count as an "
                         "ego-lane edge, in camera heights")
    ap.add_argument("--lane-pitch", type=float, default=0.0,
                    help="lane spacing in metres. Given, the expected-marking "
                         "grid uses it directly instead of bootstrapping from "
                         "nearest-straddling -- needed for a detector that "
                         "misses one ego edge, which would otherwise bootstrap "
                         "a two-lane pitch and lock the grid to it")
    ap.add_argument("--missing-persist", type=int, default=5,
                    help="consecutive frames a marking must be absent before "
                         "it is reported missing")
    ap.add_argument("--slot-tol", type=float, default=0.28,
                    help="how close a detected marking must sit to its "
                         "expected grid position, as a fraction of lane pitch")
    ap.add_argument("--phase-alpha", type=float, default=0.25,
                    help="smoothing on the lane-grid phase")
    ap.add_argument("--anchor-vehicle-width", type=float, default=0.0,
                    help="known width of a typical car (e.g. 1.80); calibrates "
                         "scale from detected vehicles instead of from an "
                         "assumed lane width, so lane width stays an "
                         "independent measurement")
    ap.add_argument("--relearn-after", type=int, default=40,
                    help="consecutive rejected frames before the lane-width "
                         "reference is relearned")
    ap.add_argument("--lane-tol", type=float, default=0.22,
                    help="reject a frame whose lane width differs from the "
                         "running median by more than this fraction")
    ap.add_argument("--min-lane", type=float, default=0.0,
                    help="0 = 0.60 x expected lane width")
    ap.add_argument("--max-lane", type=float, default=0.0,
                    help="0 = 1.50 x expected lane width")
    ap.add_argument("--min-veh", type=float, default=0.5)
    ap.add_argument("--max-veh", type=float, default=4.0)
    ap.add_argument("--no-plates", action="store_true",
                    help="skip license plate detection and OCR")
    ap.add_argument("--plate-z-max", type=float, default=25.0,
                    help="skip the plate read on frames whose nearest vehicle "
                         "is beyond this distance in metres; plates are not "
                         "legible that far out at 1080p")
    ap.add_argument("--plate-conf", type=float, default=0.4,
                    help="plate detector confidence threshold")
    ap.add_argument("--plate-mode", default="crops", choices=["crops", "frame"],
                    help="crops runs the plate detector on each vehicle box at "
                         "near-native resolution; frame runs it once on the "
                         "whole image, which is faster but loses small plates")
    ap.add_argument("--plate-min-conf", type=float, default=0.85,
                    help="a plate is only reported in the plate column above "
                         "this confidence; weaker reads stay in "
                         "plate_candidates rather than being dropped")
    ap.add_argument("--plate-min-votes", type=int, default=2,
                    help="a plate must also be read this many times across "
                         "the track; one low-confidence frame is noise")
    ap.add_argument("--plate-min-px", type=int, default=100,
                    help="crops mode: skip vehicle boxes narrower than this")
    ap.add_argument("--plate-detector",
                    default="yolo-v9-t-384-license-plate-end2end",
                    help="fast-alpr plate detector model name")
    ap.add_argument("--plate-ocr", default="cct-xs-v2-global-model",
                    help="fast-alpr OCR model name")
    ap.add_argument("--no-agnostic-nms", action="store_true",
                    help="keep per-class NMS on the vehicle detector; the "
                         "default merges boxes across classes so one vehicle "
                         "cannot appear as both a car and a truck")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"],
                    help="ByteTrack is markedly faster; BoT-SORT adds global "
                         "motion compensation, which costs more than it buys "
                         "on a forward-facing dashcam")
    ap.add_argument("--no-track", action="store_true",
                    help="disable vehicle tracking; without it the grouped "
                         "per-vehicle CSV cannot be written")

    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tag = args.detector

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    start, end, fps, total = clip_range(cap, args.part, args.clip_seconds)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_frame:
        # Resuming a killed run. Temporal state (vanishing point, learned
        # scale, smoothing) starts fresh here, so the first seconds after the
        # join settle rather than carrying over.
        start = args.start_frame
    if args.max_frames:
        end = min(end, start + args.max_frames)

    print("=" * 66)
    print(f"  lane width  |  detector={tag}")
    print(f"  {args.input}")
    print(f"  {w}x{h} @ {fps:.2f} fps   part {args.part}: frames {start}-{end}"
          f"  ({(end - start) / fps / 60:.2f} min)")
    print("=" * 66)

    lane_det = build_lane_detector(args.detector, args)
    veh_det = None
    if not args.no_vehicle:
        veh_det = VehicleDetector(args.vehicle_weights, args.device,
                                  conf=args.veh_conf,
                                  track=not args.no_track,
                                  tracker=args.tracker,
                                  agnostic_nms=not args.no_agnostic_nms)
    plate_reader = None
    if not args.no_plates and not args.no_vehicle:
        plate_reader = ld.PlateReader(detector=args.plate_detector,
                                      ocr=args.plate_ocr,
                                      conf=args.plate_conf, device=args.device)

    # Plausibility window scaled to the lane width we expect, so it tracks the
    # calibrated scale instead of assuming one.
    expect = args.anchor_lane_width if args.anchor_lane_width > 0 else 3.5
    state = {"vpf": lc.VPFilter(), "height": args.camera_height,
             "lane_ref": None, "lane_hist": [], "reject_run": 0,
             "pitch": None, "phase": None,
             "lane_lo": args.min_lane if args.min_lane > 0 else 0.60 * expect,
             "lane_hi": args.max_lane if args.max_lane > 0 else 1.50 * expect}

    anchor_used = 0
    anchor_kind = None
    if args.anchor_vehicle_width > 0:
        if veh_det is None:
            raise SystemExit("--anchor-vehicle-width needs the vehicle detector")
        anchor_kind = "vehicle"
        _, anchor_used = calibrate_from_vehicles(
            cap, start, end, lane_det, veh_det, state, args)
    elif args.anchor_lane_width > 0:
        anchor_kind = "lane"
        _, anchor_used = calibrate_height(cap, start, end, lane_det, state, args)

    writer = None
    if not args.no_video:
        writer = lc.VideoWriter(
            os.path.join(args.output, f"clip{args.part}_{tag}.mp4"),
            w, h, fps, crf=args.crf,
        )

    csv_path = os.path.join(args.output, f"clip{args.part}_{tag}.csv")
    cf = open(csv_path, "w", newline="")
    cw = csv.writer(cf)
    cw.writerow([
        "frame", "time_s", "lane_width_m", "lane_width_m_smooth",
        "lead_width_m", "lead_width_m_smooth", "lead_distance_m", "lead_class",
        "n_markings", "lane_live", "lead_live",
        "marking_missing", "ego_left_present", "ego_right_present",
        "slots_total", "slots_missing",
        "conf_left", "conf_middle", "conf_right",
        "width_left_m", "width_middle_m", "width_right_m", "n_vehicles",
    ])

    # One JSON object per frame rather than one array for the clip: a full-length
    # run is ~200k frames, which no dashboard wants to parse as a single document.
    json_path = os.path.join(args.output, f"clip{args.part}_{tag}.jsonl")
    jf = open(json_path, "w")

    # One entry per tracked vehicle, accumulated across every frame it appears
    # in and written out at the end as the grouped CSV.
    tracks = {}

    lane_sm = lc.Smoother(alpha=0.15, window=21, reject_rel=0.18, holdout=25)
    veh_sm = lc.Smoother(alpha=0.22, window=15, reject_rel=0.25, holdout=12)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    reader = lc.FrameReader(cap)
    fi = start
    t0 = time.time()
    n_lane = n_lead = 0
    last = None
    prev_lead_z = None

    while fi < end:
        ok, frame = reader.read()
        if not ok:
            break

        if (fi - start) % args.stride == 0 or last is None:
            last = measure_frame(frame, lane_det, veh_det, state, args,
                                 plate_reader=plate_reader)
        r = last

        ego = r.get("ego")
        lead = r["lead"]
        # A big jump in range means this is a different vehicle, so the width
        # history from the previous one must not bleed into it.
        if lead is not None:
            if prev_lead_z is not None and abs(lead.Z - prev_lead_z) > 12.0:
                veh_sm.reset()
            prev_lead_z = lead.Z
        else:
            prev_lead_z = None

        lane_v, lane_live = lane_sm.update(r["lane"])
        veh_v, veh_live = veh_sm.update(lead.width if lead else None)
        n_lane += r["lane"] is not None
        n_lead += lead is not None

        if writer is not None:
            vis = frame
            if r["gp"] is not None:
                lc.draw_boundary(vis, r["left"], r["gp"], lc.LANE_COLOR)
                lc.draw_boundary(vis, r["right"], r["gp"], lc.LANE_COLOR)
                if ego is not None and ego.missing:
                    # Show where the absent marking was expected, shifted off
                    # whichever edge was found, so a reviewer can judge whether
                    # the paint is worn or was never there.
                    p = ego.pitch
                    if ego.missing_right and ego.left is not None:
                        lc.draw_missing(vis, lc.offset_boundary(ego.left, p),
                                        r["gp"], p)
                    if ego.missing_left and ego.right is not None:
                        lc.draw_missing(vis, lc.offset_boundary(ego.right, -p),
                                        r["gp"], p)
            for v in r["vehicles"]:
                x1, y1, x2, y2 = [int(b) for b in v.box]
                is_lead = lead is not None and v is lead
                col = lc.VEH_COLOR if is_lead else lc.VEH_OTHER_COLOR
                cv2.rectangle(vis, (x1, y1), (x2, y2), col,
                              lc._thick(2 if is_lead else 1))
                label = f"{x2 - x1}px"
                if v.width is not None:
                    label += f" {v.width:.2f}m"
                if v.plate:
                    label += f"  {v.plate}"
                fs = 0.45 * lc.UI_SCALE
                th = lc._thick(1)
                # Dark stroke behind the glyphs: the label often lands on pale
                # road or sky, where the box colour alone disappears.
                for c, t in (((0, 0, 0), th + 2), (col, th)):
                    cv2.putText(vis, label,
                                (x1, max(lc._px(12), y1 - lc._px(5))),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, c, t,
                                cv2.LINE_AA)
            lc.draw_readout(vis, lane_v, veh_v,
                            lc.VEHICLE_CLASSES.get(lead.cls, "Vehicle").title()
                            if lead else None,
                            missing=ego.missing if ego else "",
                            lanes=ego.lanes if ego else None)
            writer.write(vis)

        cw.writerow([
            fi, round(fi / fps, 3),
            round(r["lane"], 4) if r["lane"] else "",
            round(lane_v, 4) if lane_v else "",
            round(lead.width, 4) if lead else "",
            round(veh_v, 4) if veh_v else "",
            round(lead.Z, 2) if lead else "",
            lc.VEHICLE_CLASSES.get(lead.cls, "") if lead else "",
            r["n_mark"], int(lane_live), int(veh_live),
            ego.missing if ego else "",
            "" if ego is None else int(not ego.missing_left),
            "" if ego is None else int(not ego.missing_right),
            "" if ego is None else len(ego.slots),
            "" if ego is None else sum(1 for _, ok in ego.slots if not ok),
            *([round(l.conf, 3) for l in ego.lanes] if ego else ["", "", ""]),
            *([round(l.width, 4) if l.width else "" for l in ego.lanes]
              if ego else ["", "", ""]),
            len(r["vehicles"]),
        ])

        for v in r["vehicles"]:
            if v.track_id is None:
                continue
            t = tracks.get(v.track_id)
            if t is None:
                t = tracks[v.track_id] = {
                    "first": fi, "last": fi, "n": 0, "cls": {},
                    "w_px": [], "w_m": [], "w_ok": [], "z_min": None,
                    "lead": 0, "plates": {},
                }
            t["last"] = fi
            t["n"] += 1
            t["cls"][v.cls] = t["cls"].get(v.cls, 0) + 1
            t["w_px"].append(int(v.box[2] - v.box[0]))
            if v.width is not None:
                t["w_m"].append(v.width)
                # Kept apart from w_m so the reported width comes only from
                # where the projection holds up, while the raw spread stays
                # visible in its own column.
                if lc.width_is_reliable(v.Z, v.X):
                    t["w_ok"].append(v.width)
            if v.Z is not None:
                t["z_min"] = v.Z if t["z_min"] is None \
                    else min(t["z_min"], v.Z)
            if lead is not None and v is lead:
                t["lead"] += 1
            if v.plate:
                # Vote by summed confidence, not by count: three blurred
                # agreeing misreads should not outweigh one clean read.
                p = t["plates"].setdefault(v.plate, [0.0, 0, 0.0])
                p[0] += v.plate_conf
                p[1] += 1
                p[2] = max(p[2], v.plate_conf)

        jf.write(json.dumps({
            "frame": fi,
            "time_s": round(fi / fps, 3),
            "lane_width_m": round(r["lane"], 4) if r["lane"] else None,
            "lane_width_m_smooth": round(lane_v, 4) if lane_v else None,
            "marking_missing": (ego.missing or None) if ego else None,
            "vehicles": [{
                "track_id": v.track_id,
                "box": [int(b) for b in v.box],
                "width_px": int(v.box[2] - v.box[0]),
                "width_m": None if v.width is None else round(v.width, 4),
                "distance_m": None if v.Z is None else round(v.Z, 2),
                "offset_m": None if v.X is None else round(v.X, 2),
                "width_reliable": lc.width_is_reliable(v.Z, v.X),
                "cls": lc.VEHICLE_CLASSES.get(v.cls, "unknown"),
                "score": round(float(v.score), 3),
                "plate": v.plate,
                "plate_conf": round(v.plate_conf, 3) if v.plate else None,
                "is_lead": lead is not None and v is lead,
            } for v in r["vehicles"]],
        }) + "\n")

        fi += 1
        done = fi - start
        if done % 100 == 0:
            el = time.time() - t0
            rate = done / max(el, 1e-3)
            eta = (end - fi) / max(rate, 1e-3)
            sys.stdout.write(
                f"\r  {done}/{end - start}  {100 * done / (end - start):5.1f}%"
                f"  lane {100 * n_lane / done:4.1f}%  lead {100 * n_lead / done:4.1f}%"
                f"  {rate:.1f} fps  ETA {eta / 60:.1f} min   "
            )
            sys.stdout.flush()

    reader.release()
    cap.release()
    cf.close()
    jf.close()
    if writer is not None:
        writer.release()

    # A plate identifies a vehicle, so two tracks that agree on a confident
    # plate are one vehicle the tracker lost and re-acquired. Merge them.
    accepted = {}
    for tid, t in tracks.items():
        if not t["plates"]:
            continue
        txt, (score, votes, conf) = max(t["plates"].items(),
                                        key=lambda kv: kv[1][0])
        if conf >= args.plate_min_conf and votes >= args.plate_min_votes:
            accepted.setdefault(txt, []).append(tid)

    merged_into = {}
    for txt, tids in accepted.items():
        if len(tids) < 2:
            continue
        keep = min(tids)
        for tid in tids:
            if tid == keep:
                continue
            merged_into[tid] = keep
            a, b = tracks[keep], tracks.pop(tid)
            a["first"] = min(a["first"], b["first"])
            a["last"] = max(a["last"], b["last"])
            a["n"] += b["n"]
            zs = [z for z in (a["z_min"], b["z_min"]) if z is not None]
            a["z_min"] = min(zs) if zs else None
            a["lead"] += b["lead"]
            a["w_px"] += b["w_px"]
            a["w_m"] += b["w_m"]
            a["w_ok"] += b["w_ok"]
            for k, v in b["cls"].items():
                a["cls"][k] = a["cls"].get(k, 0) + v
            for k, v in b["plates"].items():
                p = a["plates"].setdefault(k, [0.0, 0, 0.0])
                p[0] += v[0]
                p[1] += v[1]
                p[2] = max(p[2], v[2])
    merged_from = {}
    for src, dst in merged_into.items():
        merged_from.setdefault(dst, []).append(src)

    veh_path = os.path.join(args.output, f"clip{args.part}_{tag}_vehicles.csv")
    with open(veh_path, "w", newline="") as vf:
        vw = csv.writer(vf)
        vw.writerow([
            "track_id", "cls", "first_frame", "last_frame", "first_time_s",
            "last_time_s", "n_frames", "plate", "plate_conf", "plate_votes",
            "plate_candidates", "width_px_median", "width_px_max",
            "width_m_median", "width_m_samples", "width_m_median_all",
            "min_distance_m", "frames_as_lead", "merged_track_ids",
        ])
        n_short = 0
        for tid in sorted(tracks):
            t = tracks[tid]
            # A track seen for a frame or two is nearly always an ID switch or
            # a one-frame false positive. Dropped only when it carries no
            # plate, since a plate makes even a brief sighting worth a row.
            if t["n"] < args.min_track_frames and not t["plates"]:
                n_short += 1
                continue
            cls = max(t["cls"].items(), key=lambda kv: kv[1])[0]
            plate = conf = votes = ""
            cands = ""
            if t["plates"]:
                best = max(t["plates"].items(), key=lambda kv: kv[1][0])
                text, (score, votes, conf) = best[0], best[1]
                # Below the gate the string is noise: a single blurred read of
                # a plate that was never legible. It stays in the candidate
                # list, but the plate column keeps only what is trustworthy.
                if conf >= args.plate_min_conf and votes >= args.plate_min_votes:
                    plate = text
                conf = round(conf, 3)
                # Every string that was ever read for this vehicle, so a
                # disagreement is visible rather than silently resolved.
                # Capped: a plate glimpsed through 20 blurred frames produces
                # 20 one-vote misreads, which bury the reads that matter.
                cands = "|".join(
                    f"{txt}:{v[1]}"
                    for txt, v in sorted(t["plates"].items(),
                                         key=lambda kv: -kv[1][0])[:8]
                )
            vw.writerow([
                tid, lc.VEHICLE_CLASSES.get(cls, "unknown"),
                t["first"], t["last"],
                round(t["first"] / fps, 3), round(t["last"] / fps, 3),
                t["n"], plate, conf, votes, cands,
                int(np.median(t["w_px"])), max(t["w_px"]),
                round(float(np.median(t["w_ok"])), 4) if t["w_ok"] else "",
                len(t["w_ok"]),
                round(float(np.median(t["w_m"])), 4) if t["w_m"] else "",
                "" if t["z_min"] is None else round(t["z_min"], 2),
                t["lead"],
                "|".join(str(i) for i in sorted(merged_from.get(tid, []))),
            ])
    n_conf = 0
    for t in tracks.values():
        if not t["plates"]:
            continue
        score, votes, conf = max(t["plates"].values(), key=lambda v: v[0])
        if conf >= args.plate_min_conf and votes >= args.plate_min_votes:
            n_conf += 1
    n_width = sum(1 for t in tracks.values() if t["w_ok"])
    print(f"\n  tracked {len(tracks)} vehicles, {n_conf} with a confident "
          f"plate, {len(merged_into)} tracks merged by plate")
    print(f"  {n_width} with a reliable width, {n_short} short tracks dropped "
          f"(<{args.min_track_frames} frames, no plate)")

    n = max(1, fi - start)
    lanes = [float(x) for x in _col(csv_path, "lane_width_m")]
    leads = [float(x) for x in _col(csv_path, "lead_width_m")]
    summary = {
        "detector": tag,
        "part": args.part,
        "frames": n,
        "camera_height_m": round(state["height"], 4),
        "lane_pitch_m": args.lane_pitch or None,
        "anchor": anchor_kind,
        "anchored_lane_width_m": args.anchor_lane_width or None,
        "anchored_vehicle_width_m": args.anchor_vehicle_width or None,
        "anchor_samples_used": anchor_used or None,
        "lane_detection_rate": round(n_lane / n, 4),
        "lead_detection_rate": round(n_lead / n, 4),
        "lane_width_m": _stats(lanes),
        "lead_width_m": _stats(leads),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, f"clip{args.part}_{tag}_summary.json"),
              "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 66)
    print(json.dumps(summary, indent=2))
    print(f"  csv   {csv_path}")
    print(f"  json  {json_path}")
    print(f"  veh   {veh_path}")
    if writer is not None:
        print(f"  video {writer.path}")


def _col(path, name):
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(name, "")
            if v not in ("", None):
                yield v


def _stats(vals):
    if not vals:
        return None
    a = np.asarray(vals, float)
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 3),
        "mean": round(float(a.mean()), 3),
        "p10": round(float(np.percentile(a, 10)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "std": round(float(a.std()), 3),
    }


if __name__ == "__main__":
    main()
