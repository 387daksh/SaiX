"""
Lane-marking detectors.

Each detector returns a list of `lane_core.Marking` -- one entry per physical
marking instance, holding that marking's pixels. The geometry layer does not
care which detector produced them.
"""

import os

import cv2
import numpy as np

from lane_core import Marking

_KIND = {0: "solid", 1: "dashed"}


def _subsample(px, py, cap=420):
    if len(px) <= cap:
        return px, py
    idx = np.random.default_rng(0).choice(len(px), cap, replace=False)
    return px[idx], py[idx]


class YoloSegDetector:
    """
    YOLOv8 instance segmentation trained on lane markings (solid / dashed).

    Runs at 640 on purpose: at 1280 the model only fires on thin far-field
    paint and misses the wide near-field markings that matter most for width.
    """

    name = "yoloseg"

    def __init__(self, weights, device=0, conf=0.12, imgsz=640, iou=0.45):
        from ultralytics import YOLO

        if not os.path.exists(weights):
            raise FileNotFoundError(f"lane-seg weights not found: {weights}")
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou

    def __call__(self, frame):
        r = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False, retina_masks=False,
        )[0]
        if r.masks is None or r.boxes is None:
            return []

        h, w = frame.shape[:2]
        out = []
        polys = r.masks.xy
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()

        for poly, c, s in zip(polys, cls, scores):
            if poly is None or len(poly) < 3:
                continue
            p = np.asarray(poly, np.float32)
            x0 = int(max(0, np.floor(p[:, 0].min())))
            y0 = int(max(0, np.floor(p[:, 1].min())))
            x1 = int(min(w, np.ceil(p[:, 0].max()) + 1))
            y1 = int(min(h, np.ceil(p[:, 1].max()) + 1))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue

            local = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(local, [np.round(p - [x0, y0]).astype(np.int32)], 1)
            ys, xs = np.nonzero(local)
            if len(xs) < 10:
                continue

            px = xs.astype(np.float32) + x0
            py = ys.astype(np.float32) + y0
            px, py = _subsample(px, py)
            out.append(Marking(px, py, _KIND.get(int(c), "unknown"), float(s)))
        return out


class YolopV2Detector:
    """
    YOLOPv2 lane-line head.

    This is a dense binary segmentation model, so instances are recovered by
    connected components rather than predicted directly.
    """

    name = "yolopv2"

    def __init__(self, weights, device=0, min_area=40, imgsz=640):
        import torch

        if not os.path.exists(weights):
            raise FileNotFoundError(f"YOLOPv2 weights not found: {weights}")
        self.torch = torch
        self.dev = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.jit.load(weights, map_location=self.dev)
        self.half = self.dev.type == "cuda"
        if self.half:
            self.model = self.model.half()
        self.model = self.model.to(self.dev).eval()
        self.min_area = min_area
        self.imgsz = imgsz

    def __call__(self, frame):
        torch = self.torch
        oh, ow = frame.shape[:2]
        s = self.imgsz / max(oh, ow)
        nw, nh = int(round(ow * s)), int(round(oh * s))
        dw, dh = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(frame, (nw, nh))
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        t = t.to(self.dev)
        if self.half:
            t = t.half()

        with torch.no_grad():
            out = self.model(t)
        ll = out[2][..., dh:dh + nh, dw:dw + nw]
        if ll.shape[1] == 2:
            m = torch.argmax(ll, 1)[0]
        else:
            m = (torch.sigmoid(ll)[0, 0] > 0.5)
        mask = (m.float().cpu().numpy() * 255).astype(np.uint8)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        res = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                continue
            ys, xs = np.nonzero(lbl == i)
            if len(xs) < 10:
                continue
            px, py = _subsample(xs.astype(np.float32), ys.astype(np.float32))
            res.append(Marking(px, py, "unknown", 1.0))
        return res


class LaneSegAccessDetector:
    """
    ACCESSLab real-time lane segmentation (Getahun & Karimoddini, IEEE T-IV 2026).

    A lightweight binary lane-vs-background segmenter shipped as TorchScript, so
    no architecture needs reconstructing. Like the YOLOPv2 head it predicts a
    dense mask rather than instances, so lines are recovered by connected
    components and regrouped on the road plane.

    Preprocessing follows the repo's own `test_model.py`: plain resize to
    640x384 (no letterbox), RGB, /255, then sigmoid on the logits.
    """

    name = "laneseg"

    def __init__(self, weights, device=0, thresh=0.30, min_area=40,
                 size=(640, 384)):
        import torch

        if not os.path.exists(weights):
            raise FileNotFoundError(f"ACCESSLab weights not found: {weights}")
        self.torch = torch
        self.dev = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.jit.load(weights, map_location=self.dev).eval()
        self.model = self.model.to(self.dev)
        self.thresh = thresh
        self.min_area = min_area
        self.size = size

    def __call__(self, frame):
        torch = self.torch
        h, w = frame.shape[:2]
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        t = t.to(self.dev)

        with torch.no_grad():
            p = torch.sigmoid(self.model(t)).squeeze().cpu().numpy()

        mask = (p > self.thresh).astype(np.uint8)

        # Label at the model's own 640x384 rather than upscaling first: a
        # full-resolution `lbl == i` scans two million pixels per component and
        # dominated the frame time (~3 fps against ~25 for the rest). The points
        # are subsampled anyway, so the extra resolution bought nothing.
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        sx = w / float(self.size[0])
        sy = h / float(self.size[1])
        area_small = self.min_area / (sx * sy)

        res = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < area_small:
                continue
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            sub = lbl[y0:y0 + ch, x0:x0 + cw] == i
            ys, xs = np.nonzero(sub)
            if len(xs) < 10:
                continue
            # Per-marking confidence, which a binary mask would otherwise throw
            # away. The 90th percentile reads the component's core rather than
            # its mean: soft edges pull the mean down to ~0.69 even for lines
            # that are plainly correct, which makes the number unreadable as a
            # confidence.
            score = float(np.percentile(p[y0:y0 + ch, x0:x0 + cw][sub], 90))
            px = (xs + x0).astype(np.float32) * sx
            py = (ys + y0).astype(np.float32) * sy
            px, py = _subsample(px, py)
            res.append(Marking(px, py, "unknown", score))
        return res


class ClrNetDetector:
    """
    Official CLRNet (ResNet-18, CULane) run through `clrnet_shim`.

    Unlike the segmentation detectors this one predicts lane *instances*
    directly as ordered point sets, so no connected-component or dash-merging
    step is needed before the geometry stage.

    The model is trained on CULane's road crop: the bottom 320 rows of a
    1640x590 frame, aspect ~2.78. Squashing a full 16:9 frame into that shape
    compresses the scene vertically by 0.55x against 0.85x horizontally and the
    model then only fires on short near-field stubs. Instead a horizon-to-bottom
    band of matching aspect is cut and fed directly, with `cut_height` set to 0
    so the row mapping stays a straight linear rescale of that band.
    """

    name = "clrnet"

    def __init__(self, weights, clrnet_root, device=0, conf=0.4, max_lanes=4,
                 nms_thres=50, band_top=400):
        import clrnet_shim
        from addict import Dict

        import torch

        clrnet_shim.install(clrnet_root)
        self.torch = torch
        self.dev = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )

        cfg = Dict(dict(
            net=dict(type="Detector"),
            backbone=dict(type="ResNetWrapper", resnet="resnet18",
                          pretrained=False,
                          replace_stride_with_dilation=[False, False, False],
                          out_conv=False),
            neck=dict(type="FPN", in_channels=[128, 256, 512],
                      out_channels=64, num_outs=3, attention=False),
            heads=dict(type="CLRHead", num_priors=192, refine_layers=3,
                       fc_hidden_dim=64, sample_points=36),
            num_points=72, max_lanes=max_lanes, num_classes=5,
            sample_y=range(589, 230, -20),
            img_w=800, img_h=320, ori_img_w=1640, ori_img_h=590,
            cut_height=0, featuremap_out_channel=192,
            ignore_label=255, bg_weight=0.4,
            test_parameters=dict(conf_threshold=conf, nms_thres=nms_thres,
                                 nms_topk=max_lanes),
        ))
        keys = set(cfg.keys())
        cfg.haskey = lambda k: k in keys
        self.cfg = cfg
        self.band_top = band_top

        from clrnet.models.registry import build_net

        net = build_net(cfg)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd["net"] if "net" in sd else sd
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"CLRNet checkpoint mismatch: {unexpected[:5]}")
        self.net = net.to(self.dev).eval()

    def _band(self, h):
        return max(0, min(self.band_top, h - 64)), h

    def _preprocess(self, frame):
        torch = self.torch
        c = self.cfg
        h = frame.shape[0]
        y0, y1 = self._band(h)
        img = cv2.resize(frame[y0:y1], (c.img_w, c.img_h),
                         interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        return t.to(self.dev)

    def __call__(self, frame):
        torch = self.torch
        h, w = frame.shape[:2]
        with torch.no_grad():
            out = self.net(self._preprocess(frame))
            lanes = self.net.heads.get_lanes(out)[0]

        res = []
        for lane in lanes:
            pts = np.asarray(lane.points, np.float32)
            if len(pts) < 6:
                continue
            # x is a fraction of frame width; y is a fraction of the band.
            y0, y1 = self._band(h)
            px = pts[:, 0] * w
            py = y0 + pts[:, 1] * (y1 - y0)
            ok = np.isfinite(px) & np.isfinite(py) & (px > -w) & (px < 2 * w)
            if ok.sum() < 6:
                continue
            res.append(Marking(px[ok], py[ok], "lane",
                               float(lane.metadata.get("conf", 1.0))))
        return res


class VehicleDetector:
    """COCO detector restricted to road vehicles, with persistent track ids."""

    def __init__(self, weights, device=0, conf=0.35, imgsz=960, track=True,
                 tracker="botsort.yaml", agnostic_nms=False):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.track = track
        self.tracker = tracker
        # NMS runs per class by default, so one pickup can survive as both a
        # "truck" box and a "car" box and then pick up two track ids.
        self.agnostic_nms = agnostic_nms
        self.keep = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def __call__(self, frame):
        """Return boxes, classes, scores, track ids. Ids are None untracked."""
        if self.track:
            # persist=True keeps the tracker state between calls, which is what
            # makes an id mean the same vehicle from one frame to the next.
            r = self.model.track(
                frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                classes=self.keep, tracker=self.tracker, persist=True,
                agnostic_nms=self.agnostic_nms, verbose=False,
            )[0]
        else:
            r = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                classes=self.keep, agnostic_nms=self.agnostic_nms,
                verbose=False,
            )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], [], [], []
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        s = r.boxes.conf.cpu().numpy()
        # A tracked box has no id on the frame it appears, before the tracker
        # has confirmed it, so the column can be missing or hold gaps.
        if r.boxes.id is None:
            ids = [None] * len(b)
        else:
            ids = [int(i) for i in r.boxes.id.cpu().numpy()]
        return b, c, s, ids


class PlateReader:
    """License plate detection plus OCR, matched back to vehicle boxes.

    Wraps fast-alpr (MIT): a YOLOv9-t plate detector followed by a CCT OCR
    model, both ONNX.

    Two modes. Whole-frame detection is one inference per frame, but the
    detector runs at 384 px, so a plate 30 px wide in a 1920-wide frame is
    6 px by the time it is seen and nothing is read. read_crops() instead runs
    the detector per vehicle box at near-native resolution: slower per frame,
    but on a 700-frame sample it found 32 plates against 4 for whole-frame at
    the same size, which is why it is the default. Either way containment
    decides which vehicle a plate belongs to.
    """

    def __init__(self, detector="yolo-v9-t-384-license-plate-end2end",
                 ocr="cct-xs-v2-global-model", conf=0.4, device=0):
        from fast_alpr import ALPR

        # Pin CUDA. The default list leads with TensorRT, and when its DLL
        # fails to load onnxruntime falls back to CPU rather than to CUDA.
        #
        # This asks for CUDA but cannot supply it: onnxruntime-gpu builds
        # against CUDA 12 and will fall back to CPU when only CUDA 13 is
        # installed (torch 2.12+cu132 ships cublasLt64_13.dll, not _12). The
        # fallback is silent apart from a stderr warning, so check the log for
        # "Failed to create CUDAExecutionProvider" before blaming throughput
        # on the GPU.
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device != "cpu" else ["CPUExecutionProvider"])
        self.alpr = ALPR(detector_model=detector, ocr_model=ocr,
                         detector_conf_thresh=conf,
                         detector_providers=providers, ocr_providers=providers)

    def __call__(self, frame, origin=(0, 0)):
        """Return [(text, conf, (x1, y1, x2, y2)), ...] for the frame.

        origin offsets the boxes back into the full frame when the caller
        passed a crop rather than the whole image.
        """
        ox, oy = origin
        out = []
        for r in self.alpr.predict(frame):
            if r.ocr is None or not r.ocr.text:
                continue
            cs = r.ocr.confidence
            # Per-character confidences: the weakest character decides whether
            # the whole string can be trusted.
            conf = float(min(cs)) if isinstance(cs, (list, tuple)) else float(cs)
            b = r.detection.bounding_box
            out.append((r.ocr.text, conf,
                        (b.x1 + ox, b.y1 + oy, b.x2 + ox, b.y2 + oy)))
        return out

    def read_crops(self, frame, vehicles, min_px=100, pad=8):
        """Run the plate detector on each vehicle box instead of the frame.

        The detector takes a fixed 384px input, so a full 1080p frame is
        downscaled 2.8x before it looks for a plate that was only ~60px wide
        to begin with. Feeding it one vehicle at a time keeps the plate close
        to native resolution, at the cost of one inference per vehicle.
        """
        h, w = frame.shape[:2]
        out = []
        for v in vehicles:
            x1, y1, x2, y2 = (int(c) for c in v.box)
            if x2 - x1 < min_px:
                continue
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                out.extend(self(crop, origin=(x1, y1)))
        return out


def attach_plates(vehicles, plates):
    """Give each plate to the smallest vehicle box that contains its centre.

    Smallest wins because a motorcycle in front of a truck sits inside the
    truck's box too, and the tighter box is the real owner.
    """
    for text, conf, (px1, py1, px2, py2) in plates:
        cx, cy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
        best, best_area = None, None
        for v in vehicles:
            x1, y1, x2, y2 = v.box
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = v, area
        if best is not None and conf > best.plate_conf:
            best.plate, best.plate_conf = text, conf


def build_lane_detector(kind, args):
    if kind == "yoloseg":
        return YoloSegDetector(args.lane_weights, args.device,
                               conf=args.lane_conf, imgsz=args.lane_imgsz)
    if kind == "clrnet":
        return ClrNetDetector(args.clrnet_weights, args.clrnet_root,
                              args.device, conf=args.clrnet_conf,
                              max_lanes=args.clrnet_max_lanes,
                              nms_thres=args.clrnet_nms,
                              band_top=args.clrnet_top)
    if kind == "laneseg":
        return LaneSegAccessDetector(args.laneseg_weights, args.device,
                                     thresh=args.laneseg_thresh)
    if kind == "yolopv2":
        return YolopV2Detector(args.yolopv2_weights, args.device)
    raise ValueError(f"unknown lane detector: {kind}")
