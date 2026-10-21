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
    """COCO detector restricted to road vehicles."""

    def __init__(self, weights, device=0, conf=0.35, imgsz=960):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.keep = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def __call__(self, frame):
        r = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            classes=self.keep, verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], [], []
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        s = r.boxes.conf.cpu().numpy()
        return b, c, s


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
