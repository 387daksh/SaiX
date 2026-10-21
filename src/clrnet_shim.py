"""
Run official CLRNet inference without mmcv or its compiled CUDA ops.

CLRNet's training stack pulls in mmcv-full, which has no wheel for this
torch/Python/Windows combination and does not build here. At inference time it
only actually needs two things:

  * ``mmcv.cnn.ConvModule``  -- a conv/norm/activation block
  * ``clrnet.ops.nms``       -- Lane-NMS, shipped as a CUDA extension

Both are reproduced here. The ConvModule replica keeps mmcv's submodule names
(``conv`` / ``bn`` / ``activate``) so pretrained checkpoint keys still match,
and the NMS is a faithful port of the LaneATT kernel CLRNet vendors.
"""

import sys
import types

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# mmcv.cnn.ConvModule
# ---------------------------------------------------------------------------

class ConvModule(nn.Module):
    """Conv -> norm -> activation, with mmcv's attribute names preserved."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias="auto",
                 conv_cfg=None, norm_cfg=None, act_cfg=None, inplace=True,
                 order=("conv", "norm", "act"), **kwargs):
        super().__init__()
        if conv_cfg is not None and conv_cfg.get("type", "Conv2d") != "Conv2d":
            raise NotImplementedError(f"conv_cfg {conv_cfg} not supported")

        self.with_norm = norm_cfg is not None
        # mmcv drops the conv bias whenever a norm layer follows it.
        if bias == "auto":
            bias = not self.with_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        self.norm_name = None
        if self.with_norm:
            ntype = norm_cfg.get("type", "BN")
            cfg = {k: v for k, v in norm_cfg.items() if k != "type"}
            if ntype == "BN":
                norm = nn.BatchNorm2d(out_channels, **cfg)
                self.norm_name = "bn"
            elif ntype in ("GN", "GroupNorm"):
                norm = nn.GroupNorm(num_channels=out_channels, **cfg)
                self.norm_name = "gn"
            else:
                raise NotImplementedError(f"norm type {ntype} not supported")
            self.add_module(self.norm_name, norm)

        # mmcv defaults to ReLU when act_cfg is omitted; act_cfg=None means none.
        self.with_activation = act_cfg is None or act_cfg.get("type") == "ReLU"
        if act_cfg is not None and act_cfg.get("type") not in (None, "ReLU"):
            raise NotImplementedError(f"act_cfg {act_cfg} not supported")
        if self.with_activation:
            self.activate = nn.ReLU(inplace=inplace)

    @property
    def norm(self):
        return getattr(self, self.norm_name) if self.norm_name else None

    def forward(self, x, activate=True, norm=True):
        x = self.conv(x)
        if self.with_norm and norm:
            x = getattr(self, self.norm_name)(x)
        if self.with_activation and activate:
            x = self.activate(x)
        return x


# ---------------------------------------------------------------------------
# Lane NMS
# ---------------------------------------------------------------------------

def lane_nms(proposals, scores, overlap=50.0, top_k=4):
    """
    Greedy NMS over lane proposals, matching CLRNet's CUDA kernel.

    `proposals` columns are [cls_bg, cls_fg, start_y, start_x, length, xs...]
    with start_y normalised, length in strips and xs in pixels. Two lanes are
    compared by their mean |x| gap over the rows where both are defined; lanes
    that never overlap in y are never suppressed.
    """
    device = proposals.device
    n, dim = proposals.shape
    n_offsets = dim - 5
    n_strips = n_offsets - 1

    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=device), 0, None

    order = torch.argsort(scores, descending=True)
    starts = torch.clamp(
        (proposals[:, 2] * n_strips + 0.5).floor().long(), 0, n_strips
    )
    lengths = (proposals[:, 4] + 0.5).floor().long()
    xs = proposals[:, 5:]

    keep = []
    for idx in order.tolist():
        suppressed = False
        sa = int(starts[idx])
        ea = min(sa + int(lengths[idx]) - 1, n_offsets - 1)
        for k in keep:
            sb = int(starts[k])
            eb = min(sb + int(lengths[k]) - 1, n_offsets - 1)
            s = max(sa, sb)
            e = min(ea, eb)
            if e - s < 0:
                continue
            d = (xs[idx, s:e + 1] - xs[k, s:e + 1]).abs().mean()
            if float(d) < overlap:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)
            if len(keep) >= top_k:
                break

    return torch.tensor(keep, dtype=torch.long, device=device), len(keep), None


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install(clrnet_root):
    """
    Put the stand-ins on sys.modules, then make CLRNet importable.

    Must run before any `clrnet.*` import so the real mmcv lookup never happens.
    """
    if "mmcv" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        cnn = types.ModuleType("mmcv.cnn")
        cnn.ConvModule = ConvModule
        mmcv.cnn = cnn

        def _jit(*a, **kw):
            # Usable both bare (@mmcv.jit) and called (@mmcv.jit(coderize=True)).
            if len(a) == 1 and not kw and callable(a[0]):
                return a[0]

            def deco(fn):
                return fn
            return deco

        mmcv.jit = _jit
        runner = types.ModuleType("mmcv.runner")

        def auto_fp16(*a, **kw):
            def deco(fn):
                return fn
            return deco
        runner.auto_fp16 = auto_fp16
        mmcv.runner = runner

        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.cnn"] = cnn
        sys.modules["mmcv.runner"] = runner

    if clrnet_root not in sys.path:
        sys.path.insert(0, clrnet_root)

    # clrnet.ops imports a compiled nms_impl at package import time.
    ops = types.ModuleType("clrnet.ops")
    ops.nms = lane_nms
    sys.modules["clrnet.ops"] = ops
    return ops
