import argparse
import os
import glob
import json

import cv2
import numpy as np
import torch


def load_model():
    try:
        model = torch.hub.load(
            "hustvl/yolop",
            "yolop",
            pretrained=True,
            trust_repo=True
        )
        return model
    except Exception as e:
        raise RuntimeError(
            "YOLOP could not be loaded. "
            "Run this inside the environment where YOLOP already works."
        ) from e


def letterbox(img, new_shape=(640, 640), color=114):
    """
    YOLOP-compatible letterbox preprocessing.
    Keeps aspect ratio and pads to 640x640.
    """
    h, w = img.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)

    new_w = int(round(w * r))
    new_h = int(round(h * r))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (new_shape[0], new_shape[1], 3),
        color,
        dtype=np.uint8
    )

    dw = (new_shape[1] - new_w) // 2
    dh = (new_shape[0] - new_h) // 2

    canvas[
        dh:dh + new_h,
        dw:dw + new_w
    ] = resized

    return canvas, r, dw, dh, new_w, new_h


def preprocess(img):
    """
    BGR OpenCV image -> normalized RGB tensor.
    """
    canvas, r, dw, dh, new_w, new_h = letterbox(
        img,
        (640, 640)
    )

    # BGR -> RGB
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # [0,255] -> [0,1]
    x = x.astype(np.float32) / 255.0

    # ImageNet normalization used by YOLOP
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    x = (x - mean) / std

    # HWC -> CHW
    x = x.transpose(2, 0, 1)

    x = np.ascontiguousarray(x)

    tensor = torch.from_numpy(x).unsqueeze(0)

    return tensor, r, dw, dh, new_w, new_h


def crop_segmentation(
    seg,
    dh,
    dw,
    new_h,
    new_w
):
    """
    Remove letterbox padding from segmentation output.
    """

    seg = seg[
        ...,
        dh:dh + new_h,
        dw:dw + new_w
    ]

    return seg


def decode_segmentation(seg):
    """
    YOLOP segmentation output has two channels.

    Channel 0 = background
    Channel 1 = foreground

    Official YOLOP decoding uses argmax.
    """

    if seg.shape[1] < 2:
        raise RuntimeError(
            f"Unexpected segmentation shape: {seg.shape}"
        )

    mask = torch.argmax(
        seg,
        dim=1
    )[0]

    return mask


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/test_frames"
    )

    parser.add_argument(
        "--output",
        default="data/outputs/yolop"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    lane_dir = os.path.join(
        args.output,
        "lane"
    )

    drive_dir = os.path.join(
        args.output,
        "drive"
    )

    overlay_dir = os.path.join(
        args.output,
        "overlay"
    )

    os.makedirs(lane_dir, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("YOLOP device:", device)

    model = load_model()

    model = model.to(device)
    model.eval()

    paths = sorted(
        glob.glob(
            os.path.join(args.input, "*.jpg")
        )
        +
        glob.glob(
            os.path.join(args.input, "*.png")
        )
    )

    if len(paths) == 0:
        raise RuntimeError(
            f"No images found in {args.input}"
        )

    metadata = []

    with torch.no_grad():

        for index, path in enumerate(paths):

            img = cv2.imread(path)

            if img is None:
                print(
                    f"WARNING: could not read {path}"
                )
                continue

            original_h, original_w = img.shape[:2]

            (
                x,
                r,
                dw,
                dh,
                new_w,
                new_h
            ) = preprocess(img)

            x = x.to(device)

            output = model(x)

            if not isinstance(output, (tuple, list)):
                raise RuntimeError(
                    f"Unexpected YOLOP output type: "
                    f"{type(output)}"
                )

            if len(output) < 3:
                raise RuntimeError(
                    "YOLOP returned fewer than 3 outputs."
                )

            det_out = output[0]
            da_seg_out = output[1]
            ll_seg_out = output[2]

            # Remove letterbox padding
            da_seg_out = crop_segmentation(
                da_seg_out,
                dh,
                dw,
                new_h,
                new_w
            )

            ll_seg_out = crop_segmentation(
                ll_seg_out,
                dh,
                dw,
                new_h,
                new_w
            )

            # Official YOLOP-style argmax decoding
            drive_mask = decode_segmentation(
                da_seg_out
            )

            lane_mask = decode_segmentation(
                ll_seg_out
            )

            drive_mask = (
                drive_mask
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            lane_mask = (
                lane_mask
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Resize masks back to original image size
            drive_mask = cv2.resize(
                drive_mask,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

            lane_mask = cv2.resize(
                lane_mask,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Convert {0,1} -> {0,255}
            drive_png = drive_mask * 255
            lane_png = lane_mask * 255

            stem = os.path.splitext(
                os.path.basename(path)
            )[0]

            cv2.imwrite(
                os.path.join(
                    drive_dir,
                    stem + ".png"
                ),
                drive_png
            )

            cv2.imwrite(
                os.path.join(
                    lane_dir,
                    stem + ".png"
                ),
                lane_png
            )

            # -----------------------------
            # Visualization
            # -----------------------------

            overlay = img.copy()

            # Green = drivable area
            green = np.zeros_like(img)
            green[:, :, 1] = drive_png

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                green,
                0.22,
                0
            )

            # Red = lane lines
            red = np.zeros_like(img)
            red[:, :, 2] = lane_png

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                red,
                0.75,
                0
            )

            cv2.putText(
                overlay,
                "YOLOP: green=drivable red=lane",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imwrite(
                os.path.join(
                    overlay_dir,
                    stem + ".jpg"
                ),
                overlay
            )

            metadata.append(
                {
                    "file": os.path.basename(path),
                    "stem": stem,
                    "width": original_w,
                    "height": original_h
                }
            )

            if (index + 1) % 20 == 0:
                print(
                    f"{index + 1}/{len(paths)}"
                )

    with open(
        os.path.join(
            args.output,
            "frames.json"
        ),
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("Done.")


if __name__ == "__main__":
    main()