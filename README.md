# Metric Lane-Width Estimation — Fixed Dashcam

---

## Ego-lane width + lead-vehicle width (`src/lane_width_run.py`)

This is the current path. It replaces the older scripts' *road* (curb-to-curb)
measurement with the **ego lane** — the two markings straddling the camera — and
adds the metric width of the vehicle ahead in that lane.

```powershell
python src\lane_width_run.py --detector yoloseg --part 5 --camera-height 1.635 --lane-pitch 3.20
python src\lane_width_run.py --detector clrnet  --part 5 --camera-height 1.635 --lane-pitch 3.20
```

To derive that height from the clip instead of passing it:

```powershell
python src\lane_width_run.py --detector yoloseg --part 5 --anchor-vehicle-width 1.80
```

Outputs per run, in `data/outputs/lane_width/`:
`clip5_<detector>.mp4`, `clip5_<detector>.csv`, `clip5_<detector>_summary.json`.

### How it measures

Pixels are projected onto the road plane and everything is measured there:

```
marking pixels -> vanishing point -> camera pitch/yaw -> road plane (X, Z metres)
```

The vanishing point is re-estimated every frame from the markings themselves, so
the mapping follows road pitch instead of assuming a fixed homography. Lane
boundaries are fitted as `X = f(Z)`, fragments of the same dashed line are merged
into one boundary, and the ego lane is the nearest boundary on each side of
`X = 0`. That rule also handles a double centre line correctly, since the inner
stripe is simply the nearer one.

Lead-vehicle width comes from the same mapping: the bottom edge of its box is
where the tyres meet the road, so both of its corners project to true metric
positions and their separation is the vehicle's width.

### Missing lane markings

Roads are expected to carry markings at a regular lane spacing, so the detected
boundaries are fitted to a grid of expected positions (the grid's phase is
tracked across frames as the car drifts in its lane). Any expected position with
no detection near it is reported as a missing marking.

This also fixes a failure that plain nearest-boundary selection cannot: when one
ego edge is absent, the next line out gets promoted and the lane reads one whole
lane too wide. Against the grid, an absent marking is reported absent instead.

A single frame where the detector did not fire is not a missing marking, so a
flag is only raised after `--missing-persist` consecutive frames (default 5).
That roughly halves false flags. Where a marking is missing, the video draws a
red dashed line at the position it was expected, so a reviewer can judge whether
the paint is worn away or the road never had a marking there — the human step in
the workflow. CSV columns: `marking_missing` (`left` / `right` / `both`),
`ego_left_present`, `ego_right_present`, `slots_total`, `slots_missing`.

Expect some flags to be detector misses rather than absent paint; that is what
the review step is for.

### Calibration: one number, not two

Focal length is **not** recoverable from this footage — parallel lane lines plus
their vanishing point leave it unconstrained, and the lane-width-vs-depth spread
is identical for f = 600 and f = 3000. Working the algebra through, f cancels out
of the lateral mapping entirely. **Camera height is the only constant that sets
scale, and every reported width is exactly proportional to it.**

Set it three ways:

- `--camera-height 1.64` — measure the mount height once with a tape. Preferred.
- `--anchor-vehicle-width 1.80` — solves the height from the width of detected
  cars. Car bodies are a scale reference independent of the road, so **lane
  width stays a real measurement**. Use this when lane width is the question.
- `--anchor-lane-width 3.5` — solves the height from a known lane width. Lane
  width then sits on that value by construction, so only vehicle width remains
  an independent measurement.

On clip 5 the two anchors agree to within 8%: 1.635 m from 37 cars (at 1.80 m
per car) against 1.771 m from the lane anchor (at 3.5 m per lane). Read the
other way, if cars are 1.80 m then lanes here are ~3.2 m; if lanes are 3.5 m
then cars are ~1.94 m. Both runs below therefore use the vehicle-derived
height, which keeps lane width independent.

The summary reports the implied camera height; it is worth sanity-checking
against the actual mount. A ~1.6 m camera is consistent with the high viewpoint
in this footage, which looks down on adjacent sedans.

### Detectors

| `--detector` | model | notes |
|---|---|---|
| `laneseg` | ACCESSLab real-time lane segmentation (TorchScript) | **best on this footage.** Lightweight binary segmenter; gets the double-yellow and the full-length dashed divider, near and far. |
| `yoloseg` | `models/checkpoints/yolov8_lane_seg.pt` | instance segmentation of lane markings (solid/dashed). Run at `imgsz=640`: at 1280 it fires only on thin far-field paint and misses the wide near-field markings. |
| `clrnet` | official CLRNet ResNet-18 / CULane | row-anchor model predicting lane instances directly. Weakest here — see below. |
| `yolopv2` | `models/checkpoints/yolopv2.pt` | dense binary lane head; instances via connected components. |

`laneseg` is [ACCESSLab/Lane-Detection-using-Segmentation](https://github.com/ACCESSLab/Lane-Detection-using-Segmentation)
(Getahun & Karimoddini, *IEEE T-IV* 2026), cloned into `third_party/`. It ships
its own pretrained TorchScript weights, so nothing needs training or rebuilding;
the network itself runs in ~11 ms. Connected components are labelled at the
model's native 640x384 rather than after upscaling — doing it at full resolution
scans two million pixels per component and cost ~3 fps against ~25 for
everything else, for points that get subsampled anyway.

**On this footage `yoloseg` is the one to use.** CLRNet is CULane-trained and
locks onto the strong continuous lines — the double-yellow centre line and the
solid edge lines — without reliably emitting the ego lane's *dashed* right
divider as its own instance. With that divider missing the nearest right-hand
boundary becomes the next line out, so CLRNet's raw width comes out about twice
yoloseg's, consistently, at every threshold tried (conf 0.08–0.4, max_lanes
4–16, NMS 15–50). Both detectors are therefore run with the **same** camera
height and `--lane-pitch`: anchoring CLRNet separately would make a two-lane
span read as 3.5 m and silently halve every vehicle width, and letting it
bootstrap its own pitch would lock the expected-marking grid to a two-lane
spacing. Pinned to the true pitch, its weakness surfaces honestly — as missing
markings rather than as a plausible-looking wrong width.

CLRNet runs through `src/clrnet_shim.py`, which supplies the two things its
training stack needs but this environment cannot build: `mmcv.cnn.ConvModule`
(reproduced with mmcv's submodule names so checkpoint keys still match) and
Lane-NMS (ported from the CUDA kernel to Python). All pretrained weights load
with zero unexpected keys. Because the model is CULane-trained, it is
under-confident here: its native `conf=0.4` returns nothing, so the default is
`0.15` with `max_lanes=8`. It is also fed an aspect-matched horizon-to-bottom
band rather than a squashed full frame.

### Reading the results

`lane_width_m` / `lead_width_m` are raw per-frame; the `_smooth` columns are
median-gated and are what the video shows. Frames with no reported lane are
usually correct refusals — junctions and crossings in this clip genuinely have no
ego-lane markings.

---

## Running a full-length video

`--part` selects a 5-minute slice by default (`--clip-seconds 300`). Pass
`--part 0` to process the whole file, and `--input` to point at a video other
than `data/videos/dashcam.mp4`:

```powershell
python src\lane_width_run.py --detector laneseg --part 0 `
  --input "data\videos\my_drive.mp4" `
  --camera-height 1.635 --lane-pitch 3.20 --crf 24
```

Outputs land in `data/outputs/lane_width/` as `clip0_laneseg.mp4`,
`clip0_laneseg.csv` and `clip0_laneseg_summary.json`.

Budget for the output size before starting. At `--crf 24` a 1080p dashcam clip
with the overlay runs roughly **45 MB per 1000 frames**, so a 200k-frame drive
produces an ~9 GB video. Compression varies a lot with scene detail — a
featureless rural stretch costs a fraction of a busy market street — so treat
that figure as an upper estimate and check free space rather than assuming.

A reference run: 202,899 frames of 1920x1080 at 30 fps (a 11.9 GB source,
1 h 52 m of driving) produced an 8.77 GB annotated video and a 13.5 MB CSV in
about two hours on an RTX 4050 Laptop + Ryzen 7 7840HS.

### Processing a video the moment it finishes downloading

Partial downloads are not valid videos, so wait for the browser's temporary file
to disappear *and* for the size to settle before starting:

```bash
V=data/videos
while :; do
  if ! ls "$V"/*.crdownload >/dev/null 2>&1; then
    NEW=$(find "$V" -maxdepth 1 -type f -name '*.mp4' -newer "$V/dashcam.mp4" | head -1)
    if [ -n "$NEW" ]; then
      s1=$(stat -c %s "$NEW"); sleep 5; s2=$(stat -c %s "$NEW")
      [ "$s1" = "$s2" ] && break
    fi
  fi
  sleep 20
done
python src/lane_width_run.py --detector laneseg --part 0 --input "$NEW" \
  --camera-height 1.635 --lane-pitch 3.20 --crf 24
```

---

## Throughput

### The pipeline runs decode, inference and encode concurrently

The measurement loop has four stages per frame: decode, two model passes, the
overlay drawing, and the H.264 encode. Run in a single loop they take turns —
the GPU sits idle while x264 encodes, and x264 sits idle while the GPU infers.
The symptom is distinctive and easy to misread: **neither the CPU nor the GPU
appears busy**, because both numbers are duty cycles rather than ceilings.

Decode and encode therefore run on their own threads, joined to the main loop by
bounded queues (`src/lane_core.py`):

- `FrameReader` decodes ahead of the consumer. It takes ownership of the
  `cv2.VideoCapture` — seek before constructing one, and do not read from the
  capture directly afterwards.
- `VideoWriter` feeds the ffmpeg pipe from a pump thread, so `write()` returns
  immediately and x264 encodes while the next frame is being measured. A frame
  handed to `write()` must not be mutated afterwards, since it is encoded later.

Both queues are bounded (default depth 12, about 75 MB of 1080p frames each), so
a slow stage applies backpressure instead of growing without limit. ffmpeg
failures are captured on the pump thread and re-raised on the next `write()` or
on `release()`, so a broken pipe still surfaces rather than being swallowed.

Measured on the reference run above, at identical output — every frame, same
`--crf`, same CSV:

| | throughput | wall clock |
|---|---|---|
| single-threaded loop | 14.9 fps | 223 min |
| threaded | 28.4–30.6 fps | 110 min |

### Remaining headroom

GPU utilisation sits near 45% even when saturated, because the two models run
sequentially at batch 1. In rough order of return per unit of effort:

1. **Encode on NVENC** rather than libx264. x264 is the largest single CPU
   consumer, and the GPU's encoder block is completely idle. Replace the codec
   flags in `lane_core.VideoWriter` with
   `-c:v h264_nvenc -preset p4 -tune hq -cq 24`.
2. **Decode on NVDEC.** `cv2.VideoCapture` decodes on the CPU; the GPU's decode
   block is also idle.
3. **`half=True` on the vehicle model.** `LaneSegDetector` already runs fp16
   (`lane_detectors.py:105`), but the ultralytics `predict()` call at
   `lane_detectors.py:48` does not, so half the per-frame GPU work is fp32.
4. **Batch 4–8 frames through both models.** Detection is stateless per frame,
   so a batch can be inferred at once and the geometry and temporal smoothing
   then run sequentially over the results, leaving the measurement unchanged.
   This is the real fix for the 45% ceiling, and a genuine refactor of the loop.
5. **TensorRT export**, for another 2–3x on model time at the cost of a brittle
   export step.

After 1–4 the bottleneck becomes the per-frame numpy geometry and `cv2` drawing,
which is single-threaded Python and considerably harder to move.

### Reading GPU utilisation on Windows

Task Manager's GPU panes default to 3D, Copy, Video Encode and Video Decode.
**None of them show CUDA compute**, so a saturated inference job reads as 0%.
Switch one pane's dropdown to *Cuda*, or use `nvidia-smi`:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

---

## Recovering a killed run

ffmpeg writes the mp4 index (`moov`) only when it exits cleanly. A run killed
part-way — by the OOM killer, a crash, or Ctrl-C — leaves a file that no player
will open, reporting `moov atom not found`, even though every encoded frame is
sitting in it. Diagnose with:

```powershell
ffprobe -v error -show_format "data\outputs\lane_width\clip0_laneseg.mp4"
```

The frames are recoverable without re-encoding. The surviving `mdat` holds the
samples as 4-byte length-prefixed AVCC NAL units; only the index and the SPS/PPS
(which live in `avcC`, inside the lost `moov`) are gone. The script regenerates
the SPS/PPS with a one-second throwaway encode at the same x264 settings,
converts the payload to Annex-B, and remuxes with `-c copy`:

```powershell
python scripts\recover_truncated_mp4.py `
  data\outputs\lane_width\clip0_laneseg.mp4 `
  data\outputs\lane_width\clip0_laneseg_fixed.mp4
```

Nothing is re-encoded, so the result is bit-identical to what the killed run
produced, minus the final sample if it was cut mid-write. Pass
`--crf/--preset/--fps/--width/--height` if the original did not use the
`VideoWriter` defaults, since the regenerated SPS must match the original
encode. The script refuses to overwrite an existing output, and leaves the
broken file untouched — keep it until the recovered one plays.

**The CSV will be shorter than the video.** Python buffers CSV writes, and a
killed process never flushes them, whereas the video frames had already been
handed to ffmpeg. Expect the video to run a few seconds past the last CSV row.
`clip0_laneseg_summary.json` is written after the loop, so a killed run produces
none at all; the summary statistics can be recomputed from the CSV without
re-running inference.

If the kill was the operating system reclaiming memory, check headroom before
restarting — the reader and writer queues plus both models need a couple of GB
on top of whatever else is resident.

---

## Older pipeline (road width, pixel-only)

End-to-end baseline for a **fixed 1920x1080 dashcam**. The pipeline is designed for your footage and an RTX 4050.

## Architecture

`video -> YOLOP road/lane perception + Video-Depth-Anything Small Metric -> temporal filtering -> ground-plane geometry -> fixed calibration -> lane width at a chosen distance -> annotated video + CSV`

The project deliberately does **not** regress metres directly with a neural network. AI models provide perception/depth; calibration and geometry perform the metric measurement.

### Models
- **YOLOP**: drivable-area + lane-line segmentation. Loaded through PyTorch Hub.
- **Metric Video-Depth-Anything Small (VDA-S-Metric)**: 28.4M parameter metric video-depth model.

Official VDA docs: https://github.com/DepthAnything/Video-Depth-Anything
Official YOLOP: https://github.com/hustvl/YOLOP

## Important: calibration is required for trustworthy metres

The repository cannot contain your real camera calibration because it depends on your physical mounting and known road measurements. Before trusting output in metres, run the calibration workflow. A placeholder config is included.

The recommended calibration uses at least 4 road-plane image points with known real-world `(X,Z)` coordinates in metres. The points must be on the same road plane and must not be collinear. If you have lane markings with a known measured separation, use them.

## Windows setup

### 1. Put the video here

`data/videos/dashcam.mp4`

### 2. YOLOP environment

You already have a working YOLOP environment. Activate it and ensure `prefetch_generator` is installed. If needed:

```powershell
pip install prefetch_generator opencv-python pyyaml pandas tqdm scipy
```

### 3. Create the separate VDA environment

The official VDA requirements currently pin PyTorch 2.1.1/torchvision 0.16.1 and xformers 0.0.23, so do **not** overwrite your existing YOLOP/PyTorch environment. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_vda.ps1
```

The script clones the official VDA repository and downloads the 116 MB Small Metric checkpoint automatically.

### 4. Test on the existing 200 frames

From the project root:

```powershell
python src\video_info.py --video data\videos\dashcam.mp4
python src\extract_frames.py --video data\videos\dashcam.mp4 --output data\test_frames --count 200
```

Then activate your YOLOP environment and run:

```powershell
python src\01_yolop_frames.py --input data\test_frames --output data\outputs\yolop
```

Then activate the VDA environment and run:

```powershell
.\scripts\run_vda_test.ps1
```

This generates depth visualisations/NPZ for the 200-frame test sequence.

## Calibration

### Camera intrinsics

If you do not have a checkerboard calibration, start with the approximate intrinsics in `calibration/camera.json` and replace them later with a real calibration. The principal point defaults to the image centre.

### Metric road homography

For the most reliable fixed-camera metric measurement, use `src/03_calibrate_homography.py`. It lets you click 4+ road points in one frame and enter their real-world coordinates in metres.

Example coordinate convention:
- `X`: right is positive, left is negative
- `Z`: forward is positive

For example, if a measured lane line crosses the road at known positions, use those measured positions. **Do not use guessed values and call the result ground truth.**

## Full pipeline

After calibration, the recommended production run is at 5 FPS first:

```powershell
.\scripts\run_full.ps1 -Fps 5 -MeasurementDistance 15
```

The script runs YOLOP in your YOLOP Python and VDA streaming inference in the dedicated VDA Python. Depth is consumed frame-by-frame, so the 23.6-minute video is not loaded into RAM and giant depth NPZ files are not required. You can pass explicit interpreters:

```powershell
.\scripts\run_full.ps1 -YoloPython "C:\path\to\yolop\python.exe" -VdaPython ".\.venv-vda\Scripts\python.exe" -Fps 5
```

Outputs:
- `data/outputs/final/lane_width.csv`
- `data/outputs/final/lane_width_annotated.mp4`
- `data/outputs/final/summary.json`
- intermediate YOLOP masks/overlays
- depth output

## What counts as a valid result?

A lane width is reported only when there are two sufficiently strong boundary candidates with a plausible road/ground geometry and temporal consistency. On an unmarked road, the system may correctly report `NO_RELIABLE_LANE_MARKINGS` rather than invent a lane width.

## Notes

- Start with 200 frames and inspect overlays before processing the 23.6-minute video.
- Use 2–5 FPS for development. Full 30 FPS inference is unnecessary initially.
- The 15 m measurement distance is configurable.
- If you later label 300–800 local frames, the perception branch can be fine-tuned without changing the metric geometry layer.
