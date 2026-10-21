"""
Recover an mp4 whose `moov` atom was never written.

ffmpeg writes the index only when it exits cleanly, so a run that is killed
leaves a file that no player will open ("moov atom not found") even though every
encoded frame is present. The surviving `mdat` holds the samples as 4-byte
length-prefixed AVCC NAL units; only the index and the SPS/PPS (which live in
`avcC`, inside the lost `moov`) are gone.

This regenerates the SPS/PPS with a throwaway encode using the same x264
settings, converts the payload to Annex-B, and remuxes with `-c copy`. Nothing
is re-encoded, so the output is bit-identical to what the killed run produced,
minus the final sample if it was cut mid-write.

  python scripts/recover_truncated_mp4.py broken.mp4 fixed.mp4

Use --crf/--preset/--fps to match the original encode if it did not use the
VideoWriter defaults in src/lane_core.py.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile


def find_mdat(f):
    """Return the offset of the mdat payload."""
    off = 0
    while True:
        f.seek(off)
        h = f.read(8)
        if len(h) < 8:
            raise SystemExit("no mdat atom found; this is not a truncated mp4")
        size, typ = struct.unpack(">I4s", h)
        body = off + 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            body = off + 16
        if typ == b"mdat":
            return body
        if size == 0:
            raise SystemExit(f"atom {typ.decode('latin1')} has size 0 before mdat")
        off += size


def reference_sps_pps(w, h, fps, crf, preset, workdir):
    """Encode one second of black at the same settings and take its SPS/PPS."""
    ref = os.path.join(workdir, "ref.mp4")
    raw = os.path.join(workdir, "ref.h264")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=black:s={w}x{h}:r={fps}:d=1",
         "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
         "-pix_fmt", "yuv420p", ref],
        check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", ref,
         "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "h264", raw],
        check=True)

    d = open(raw, "rb").read(200_000)
    starts, i = [], 0
    while True:
        j = d.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j)
        i = j + 3
    out = b""
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(d)
        nal = d[s + 3:e]
        if nal and (nal[0] & 0x1F) in (7, 8):
            out += b"\x00\x00\x00\x01" + nal.rstrip(b"\x00")
    if not out:
        raise SystemExit("could not extract SPS/PPS from the reference encode")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("broken")
    ap.add_argument("fixed")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--preset", default="veryfast")
    args = ap.parse_args()

    if os.path.exists(args.fixed):
        raise SystemExit(f"{args.fixed} already exists; refusing to overwrite")

    with tempfile.TemporaryDirectory() as tmp:
        hdr = reference_sps_pps(args.width, args.height, args.fps,
                                args.crf, args.preset, tmp)
        print(f"regenerated SPS/PPS ({len(hdr)} bytes)", file=sys.stderr)

        f = open(args.broken, "rb")
        body = find_mdat(f)
        print(f"mdat payload at offset {body}", file=sys.stderr)

        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts",
             "-framerate", str(args.fps), "-f", "h264", "-i", "-",
             "-c", "copy", args.fixed],
            stdin=subprocess.PIPE)
        out = ff.stdin
        out.write(hdr)

        f.seek(body)
        pos, n = body, 0
        while True:
            b = f.read(4)
            if len(b) < 4:
                break
            ln = struct.unpack(">I", b)[0]
            if ln < 1 or ln > 50_000_000:
                print(f"stop: implausible NAL length {ln} at {pos}",
                      file=sys.stderr)
                break
            payload = f.read(ln)
            if len(payload) < ln:
                # Killed mid-sample; drop the partial tail.
                print(f"dropped truncated final NAL at {pos} "
                      f"({len(payload)}/{ln} bytes)", file=sys.stderr)
                break
            out.write(b"\x00\x00\x00\x01")
            out.write(payload)
            pos += 4 + ln
            n += 1

        out.close()
        if ff.wait() != 0:
            raise SystemExit("ffmpeg remux failed")
        print(f"recovered {n} NAL units -> {args.fixed}", file=sys.stderr)


if __name__ == "__main__":
    main()
