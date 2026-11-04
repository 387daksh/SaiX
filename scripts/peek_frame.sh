#!/usr/bin/env bash
# Pull one annotated frame out of a run that is still encoding.
#
# The live mp4 has no moov atom until ffmpeg exits, so no player or decoder
# will open it. This copies enough of the head to contain the wanted frame,
# rebuilds a playable slice with recover_truncated_mp4.py, and extracts it.
#
#   scripts/peek_frame.sh 11403 [out.png]
set -euo pipefail

N=${1:?frame number}
OUT=${2:-frame_$N.png}
SRC=${SRC:-data/outputs/lane_width/clip0_laneseg.mp4}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ~56 KB per frame at crf 24, doubled for headroom plus a 200 MB floor.
BYTES=$(( N * 56000 * 2 + 200000000 ))
head -c "$BYTES" "$SRC" > "$TMP/head.mp4"
python scripts/recover_truncated_mp4.py "$TMP/head.mp4" "$TMP/fixed.mp4" --crf 24

rm -f "$OUT"
ffmpeg -v error -i "$TMP/fixed.mp4" -vf "select=eq(n\,$N)" \
  -fps_mode passthrough -frames:v 1 -y "$OUT"
# select matches nothing and still exits 0 when that frame is not encoded yet.
[ -s "$OUT" ] || { echo "frame $N is not encoded yet" >&2; exit 1; }
echo "$OUT"
