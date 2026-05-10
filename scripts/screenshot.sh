#!/usr/bin/env bash
# Usage:
#   screenshot.sh <viewer.html> <out.png> [width] [height]            # iso only
#   screenshot.sh <viewer.html> <out.png> [width] [height] --multi    # 6 angles + iso, montaged 2x4
#
# Headlessly renders the viewer into a PNG. With --multi, captures 7 camera
# angles (iso/front/back/left/right/top/bottom) via the viewer's
# `?angle=...` URL param and montages them with ImageMagick into a single
# image (2 rows × 4 cols, last cell blank). Useful when you want to spot
# bad primitives on the back/top of a mesh that the iso view hides.

set -euo pipefail
HTML=${1:?missing viewer.html path}
PNG=${2:?missing output png path}
W=${3:-1920}
H=${4:-1080}
MULTI=${5:-}

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Chrome not found at $CHROME" >&2
  exit 1
fi

abs_html=$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")

shoot() {
  local angle=$1
  local out=$2
  local url="file://$abs_html?ui=0"
  if [[ -n "$angle" ]]; then
    url="file://$abs_html?angle=$angle&ui=0"
  fi
  "$CHROME" \
    --headless=new \
    --hide-scrollbars \
    --enable-webgl \
    --use-angle=metal \
    --virtual-time-budget=8000 \
    --window-size="${W},${H}" \
    --screenshot="$out" \
    "$url" 2>/tmp/cpd-shot.log >/dev/null
}

if [[ "$MULTI" != "--multi" ]]; then
  shoot "" "$PNG"
  if [[ ! -s "$PNG" ]]; then
    echo "screenshot failed (empty file)" >&2
    exit 1
  fi
  echo "wrote $PNG ($(du -h "$PNG" | cut -f1))"
  exit 0
fi

# Multi-angle mode: take all 7, label each with the angle name, montage.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
ANGLES=(iso front back left right top bottom)
FONT="/System/Library/Fonts/Helvetica.ttc"
for a in "${ANGLES[@]}"; do
  shoot "$a" "$TMP/$a.png"
  magick "$TMP/$a.png" \
    -font "$FONT" \
    -gravity NorthEast \
    -fill white -undercolor "#0008" -pointsize 32 -annotate +20+20 " $a " \
    "$TMP/$a.png"
done

# Pad the 7 images to 8 cells (last cell black) and montage 4 across.
magick -size "${W}x${H}" canvas:black "$TMP/_blank.png"
magick montage \
  "$TMP/iso.png" "$TMP/front.png" "$TMP/back.png" "$TMP/_blank.png" \
  "$TMP/right.png" "$TMP/left.png" "$TMP/top.png" "$TMP/bottom.png" \
  -tile 4x2 -geometry "${W}x${H}+4+4" -background black "$PNG"

if [[ ! -s "$PNG" ]]; then
  echo "multi-angle screenshot failed" >&2
  exit 1
fi
echo "wrote $PNG (multi-angle, $(du -h "$PNG" | cut -f1))"
