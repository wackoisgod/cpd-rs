#!/usr/bin/env bash
# Usage: screenshot.sh <viewer.html> <out.png> [width] [height]
#
# Headlessly renders the side-by-side viewer into a PNG so a future-you (or
# Claude) can read the result without launching a GUI browser. The viewer
# loads three.js from a CDN so this needs network access on first run.

set -euo pipefail
HTML=${1:?missing viewer.html path}
PNG=${2:?missing output png path}
W=${3:-1920}
H=${4:-1080}

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Chrome not found at $CHROME" >&2
  exit 1
fi

abs_html=$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")

# Need WebGL on (so don't pass --disable-gpu in --headless=new mode), and
# --virtual-time-budget gives the page time to fetch three.js from the CDN
# and render the scene.
"$CHROME" \
  --headless=new \
  --hide-scrollbars \
  --enable-webgl \
  --use-angle=metal \
  --virtual-time-budget=8000 \
  --window-size="${W},${H}" \
  --screenshot="$PNG" \
  "file://$abs_html" 2>/tmp/cpd-shot.log >/dev/null

if [[ ! -s "$PNG" ]]; then
  echo "screenshot failed (empty file)" >&2
  exit 1
fi
echo "wrote $PNG ($(du -h "$PNG" | cut -f1))"
