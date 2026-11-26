#!/usr/bin/env bash
# Builds gpu_fluid and runs the three experiments described in the README:
#   1. demo     1024x1024, 900 steps, frames -> output/demo/fluid.mp4 (+ gif)
#   2. scaling  grids 128..2048, naive vs tiled Jacobi, CPU comparison
#   3. jacobi   5..320 pressure sweeps per step: accuracy vs cost at 512x512
# then draws the plots and packs output/ into proof_of_execution.tar.gz.
#
# Env overrides: BIN (program path), FFMPEG, PYTHON, SKIP_BUILD=1
set -euo pipefail

cd "$(dirname "$0")"

BIN=${BIN:-bin/gpu_fluid}
PYTHON=${PYTHON:-python3}
FFMPEG=${FFMPEG:-ffmpeg}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  make
fi

rm -rf output
mkdir -p output/demo output/scaling output/jacobi

echo "=== 1/3 demo run ==="
"$BIN" --grid=1024 --steps=900 --frame_every=2 --log_every=50 \
  --cpu_compare_steps=5 --output_dir=output/demo 2>&1 | tee output/demo/run.log

echo "=== 2/3 scaling sweep ==="
for kernel in naive tiled; do
  for grid in 128 256 512 1024 2048; do
    "$BIN" --grid="$grid" --steps=300 --jacobi_kernel="$kernel" \
      --cpu_compare_steps=3 --log_every=100 \
      --output_dir="output/scaling/${kernel}_${grid}" \
      --summary_csv=output/scaling.csv 2>&1 \
      | tee "output/scaling/${kernel}_${grid}.log"
  done
done

echo "=== 3/3 jacobi sweep ==="
for iters in 5 10 20 40 80 160 320; do
  "$BIN" --grid=512 --steps=300 --jacobi_iters="$iters" --frame_every=299 \
    --log_every=100 --output_dir="output/jacobi/iters_${iters}" \
    --summary_csv=output/jacobi.csv 2>&1 \
    | tee "output/jacobi/iters_${iters}.log"
done

if command -v "$FFMPEG" >/dev/null 2>&1; then
  echo "=== encoding video ==="
  "$FFMPEG" -y -loglevel error -framerate 30 \
    -i output/demo/frames/frame_%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 20 output/demo/fluid.mp4
  "$FFMPEG" -y -loglevel error -framerate 30 \
    -i output/demo/frames/frame_%05d.png \
    -vf "fps=12,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    output/demo/fluid.gif
else
  echo "ffmpeg not found, skipping the video (frames are in output/demo/frames)"
fi

if "$PYTHON" -c "import matplotlib, PIL" 2>/dev/null; then
  "$PYTHON" scripts/plot_results.py output
else
  echo "matplotlib/Pillow not found, skipping plots"
fi

tar -czf proof_of_execution.tar.gz --exclude=output/demo/frames output
echo "packed output/ (without raw frames) into proof_of_execution.tar.gz"
