#!/bin/bash
# One-shot preprocessing for a single object.
#
# Usage:
#   bash preprocess/run.sh data/your_object
#   bash preprocess/run.sh data/my_pile --instance_num 7 --crop_size 1000 --rotate_delta_angle 4
#   bash preprocess/run.sh data/my_pile --stages 5-7        # re-run SfM only
#
# The object directory must already contain raw/000_rgb.{exr,png} and raw/000_instance_seg.png;
# only data/airplane and data/your_object ship with this repo.
# The instance count and the input image (raw/000_rgb.exr, else raw/000_rgb.png) are
# detected automatically. Any extra argument is forwarded to preprocess/run.py, see
#   python preprocess/run.py --help

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash preprocess/run.sh <instance_dir> [extra args for preprocess/run.py]" >&2
  echo "   e.g. bash preprocess/run.sh data/your_object" >&2
  exit 1
fi

SFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_DIR="$1"
shift

cd "$SFD_DIR"
exec python preprocess/run.py --instance_dir "$INSTANCE_DIR" "$@"
