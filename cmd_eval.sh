#!/bin/bash
# Evaluate every DuplicateSingleImage sample, sharding the samples round-robin
# across 4 GPUs. Two kinds of metrics are produced per sample:
#
#   2D (every sample): rgb/albedo/normal/roughness metrics vs the held-out test
#       frame, via exp_runner.py --eval.
#   3D (synthetic samples only, RUN_3D=1): Chamfer distance / F-score / normal
#       consistency of the exported mesh against the Blender ground truth, via
#       --to_mesh + scripts/blender_export_gt_mesh.py + scripts/eval_mesh_3d.py.
#
# Requires the Mat checkpoint of every sample to already exist (run cmd_train.sh
# first). A failed sample is logged and skipped (does not abort the script or
# other samples/GPUs).
#
# Usage (inside tmux, so it survives disconnects):
#   tmux new -s eval
#   bash /path/to/SfD/cmd_eval.sh
#   # detach with Ctrl-b d, reattach later with: tmux attach -t eval
#
# Paths default relative to this script's location (see below) but can be overridden, e.g.:
#   DATA_ROOT=/path/to/DuplicateSingleImage/train_split \
#   EVAL_DATA_ROOT=/path/to/DuplicateSingleImage/eval_split bash cmd_eval.sh
#
# 2D metrics only (skip the mesh export and the Blender ground truth):
#   RUN_3D=0 bash cmd_eval.sh

set -uo pipefail

SFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$SFD_DIR")/DuplicateSingleImage/train_split}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$(dirname "$SFD_DIR")/DuplicateSingleImage/eval_split}"
LOG_DIR="${LOG_DIR:-$SFD_DIR/train_logs}"
NUM_GPUS="${NUM_GPUS:-4}"

# --- 3D metrics -------------------------------------------------------------
RUN_3D="${RUN_3D:-1}"                                    # 0 disables the whole 3D block
BLENDER_DATA_ROOT="${BLENDER_DATA_ROOT:-$SFD_DIR/blender_data}"
BPY_PYTHON="${BPY_PYTHON:-python}"                       # a python whose "import bpy" works
BPY_LIB_DIR="${BPY_LIB_DIR:-}"                           # prepended to LD_LIBRARY_PATH for bpy
MESH_RES="${MESH_RES:-512}"                              # marching cubes resolution of --to_mesh
MESH_SAMPLES="${MESH_SAMPLES:-200000}"                   # surface samples per mesh

SAMPLES=(airplane box cake cash cheese cleaner clock coffee cola fire gitar potato sign tin yogurt)

# Samples with Blender ground truth. The six real-world captures (airplane cake
# cheese cola potato yogurt) have no .blend and no GT mesh, so they get 2D
# metrics only.
SYNTHETIC=(box cash cleaner clock coffee fire gitar sign tin)

mkdir -p "$LOG_DIR"
cd "$SFD_DIR"

# Merge ground-truth albedo/roughness (only shipped for synthetic samples) from the
# eval dataset into the training sample's train/ dir, where the code expects them.
merge_eval_gt() {
  local name=$1
  local src="$EVAL_DATA_ROOT/$name/train"
  local dst="$DATA_ROOT/$name/train"
  if [ -f "$src/000_diffuse.png" ] && [ ! -f "$dst/000_diffuse.png" ]; then
    cp "$src/000_diffuse.png" "$dst/000_diffuse.png"
  fi
  if [ -f "$src/000_roughness.png" ] && [ ! -f "$dst/000_roughness.png" ]; then
    cp "$src/000_roughness.png" "$dst/000_roughness.png"
  fi
}

is_synthetic() {
  local name=$1 candidate
  for candidate in "${SYNTHETIC[@]}"; do
    [ "$candidate" = "$name" ] && return 0
  done
  return 1
}

# Path of a sample's .blend. Most are <name>_clean.blend; cash ships scene.blend.
find_blend() {
  local name=$1 candidate
  for candidate in "$BLENDER_DATA_ROOT/$name/${name}_clean.blend" \
                   "$BLENDER_DATA_ROOT/$name/scene.blend"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  candidate=$(ls "$BLENDER_DATA_ROOT/$name"/*.blend 2>/dev/null | head -1)
  if [ -n "$candidate" ]; then
    echo "$candidate"
    return 0
  fi
  return 1
}

# Run a script under the python that has bpy, with the .so directory it needs.
run_bpy() {
  LD_LIBRARY_PATH="${BPY_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$BPY_PYTHON" "$@"
}

# Check once, before any worker starts, that the 3D path can run at all: no
# point exporting nine meshes if the Blender ground truth cannot be produced.
# Turns RUN_3D off (2D metrics still run) instead of failing every sample.
check_3d_prerequisites() {
  [ "$RUN_3D" = "1" ] || return 0
  local reason=''
  if ! run_bpy -c 'import bpy' >/dev/null 2>&1; then
    reason="'import bpy' failed under $BPY_PYTHON. Point BPY_PYTHON at a python with the bpy
  module (pip install \"bpy==5.2.0\", needs CPython 3.13) and BPY_LIB_DIR at its lib/ directory
  if bpy cannot find libX11"
  elif [ ! -d "$BLENDER_DATA_ROOT" ]; then
    reason="no blender scenes under $BLENDER_DATA_ROOT (get them with:
  python download_assets.py --blender-data blender_data), or set BLENDER_DATA_ROOT"
  fi
  if [ -n "$reason" ]; then
    echo "WARNING: 3D metrics disabled -- $reason"
    echo "         Evaluating 2D metrics only. Pass RUN_3D=0 to silence this."
    RUN_3D=0
  fi
}

# Log line prefix, stamped when it is printed (steps below take minutes).
log_prefix() {
  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] $1"
}

# Chamfer / F-score / normal consistency of the reconstructed mesh against the
# Blender ground truth: export the prediction, export the ground truth, compare.
eval_3d_sample() {
  local name=$1
  local log="$LOG_DIR/${name}_eval.log"

  if ! is_synthetic "$name"; then
    echo "$(log_prefix "$name"): real-world sample, no Blender ground truth -> no 3D metrics" | tee -a "$log"
    return 0
  fi

  local blend
  blend=$(find_blend "$name")
  if [ -z "$blend" ]; then
    echo "$(log_prefix "$name"): no .blend under $BLENDER_DATA_ROOT/$name -> 3D metrics SKIPPED" | tee -a "$log"
    return 1
  fi

  # 1/3 the prediction: marching cubes on the canonical SDF of the latest Mat
  # checkpoint. This only loads a checkpoint, it never trains.
  echo "$(log_prefix "$name"): 3D 1/3 mesh export (res $MESH_RES)" | tee -a "$log"
  python exp_runner.py --conf configs/default.yaml \
    --data_split_dir "$DATA_ROOT/$name" --expname "$name" \
    --trainstage Mat --init_method SFM --to_mesh --mesh_res "$MESH_RES" >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "$(log_prefix "$name"): mesh export FAILED, no 3D metrics" | tee -a "$log"
    return 1
  fi

  # --to_mesh on Mat writes exps/Mat-<name>-mesh/<new timestamp>/mesh/.
  local mesh_dir
  mesh_dir=$(ls -td "$SFD_DIR/exps/Mat-$name-mesh"/*/mesh 2>/dev/null | head -1)
  if [ -z "$mesh_dir" ] || [ ! -f "$mesh_dir/mesh.ply" ]; then
    echo "$(log_prefix "$name"): no mesh.ply under exps/Mat-$name-mesh -> 3D metrics SKIPPED" | tee -a "$log"
    return 1
  fi

  # 2/3 the ground truth, straight out of the .blend (CPU only). Cached in the
  # sample folder, so re-running the eval does not redo it.
  local gt_dir="$DATA_ROOT/$name/gt"
  if [ -f "$gt_dir/gt_mesh_local.ply" ] && [ -f "$gt_dir/gt_mesh_world.ply" ]; then
    echo "$(log_prefix "$name"): 3D 2/3 reusing the ground truth mesh in $gt_dir" | tee -a "$log"
  else
    echo "$(log_prefix "$name"): 3D 2/3 exporting the ground truth mesh from $(basename "$blend")" | tee -a "$log"
    run_bpy scripts/blender_export_gt_mesh.py -- --blend_file "$blend" \
      --data_split_dir "$DATA_ROOT/$name" --output "$gt_dir" \
      --world --all_instances >> "$log" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(log_prefix "$name"): ground truth mesh export FAILED, no 3D metrics" | tee -a "$log"
      return 1
    fi
  fi

  # 3/3 the metrics, in both frames: blender_local compares one instance in its
  # own frame, blender_world the whole pile (the number to quote, with a ~0.2%
  # floor from the SfM-vs-Blender pose disagreement).
  local status=0 frame gt_mesh out
  for frame in blender_local blender_world; do
    if [ "$frame" = blender_local ]; then
      gt_mesh=gt_mesh_local.ply
      out=metrics_3d_local.json
    else
      gt_mesh=gt_mesh_world.ply
      out=metrics_3d_world.json
    fi
    echo "$(log_prefix "$name"): 3D 3/3 metrics ($frame)" | tee -a "$log"
    python scripts/eval_mesh_3d.py \
      --mesh "$mesh_dir/mesh.ply" --gt_mesh "$gt_dir/$gt_mesh" \
      --frame "$frame" --data_split_dir "$DATA_ROOT/$name" \
      --samples "$MESH_SAMPLES" --output "$mesh_dir/$out" >> "$log" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(log_prefix "$name"): 3D metrics ($frame) FAILED" | tee -a "$log"
      status=1
    fi
  done
  if [ $status -eq 0 ]; then
    echo "$(log_prefix "$name"): 3D metrics DONE -> $mesh_dir/metrics_3d_{local,world}.json" | tee -a "$log"
  fi
  return $status
}

eval_sample() {
  local name=$1
  local log="$LOG_DIR/${name}_eval.log"
  local status=0

  merge_eval_gt "$name"

  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: eval" | tee -a "$log"
  python exp_runner.py --conf configs/default.yaml \
    --data_split_dir "$DATA_ROOT/$name" --expname "$name" \
    --trainstage Mat --init_method SFM --is_continue --eval >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: eval FAILED" | tee -a "$log"
    status=1
  fi

  # 3D metrics are independent of the 2D eval above (they only need the Mat
  # checkpoint), so they run even if it failed.
  if [ "$RUN_3D" = "1" ]; then
    eval_3d_sample "$name" || status=1
  fi

  if [ $status -ne 0 ]; then
    return 1
  fi
  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: eval DONE" | tee -a "$log"
}

eval_gpu_worker() {
  local gpu_id=$1
  shift
  export CUDA_VISIBLE_DEVICES=$gpu_id
  for name in "$@"; do
    eval_sample "$name"
  done
}

check_3d_prerequisites

# Shard samples round-robin across GPUs 0..NUM_GPUS-1
declare -a SHARDS
for ((i = 0; i < NUM_GPUS; i++)); do SHARDS[$i]=""; done
for ((i = 0; i < ${#SAMPLES[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  SHARDS[$gpu]="${SHARDS[$gpu]} ${SAMPLES[$i]}"
done

eval_pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  eval_gpu_worker "$gpu" ${SHARDS[$gpu]} &
  eval_pids+=($!)
  echo "Launched GPU $gpu eval worker (pid ${eval_pids[-1]}) for samples:${SHARDS[$gpu]}"
done

wait "${eval_pids[@]}"
echo "All samples evaluated. Per-sample eval logs are in $LOG_DIR (*_eval.log)"
echo "2D metrics: exps/Mat-<name>-eval/<timestamp>/evals_value/ (+ evals_image/)"
if [ "$RUN_3D" = "1" ]; then
  echo "3D metrics: exps/Mat-<name>-mesh/<timestamp>/mesh/metrics_3d_{local,world}.json"
fi
