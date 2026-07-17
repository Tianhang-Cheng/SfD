#!/bin/bash
# Evaluate every DuplicateSingleImage sample (rgb/albedo/normal/roughness metrics
# vs the held-out test frame), sharding the samples round-robin across 4 GPUs.
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

set -uo pipefail

SFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$SFD_DIR")/DuplicateSingleImage/train_split}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$(dirname "$SFD_DIR")/DuplicateSingleImage/eval_split}"
LOG_DIR="${LOG_DIR:-$SFD_DIR/train_logs}"
NUM_GPUS="${NUM_GPUS:-4}"

SAMPLES=(airplane box cake cash cheese cleaner clock coffee cola fire gitar potato sign tin yogurt)

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

eval_sample() {
  local name=$1
  local log="$LOG_DIR/${name}_eval.log"

  merge_eval_gt "$name"

  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: eval" | tee -a "$log"
  python exp_runner.py --conf configs/default.yaml \
    --data_split_dir "$DATA_ROOT/$name" --expname "$name" \
    --trainstage Mat --init_method SFM --is_continue --eval >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: eval FAILED" | tee -a "$log"
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
