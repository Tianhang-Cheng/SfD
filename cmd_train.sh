#!/bin/bash
# Train every DuplicateSingleImage sample through all 3 stages (Geo -> Vis -> Mat),
# sharding the samples round-robin across 4 GPUs. A failed stage/sample is logged
# and skipped (does not abort the script or other samples/GPUs).
#
# Once training finishes, run cmd_eval.sh to evaluate every sample.
#
# Usage (inside tmux, so it survives disconnects):
#   tmux new -s train
#   bash /path/to/SfD/cmd_train.sh
#   # detach with Ctrl-b d, reattach later with: tmux attach -t train
#
# Paths default relative to this script's location (see below) but can be overridden, e.g.:
#   DATA_ROOT=/path/to/DuplicateSingleImage/train_split bash cmd_train.sh
#
# Rough total training time per GPU: ~4 samples * (~10h Geo + ~0.5h Vis + ~1h Mat) ≈ 46h.

set -uo pipefail

SFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$SFD_DIR")/DuplicateSingleImage/train_split}"
LOG_DIR="${LOG_DIR:-$SFD_DIR/train_logs}"
NUM_GPUS="${NUM_GPUS:-4}"

SAMPLES=(airplane box cake cash cheese cleaner clock coffee cola fire gitar potato sign tin yogurt)

mkdir -p "$LOG_DIR"
cd "$SFD_DIR"

train_sample() {
  local name=$1
  local log="$LOG_DIR/${name}.log"

  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: stage 1/3 Geo" | tee -a "$log"
  python exp_runner.py --conf configs/default.yaml \
    --data_split_dir "$DATA_ROOT/$name" --expname "$name" \
    --trainstage Geo --use_pretrain_normal --init_method SFM >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: Geo FAILED, skipping Vis/Mat" | tee -a "$log"
    return 1
  fi

  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: stage 2/3 Vis" | tee -a "$log"
  python exp_runner.py --conf configs/default.yaml \
    --data_split_dir "$DATA_ROOT/$name" --expname "$name" \
    --trainstage Vis --init_method SFM >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: Vis FAILED, skipping Mat" | tee -a "$log"
    return 1
  fi

  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: stage 3/3 Mat" | tee -a "$log"
  python exp_runner.py --conf configs/default.yaml \
    --data_split_dir "$DATA_ROOT/$name" --expname "$name" \
    --trainstage Mat --init_method SFM >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: Mat FAILED" | tee -a "$log"
    return 1
  fi

  echo "[$(date '+%F %T')] [GPU $CUDA_VISIBLE_DEVICES] ${name}: DONE (all 3 stages)" | tee -a "$log"
}

gpu_worker() {
  local gpu_id=$1
  shift
  export CUDA_VISIBLE_DEVICES=$gpu_id
  for name in "$@"; do
    train_sample "$name"
  done
}

# Shard samples round-robin across GPUs 0..NUM_GPUS-1
declare -a SHARDS
for ((i = 0; i < NUM_GPUS; i++)); do SHARDS[$i]=""; done
for ((i = 0; i < ${#SAMPLES[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  SHARDS[$gpu]="${SHARDS[$gpu]} ${SAMPLES[$i]}"
done

pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  gpu_worker "$gpu" ${SHARDS[$gpu]} &
  pids+=($!)
  echo "Launched GPU $gpu worker (pid ${pids[-1]}) for samples:${SHARDS[$gpu]}"
done

wait "${pids[@]}"
echo "All samples finished training. Per-sample logs are in $LOG_DIR"
echo "Run cmd_eval.sh to evaluate all samples."
