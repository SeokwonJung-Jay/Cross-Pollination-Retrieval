#!/usr/bin/env bash
# Held-out domain training: {Qwen3-Embedding-0.6B, 4B} x 4 folds (training set without Life / Social / Physical / Health), 8 runs,
# one after another on all GPUs. Recipe = the dataset-ablation runs (training/qwen/train.sh defaults, official SWIFT.md): batch 128
# (8 GPUs x 4 x grad-acc 4), lr 6e-6 cosine, InfoNCE 0.1, all 9 hard negatives + in-batch, max length 1024 (data pre-cut), 1 epoch
# (5,896 pairs -> 46 steps), final checkpoint, ZeRO-2, gradient checkpointing, flash attention, seed 42 / data seed 42.
#
#   DATA=/path/folds_ml1024 OUT=/path/runs bash run.sh [qwen3_0.6b qwen3_4b]      FOLDS="life social" restricts the folds
#
# DATA: <DATA>/fold_<life|social|physical|health>/train.jsonl from build_folds.py run on the 1024-token copy of the training set (README).
# A run with <OUT>/<base>/<fold>/DONE is skipped. The final checkpoint is finalized for sentence-transformers (query prompt 'Query:').
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); TR=$(cd "$HERE/../../training" && pwd)
DATA=${DATA:?DATA (fold data root) is required}; OUT=${OUT:?OUT (run output root) is required}
QWEN_VENV=${QWEN_VENV:-$TR/.venv-qwen}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

for b in ${*:-qwen3_0.6b qwen3_4b}; do for f in ${FOLDS:-life social physical health}; do
  out=$OUT/$b/$f; [ -f "$out/DONE" ] && { echo "skip $b/$f"; continue; }
  mkdir -p "$out"; echo "== $b / without $f"
  case $b in
    qwen3_0.6b) model=Qwen/Qwen3-Embedding-0.6B; gc=(--gradient_checkpointing true) ;;
    qwen3_4b)   model=Qwen/Qwen3-Embedding-4B; gc=() ;;   # ms-swift default: gradient checkpointing on
    *) echo "unknown base $b"; exit 1 ;;
  esac
  ( . "$QWEN_VENV/bin/activate"
    MODEL=$model DEEPSPEED=zero2 SAVE_STEPS=50 MIN_NEG=9 TRAIN_DATA=$DATA/fold_$f/train.jsonl OUTPUT_DIR=$out \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_NAME=$b-no$f \
    bash "$TR/qwen/train.sh" --split_dataset_ratio 0 --eval_strategy no --attn_impl flash_attn --seed 42 --data_seed 42 "${gc[@]}"
    last=$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)
    CUDA_VISIBLE_DEVICES= python "$TR/qwen/finalize.py" "$last" ) > "$out/train.log" 2>&1
  touch "$out/DONE"
done; done
