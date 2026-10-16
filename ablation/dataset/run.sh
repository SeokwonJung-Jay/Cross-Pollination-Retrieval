#!/usr/bin/env bash
# Dataset ablation training: base models x 4 training sets x 2 negative settings, one run after another on all GPUs (paper: 8). Bases:
# Qwen3-Embedding 0.6B and 4B (training/qwen/train.sh, official SWIFT.md recipe) and llama-embed-nemotron-8b (training/nemotron/train.sh,
# official NeMo AutoModel recipe); 24 runs by default. Every run: 1 epoch (83 steps), final checkpoint, max length 1024, flash attention, seed 42.
#
#   DATA=/path/data_ml1024 DATA_FULL=/path/data OUT=/path/runs bash run.sh [qwen3_0.6b] [qwen3_4b] [nemotron8b]
#
# DATA:      <DATA>/<hardneg|inbatch>/<ours|msmarco|reasonir|specter>/train.jsonl cut by make_maxlen.py (Qwen; see README.md)
# DATA_FULL: the same layout before make_maxlen.py (llama-embed-nemotron-8b: the NeMo collator truncates to 1024 tokens); needed for nemotron8b
# A run with <OUT>/<base>/<neg>/<set>/DONE is skipped. Qwen checkpoints are finalized for sentence-transformers (query prompt 'Query:');
# llama-embed-nemotron-8b checkpoints are exported to <OUT>/nemotron8b/<neg>/<set>/eval_model (training/nemotron/export.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); TR=$(cd "$HERE/../../training" && pwd)
OUT=${OUT:?OUT (run output root) is required}
QWEN_VENV=${QWEN_VENV:-$TR/.venv-qwen}; NEMO_VENV=${NEMO_VENV:-$TR/.venv-nemotron}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

for b in ${*:-qwen3_0.6b qwen3_4b nemotron8b}; do for n in hardneg inbatch; do for d in ours msmarco reasonir specter; do
  out=$OUT/$b/$n/$d; [ -f "$out/DONE" ] && { echo "skip $b/$n/$d"; continue; }
  mkdir -p "$out"
  # hard negatives per row: ours / msmarco 9, reasonir / specter 1, in-batch sets 0
  mn=9; [ $n = inbatch ] && mn=0; [ $n = hardneg ] && { [ $d = reasonir ] || [ $d = specter ]; } && mn=1
  echo "== $b / $n / $d"
  case $b in
    qwen3_0.6b|qwen3_4b)
      DATA=${DATA:?DATA (training data root, make_maxlen.py output) is required}
      model=Qwen/Qwen3-Embedding-0.6B; [ $b = qwen3_4b ] && model=Qwen/Qwen3-Embedding-4B
      ( . "$QWEN_VENV/bin/activate"
        MODEL=$model DEEPSPEED=zero2 SAVE_STEPS=50 MIN_NEG=$mn TRAIN_DATA=$DATA/$n/$d/train.jsonl OUTPUT_DIR=$out \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_NAME=$b-$d-$n \
        bash "$TR/qwen/train.sh" --split_dataset_ratio 0 --eval_strategy no --attn_impl flash_attn --seed 42 --data_seed 42 \
          --gradient_checkpointing true
        last=$(ls -d "$out"/checkpoint-* | sort -t- -k2 -n | tail -1)
        CUDA_VISIBLE_DEVICES= python "$TR/qwen/finalize.py" "$last" ) > "$out/train.log" 2>&1 ;;
    nemotron8b)
      DATA_FULL=${DATA_FULL:?DATA_FULL (training data root before make_maxlen.py) is required for nemotron8b}
      ( . "$NEMO_VENV/bin/activate"
        MIN_NEG=$mn TRAIN_DATA=$DATA_FULL/$n/$d/train.jsonl OUTPUT_DIR=$out WANDB_NAME=$b-$d-$n bash "$TR/nemotron/train.sh"
        bash "$TR/nemotron/export.sh" "$out" ) > "$out/train.log" 2>&1 ;;
    *) echo "unknown base $b"; exit 1 ;;
  esac
  touch "$out/DONE"
done; done; done
