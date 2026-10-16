#!/usr/bin/env bash
# Full fine-tuning of llama-embed-nemotron-8b with NeMo AutoModel, following its official bi-encoder recipe
# (examples/retrieval/bi_encoder/llama_embed_nemotron_8b @ 8e562fce); the differences are listed in train.yaml.
#
#   TRAIN_DATA=/abs/train.jsonl OUTPUT_DIR=runs/exp bash nemotron/train.sh [extra --dotted.key value overrides]
#
# TRAIN_DATA: one JSONL file in our format, texts not cut (the collator truncates queries / documents to 1024 tokens); every row has the
# same number of hard negatives (all are used: n_passages = negatives + 1; 0 = in-batch negatives only).
# Environment variables (defaults = paper settings):
#   NPROC (default: number of CUDA_VISIBLE_DEVICES, else all visible GPUs; paper: 8). Global batch 128 = NPROC x 1 x gradient accumulation
#         128 / NPROC; in-batch negatives are gathered over the NPROC GPUs (NPROC queries x n_passages documents per micro-step).
#   MIN_NEG (9)  minimum hard negatives per row checked by common/validate_data.py      SKIP_VALIDATE  1 skips the check
#   BASE_DIR (OUTPUT_DIR/base_model)  base model folder written by prepare_base.py
#   WANDB_API_KEY (optional; W&B is off without it), WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME; HF_TOKEN. Read from the environment or ENV_FILE.
# Outputs in OUTPUT_DIR: nemo_train.jsonl (prepare_data.py), train.yaml (the filled config), checkpoints/epoch_0_step_<N>/ (every 50 steps
# and the final step; 1 epoch of 10,573 rows on 8 GPUs = 83 steps, final = epoch_0_step_82). Then: bash nemotron/export.sh OUTPUT_DIR.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=$(cd "$HERE/../common" && pwd)

ENV_FILE=${ENV_FILE:-$HERE/../../.env}   # default: <repo>/.env
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
for v in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_NAME HF_TOKEN; do [ -n "${!v:-}" ] || unset "$v"; done

TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA (a jsonl file) is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
if [ -z "${NPROC:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  elif command -v nvidia-smi >/dev/null; then NPROC=$(nvidia-smi -L | wc -l)
  else NPROC=1; fi
fi
mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd); TRAIN_DATA=$(cd "$(dirname "$TRAIN_DATA")" && pwd)/$(basename "$TRAIN_DATA")
BASE_DIR=${BASE_DIR:-$OUTPUT_DIR/base_model}

if [ "${SKIP_VALIDATE:-0}" != 1 ]; then
  python "$COMMON/validate_data.py" "$TRAIN_DATA" --min-neg "${MIN_NEG:-9}" --global-batch "$NPROC"
fi
NNEG=$(python -I "$HERE/prepare_data.py" "$TRAIN_DATA" "$OUTPUT_DIR/nemo_train.jsonl" | tee /dev/stderr | awk '/negatives per row/ {print $NF}')
SNAP=$(python -I "$HERE/prepare_base.py" "$BASE_DIR" | tail -n 1)

# OUTPUT_DIR/train.yaml = train.yaml with the run's paths and n_passages (+ a wandb section when WANDB_API_KEY is set)
RUN_NAME=${WANDB_NAME:-$(basename "$OUTPUT_DIR")-$(date -u +%Y%m%d-%H%M%S)}
python -I - "$HERE/train.yaml" "$OUTPUT_DIR/train.yaml" "$BASE_DIR" "$SNAP" "$OUTPUT_DIR/nemo_train.jsonl" "$((NNEG + 1))" "$OUTPUT_DIR/checkpoints" \
    "${WANDB_API_KEY:+${WANDB_PROJECT:-nemotron-embed-ft}}" "$RUN_NAME" "${WANDB_ENTITY:-}" <<'PY'
import sys
src, dst, base, tok, data, npass, ckpt, project, name, entity = sys.argv[1:]
s = open(src).read()
for k, v in (('@BASE_MODEL@', base), ('@TOKENIZER@', tok), ('@TRAIN_FILE@', data), ('@N_PASSAGES@', npass), ('@CKPT_DIR@', ckpt)):
    assert k in s, k; s = s.replace(k, v)
if project: s += f'\nwandb:\n  project: {project}\n  name: {name}\n' + (f'  entity: {entity}\n' if entity else '')
open(dst, 'w').write(s)
PY
if [ -n "${WANDB_API_KEY:-}" ]; then echo "W&B: project ${WANDB_PROJECT:-nemotron-embed-ft} run $RUN_NAME"; else export WANDB_MODE=disabled; echo "W&B: off (no WANDB_API_KEY)"; fi
echo "GPUs $NPROC x local 1 x grad-acc $((128 / NPROC)) = 128 queries/step; n_passages $((NNEG + 1)) (hard negatives $NNEG) + in-batch negatives over $NPROC GPUs;" \
     "max length 1024; lr 2e-6; temperature 0.02"

export NO_VCS_VERSION=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$OUTPUT_DIR"   # relative paths of the recipe resolve here (use a local disk: 8 ranks stat-ing a network mount can stall)
torchrun --nproc-per-node "$NPROC" "$HERE/finetune.py" --config "$OUTPUT_DIR/train.yaml" "$@"
