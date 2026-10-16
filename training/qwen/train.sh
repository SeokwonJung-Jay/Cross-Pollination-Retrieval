#!/usr/bin/env bash
# Full fine-tuning of Qwen3-Embedding with ms-swift, following the official command in
# QwenLM/Qwen3-Embedding@44548aa docs/training/SWIFT.md.
#
#   TRAIN_DATA=/abs/train.jsonl OUTPUT_DIR=runs/exp bash qwen/train.sh [extra swift sft args...]
#
# Differences from the SWIFT.md command: --tuner_type full (renamed from --train_type in ms-swift 4.1),
# INFONCE_TEMPERATURE=0.1 set explicitly, our data converted by prepare_data.py, --use_hf true, --add_version false, --report_to.
#
# Environment variables (defaults = official command, except the paper settings: 1 epoch, max length 1024, 'Query:' prompt):
#   MODEL (Qwen/Qwen3-Embedding-0.6B)  NPROC (default: number of CUDA_VISIBLE_DEVICES, else all visible GPUs)
#   PER_DEVICE_BS (4)  GRAD_ACC (4)  LR (6e-6)  EPOCHS (1)  EVAL_STEPS (20)  SAVE_STEPS (20)  VAL_RATIO (0.05)
#   MAX_LENGTH (1024)  per-text token limit; cut the data first with common/make_maxlen.py so the final <|endoftext|> is kept
#   DEEPSPEED (zero3; "none" disables)  INFONCE_TEMPERATURE (0.1)
#   INFONCE_MASK_FAKE_NEGATIVE (False)  INFONCE_FAKE_NEG_MARGIN (0.1)  INFONCE_INCLUDE_QQ (False)  INFONCE_INCLUDE_DD (False)
#   QUERY_PROMPT ('Query:')  string prepended to each query as is (written to the checkpoint by finalize.py)
#   QUERY_INSTRUCTION  if set (and QUERY_PROMPT is not), queries use the 'Instruct: {instruction}\nQuery:' format instead
#   MIN_NEG (9)  minimum hard negatives per row checked by validate_data.py; ms-swift uses every negative in a row (9 in our data)
#   SKIP_VALIDATE  1 skips common/validate_data.py
# Extra arguments override arguments of the same name (e.g. --max_length 1024 --max_steps 2).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=$(cd "$HERE/../common" && pwd)

ENV_FILE=${ENV_FILE:-$HERE/../../.env}   # default: <repo>/.env
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
for v in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_NAME HF_TOKEN; do [ -n "${!v:-}" ] || unset "$v"; done

TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA (a jsonl file or a directory of jsonl files) is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
MODEL=${MODEL:-Qwen/Qwen3-Embedding-0.6B}
if [ -z "${NPROC:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  elif command -v nvidia-smi >/dev/null; then NPROC=$(nvidia-smi -L | wc -l)
  else NPROC=1; fi
fi
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-6e-6}
EPOCHS=${EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-1024}
EVAL_STEPS=${EVAL_STEPS:-20}
SAVE_STEPS=${SAVE_STEPS:-20}
VAL_RATIO=${VAL_RATIO:-0.05}
DEEPSPEED=${DEEPSPEED:-zero3}
export INFONCE_TEMPERATURE=${INFONCE_TEMPERATURE:-0.1}
export INFONCE_MASK_FAKE_NEGATIVE=${INFONCE_MASK_FAKE_NEGATIVE:-False}
export INFONCE_FAKE_NEG_MARGIN=${INFONCE_FAKE_NEG_MARGIN:-0.1}
export INFONCE_INCLUDE_QQ=${INFONCE_INCLUDE_QQ:-False}
export INFONCE_INCLUDE_DD=${INFONCE_INCLUDE_DD:-False}
if [ -z "${QUERY_PROMPT+x}" ] && [ -z "${QUERY_INSTRUCTION+x}" ]; then QUERY_PROMPT='Query:'; fi
QUERY_INSTRUCTION=${QUERY_INSTRUCTION:-Given a research problem, retrieve passages proposing a method or technique that could be used to solve it}

mkdir -p "$OUTPUT_DIR"
if [ "${SKIP_VALIDATE:-0}" != 1 ]; then
  # in-batch negatives are per micro-batch (GPUs x PER_DEVICE_BS) with drop_last, so the data must not be smaller than that
  python "$COMMON/validate_data.py" "$TRAIN_DATA" --min-neg "${MIN_NEG:-9}" --global-batch "$((NPROC * PER_DEVICE_BS))"
fi
# query_prompt.txt = the exact string prepended to queries (read by finalize.py)
if [ -n "${QUERY_PROMPT+x}" ]; then
  python "$HERE/prepare_data.py" "$TRAIN_DATA" "$OUTPUT_DIR/swift_train.jsonl" --query-prompt "$QUERY_PROMPT"
  printf '%s' "$QUERY_PROMPT" > "$OUTPUT_DIR/query_prompt.txt"
else
  python "$HERE/prepare_data.py" "$TRAIN_DATA" "$OUTPUT_DIR/swift_train.jsonl" --instruction "$QUERY_INSTRUCTION"
  printf '%s' "$QUERY_INSTRUCTION" > "$OUTPUT_DIR/query_instruction.txt"
  printf 'Instruct: %s\nQuery:' "$QUERY_INSTRUCTION" > "$OUTPUT_DIR/query_prompt.txt"
fi

DS_ARGS=(); [ "$DEEPSPEED" = none ] || DS_ARGS=(--deepspeed "$DEEPSPEED")
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_PROJECT=${WANDB_PROJECT:-qwen3-embedding-ft}
  RUN_NAME=${WANDB_NAME:-$(basename "$OUTPUT_DIR")-$(date -u +%Y%m%d-%H%M%S)}
  REPORT_ARGS=(--report_to wandb --run_name "$RUN_NAME")
  echo "W&B: project $WANDB_PROJECT run $RUN_NAME"
else
  REPORT_ARGS=(--report_to "${REPORT_TO:-none}")
  echo "W&B: off (no WANDB_API_KEY) | report_to ${REPORT_TO:-none}"
fi
echo "GPUs $NPROC x per-device $PER_DEVICE_BS x grad-acc $GRAD_ACC = $((NPROC * PER_DEVICE_BS * GRAD_ACC)) queries/step (official: 8x4x4=128);" \
     "in-batch negatives per micro-batch of $((NPROC * PER_DEVICE_BS)) queries; temperature $INFONCE_TEMPERATURE;" \
     "fake-neg mask $INFONCE_MASK_FAKE_NEGATIVE (margin $INFONCE_FAKE_NEG_MARGIN), q-q $INFONCE_INCLUDE_QQ, d-d $INFONCE_INCLUDE_DD; deepspeed $DEEPSPEED"

# ms-swift launches torchrun when NPROC_PER_NODE is set; with DeepSpeed this is needed even on one GPU
# (otherwise ZeRO-3 init falls back to MPI discovery and fails without mpi4py).
if [ "$NPROC" -gt 1 ] || [ "$DEEPSPEED" != none ]; then export NPROC_PER_NODE=$NPROC; fi
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --task_type embedding \
    --model_type qwen3_emb \
    --tuner_type full \
    --dataset "$OUTPUT_DIR/swift_train.jsonl" \
    --split_dataset_ratio "$VAL_RATIO" \
    --eval_strategy steps \
    --output_dir "$OUTPUT_DIR" \
    --add_version false \
    --eval_steps "$EVAL_STEPS" \
    --num_train_epochs "$EPOCHS" \
    --save_steps "$SAVE_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --learning_rate "$LR" \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --max_length "$MAX_LENGTH" \
    --truncation_strategy right \
    "${DS_ARGS[@]}" \
    "${REPORT_ARGS[@]}" \
    "$@"
