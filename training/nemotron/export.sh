#!/usr/bin/env bash
# Final checkpoint of a train.sh run -> sentence-transformers folder for evaluation.
#  1. consolidates the sharded FSDP checkpoint OUTPUT_DIR/checkpoints/epoch_*_step_<last>/model to bf16 safetensors with NeMo AutoModel's
#     tools/offline_hf_consolidation.py (CPU, 16 processes x 5 threads; same tensor names as the base model);
#  2. EVAL_DIR = the base repo's config.json (with auto_map), remote code, sentence-transformers configs and tokenizer + these weights
#     (the encoders load it with trust_remote_code under transformers 4.51.0 and pass the query prompt 'Query:' explicitly).
#   bash nemotron/export.sh OUTPUT_DIR [EVAL_DIR=OUTPUT_DIR/eval_model]      (in the training environment)
#   AUTOMODEL_SRC: NeMo AutoModel clone at the pinned commit (default: <venv>/src/Automodel from setup_env.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$(cd "${1:?OUTPUT_DIR of train.sh is required}" && pwd); EVAL_DIR=${2:-$RUN/eval_model}
SRC=${AUTOMODEL_SRC:-$(python -c 'import sys; print(sys.prefix)')/src/Automodel}
[ -f "$SRC/tools/offline_hf_consolidation.py" ] || { echo "NeMo AutoModel source not found at $SRC (set AUTOMODEL_SRC)"; exit 1; }
C=$(ls -d "$RUN"/checkpoints/epoch_*_step_* | awk -F_step_ '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
echo "final checkpoint: $C"
SNAP=$(python -I "$HERE/prepare_base.py" "${BASE_DIR:-$RUN/base_model}" | tail -n 1)
if [ ! -f "$C/model/consolidated_bf16/model.safetensors.index.json" ]; then
  CUDA_VISIBLE_DEVICES="" NO_VCS_VERSION=1 torchrun --nproc-per-node=16 "$SRC/tools/offline_hf_consolidation.py" --backend gloo --num-threads 5 \
    --model-name "${BASE_DIR:-$RUN/base_model}" --input-dir "$C/model" --output-dir "$C/model/consolidated_bf16" --cast-dtype bf16
fi
mkdir -p "$EVAL_DIR/1_Pooling"
for f in config.json config_sentence_transformers.json sentence_bert_config.json modules.json llama_bidirectional_model.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json LICENSE Notice.txt 1_Pooling/config.json; do cp -L "$SNAP/$f" "$EVAL_DIR/$f"; done
cp "$C"/model/consolidated_bf16/model-*.safetensors "$C/model/consolidated_bf16/model.safetensors.index.json" "$EVAL_DIR/"
python -I - "$SNAP" "$EVAL_DIR" <<'PY'
import json, sys
b, o = sys.argv[1:3]
a = json.load(open(f'{b}/model.safetensors.index.json'))['weight_map']; n = json.load(open(f'{o}/model.safetensors.index.json'))['weight_map']
assert set(a) == set(n), 'tensor names differ from the base model'
print('ok', len(n), 'tensors ->', o)
PY
