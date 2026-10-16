#!/usr/bin/env bash
# Installs the llama-embed-nemotron-8b evaluation environment from the pinned versions in requirements-eval.lock (--no-deps):
# transformers 4.51.0 (required by the model's official remote code), sentence-transformers 5.1.2, flash-attn 2.8.3.
# Used by evaluation/{bright,analobench,main}/encode.py for nemotron8b_qp and the abl2v2_nemotron8b_* checkpoints.
#
#   bash nemotron/setup_env_eval.sh                     # default venv: training/.venv-nemotron-eval, CUDA 12.8 torch
#   VENV=/path/venv bash nemotron/setup_env_eval.sh
#   SKIP_FLASH_ATTN=1 bash nemotron/setup_env_eval.sh   # skip building flash-attn (the encoders load the model with flash_attention_2)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HERE/../.venv-nemotron-eval}
LOCK=$HERE/requirements-eval.lock

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
grep -vE '^torch==' "$LOCK" > "$TMP/rest.txt"
uv pip install -p "$VENV" --torch-backend cu128 "torch==$(grep -oP '^torch==\K[0-9.]+' "$LOCK")"
uv pip install -p "$VENV" --no-deps -r "$TMP/rest.txt"
"$VENV/bin/python" -c "import torch, transformers, sentence_transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| sentence-transformers', sentence_transformers.__version__)"
if [ "${SKIP_FLASH_ATTN:-0}" != 1 ]; then
  FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-80} uv pip install -p "$VENV" --no-deps --no-build-isolation "flash-attn==2.8.3"
fi
echo "Next: NEMO_PY=$VENV/bin/python (ablation/dataset/eval.sh)"
