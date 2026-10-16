#!/usr/bin/env bash
# Installs the nemotron/ training environment (NeMo AutoModel) from the pinned versions in requirements.lock (--no-deps).
# NeMo AutoModel is installed from a clone at the pinned commit, kept in VENV/src/Automodel (export.sh runs its tools/offline_hf_consolidation.py).
#
#   bash nemotron/setup_env.sh                     # default venv: training/.venv-nemotron, CUDA 12.8 torch
#   VENV=/path/venv bash nemotron/setup_env.sh
#   SKIP_FLASH_ATTN=1 bash nemotron/setup_env.sh   # skip building flash-attn (training uses flash_attention_2)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HERE/../.venv-nemotron}
LOCK=$HERE/requirements.lock
AUTOMODEL_COMMIT=8e562fceb1c1271f579e85de0bfd4f30bec746a9   # = requirements.in
export NO_VCS_VERSION=1   # NeMo AutoModel's setup otherwise queries git for the version string

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"
SRC=$VENV/src/Automodel
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$SRC"; git -C "$SRC" init -q
  git -C "$SRC" fetch -q --depth 1 https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$AUTOMODEL_COMMIT" ] || { echo "$SRC is not at $AUTOMODEL_COMMIT"; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
grep -vE '^(torch==|nemo-automodel )' "$LOCK" > "$TMP/rest.txt"
uv pip install -p "$VENV" --torch-backend cu128 "torch==$(grep -oP '^torch==\K[0-9.]+' "$LOCK")"
uv pip install -p "$VENV" --no-deps -r "$TMP/rest.txt"
uv pip install -p "$VENV" --no-deps "$SRC"

"$VENV/bin/python" -c "import torch, transformers, nemo_automodel; print('nemo_automodel', nemo_automodel.__version__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
# flash-attn 2.8.3 (used in all our runs, attn_implementation flash_attention_2) is built from source against the installed torch (CUDA 12.8, A100 = sm80).
# Triton compiles a C helper at the first step and needs Python.h (uv's managed Python has it; a system Python needs python3.12-dev).
if [ "${SKIP_FLASH_ATTN:-0}" != 1 ]; then
  FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-80} uv pip install -p "$VENV" --no-deps --no-build-isolation "flash-attn==2.8.3"
fi
echo "Next: . $VENV/bin/activate"
