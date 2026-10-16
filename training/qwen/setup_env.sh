#!/usr/bin/env bash
# Installs the qwen/ training environment from the pinned versions in requirements.lock (--no-deps).
#
#   bash qwen/setup_env.sh                     # default venv: training/.venv-qwen, CUDA 12.8 torch
#   TORCH_BACKEND=cpu bash qwen/setup_env.sh   # CPU-only torch
#   VENV=/path/venv bash qwen/setup_env.sh
#   SKIP_FLASH_ATTN=1 bash qwen/setup_env.sh   # skip building flash-attn
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HERE/../.venv-qwen}
TORCH_BACKEND=${TORCH_BACKEND:-cu128}
LOCK=$HERE/requirements.lock

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"

# torch for TORCH_BACKEND, the rest as locked; deepspeed is an sdist built without isolation after torch.
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
grep -vE '^(deepspeed|torch)==' "$LOCK" > "$TMP/rest.txt"
if [ "$TORCH_BACKEND" != cu128 ]; then sed -i '/^nvidia-\|^triton==\|^cuda-/d' "$TMP/rest.txt"; fi
uv pip install -p "$VENV" --torch-backend "$TORCH_BACKEND" "torch==$(grep -oP '^torch==\K[0-9.]+' "$LOCK")"
uv pip install -p "$VENV" --no-deps -r "$TMP/rest.txt"
uv pip install -p "$VENV" --no-deps --no-build-isolation "$(grep '^deepspeed==' "$LOCK")"

"$VENV/bin/python" -c "import swift, torch, transformers, deepspeed; print('ms-swift', swift.__version__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| deepspeed', deepspeed.__version__)"
# flash-attn 2.8.3 (used in all our runs, --attn_impl flash_attn) is built from source against the installed torch (CUDA 12.8, A100 = sm80).
if [ "${SKIP_FLASH_ATTN:-0}" != 1 ] && [ "$TORCH_BACKEND" = cu128 ]; then
  FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-80} uv pip install -p "$VENV" --no-deps --no-build-isolation "flash-attn==2.8.3"
fi
echo "Next: . $VENV/bin/activate"
