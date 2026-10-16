#!/bin/bash
# Main experiment: prepare -> encode -> score (11 models, CPR-eval, max length 512).
# GPU models go through one shared queue (largest first); each GPU worker claims the next model, so N GPUs run N models at once.
# inf-retriever-v1-pro (kind hf_lasttoken) runs after the queue on all GPUs (encode_lasttoken.py). API / CPU models (gemini, bm25) run beside
# the GPU workers. A model whose results/scores.json exists is skipped (rerun = delete its dir).
#   RELEASE_DIR=/path/CPR-Dataset WORK=/path/main_work [GPUS="0 1 ..."] [MODELS="..."] [CPU_MODELS="..."] bash run.sh      (logs: $WORK/_logs/)
# Python environments (exact package lists in env/): QWEN_PY, LATEON_PY, SPECTER2_PY, M3_PY (BM25, BGE-M3, OpenScholar, Gemini), INFRET_PY.
# Eval set: EVAL_DIR (else RELEASE_DIR/eval) with eval_texts.jsonl of dataset/release/fetch_abstracts.py (prepare.py).
# Secrets from the environment or ENV_FILE (default <repo>/.env): HF_TOKEN (optional, HF downloads), GEMINI_API_KEY.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); : ${WORK:?WORK (output dir) is required}
REGISTRY=${REGISTRY:-$HERE/models.json}; OUT=${OUT:-$HERE/results}; TITLE=${TITLE:-Main results}
QWEN_PY=${QWEN_PY:-$HERE/../../training/.venv-qwen/bin/python}; LATEON_PY=${LATEON_PY:-$HERE/.venv-lateon/bin/python}
SPECTER2_PY=${SPECTER2_PY:-$HERE/.venv-specter2/bin/python}; M3_PY=${M3_PY:-$HERE/.venv-m3/bin/python}; INFRET_PY=${INFRET_PY:-$HERE/.venv-infret/bin/python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ENV_FILE=${ENV_FILE:-$HERE/../../.env}; [ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }   # default: <repo>/.env
py() { case $1 in qwen) echo $QWEN_PY;; lateon) echo $LATEON_PY;; specter2) echo $SPECTER2_PY;; m3) echo $M3_PY;; infret) echo $INFRET_PY;; esac; }
env_of() { $M3_PY -c "import json,sys;d=json.load(open('$REGISTRY'));print((d['_shared'].get(sys.argv[1]) or d[sys.argv[1]])['env'])" "$1"; }
done_() { [ "$1" = bge_m3 ] && set -- bge_m3_multivec; [ -f "$WORK/$1/results/scores.json" ]; }
GPU_MODELS=${MODELS:-"qwen3_emb_4b bge_m3 qwen3_emb_0.6b openscholar_retriever lateon specter2"}; CPU_MODELS=${CPU_MODELS-"bm25 gemini_emb_001"}
LASTTOKEN_MODELS=${LASTTOKEN_MODELS-"inf_retriever_v1_pro"}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}; read -ra G <<< "$GPUS"
$M3_PY -I "$HERE/prepare.py" --work "$WORK" || exit 1
L=$WORK/_logs; mkdir -p "$L"; Q=$L/todo; : > "$Q"; for m in $GPU_MODELS; do done_ $m || echo "$m" >> "$Q"; done
take() { local m; m=$(head -n 1 "$Q"); [ -n "$m" ] && sed -i 1d "$Q"; echo "$m"; }
export -f take; export Q
run1() {  # model device label
  echo "[$3] $1 start $(date -u +%FT%TZ)" >> "$L/schedule.txt"
  $(py $(env_of $1)) -I "$HERE/encode.py" "$1" --work "$WORK" --device "$2" --registry "$REGISTRY" >> "$L/$1.txt" 2>&1 && s=done || s=FAILED
  echo "[$3] $1 $s $(date -u +%FT%TZ)" >> "$L/schedule.txt"
}
worker() { while true; do m=$(flock "$Q.lock" bash -c take); [ -z "$m" ] && break; CUDA_VISIBLE_DEVICES=$1 run1 "$m" cuda:0 "gpu$1"; done; }
for m in $CPU_MODELS; do done_ $m || run1 "$m" cpu cpu & done
for g in "${G[@]}"; do worker "$g" & done; wait
DEVS=$(for g in "${G[@]}"; do printf "cuda:%s," $g; done); DEVS=${DEVS%,}
for m in $LASTTOKEN_MODELS; do
  done_ $m && continue
  echo "[all] $m start $(date -u +%FT%TZ)" >> "$L/schedule.txt"
  $(py $(env_of $m)) -I "$HERE/encode_lasttoken.py" "$m" --work "$WORK" --devices "$DEVS" --registry "$REGISTRY" >> "$L/$m.txt" 2>&1 && s=done || s=FAILED
  echo "[all] $m $s $(date -u +%FT%TZ)" >> "$L/schedule.txt"
done
grep FAILED "$L/schedule.txt" && echo "some models failed, see $L/<model>.txt"
$M3_PY -I "$HERE/score.py" --work "$WORK" --registry "$REGISTRY" --out "$OUT" --title "$TITLE"
