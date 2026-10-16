#!/bin/bash
# Leave-one-out evaluation: the main-table retrievers (evaluation/main/models.json, same settings, max length 512) on the three leave-one-out pools.
#   prepare.py (3 x 1,300 pseudo-queries in one work dir) -> evaluation/main/encode.py / encode_lasttoken.py per model -> split.py (score.py per
#   pool) -> compare.py (Recall@10 change and Kendall's tau against the original pools).
# Same scheduling as evaluation/main/run.sh: GPU models through one shared queue (one worker per GPU), inf-retriever after the queue on all
# GPUs, CPU / API models (bm25, gemini) beside the GPU workers. A model whose WORK/<model>/results/scores.json exists is skipped.
#   WORK=/path/loo_work LOO_DIR=/path/loo_pools ORIG=/path/main_results/per_query.json [EVAL=EVAL_DIR/eval_texts.jsonl] [GPUS="0 1 ..."] bash run_eval.sh
# LOO_DIR = export.py output (eval_loo_{e5,splade,scincl}.jsonl). EVAL defaults to eval_texts.jsonl in EVAL_DIR, else RELEASE_DIR/eval.
# ORIG = per_query.json of the main evaluation on the original pools
# (evaluation/main/run.sh output, OUT/per_query.json). Python environments and secrets as in evaluation/main/run.sh (QWEN_PY, LATEON_PY,
# SPECTER2_PY, M3_PY, INFRET_PY; GEMINI_API_KEY, optional HF_TOKEN from the environment or ENV_FILE, default <repo>/.env). Logs: $WORK/_logs/.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); MAIN=$HERE/../main
: ${WORK:?WORK (output dir) is required}; : ${LOO_DIR:?LOO_DIR (export.py output) is required}
REGISTRY=${REGISTRY:-$MAIN/models.json}; OUT=${OUT:-$WORK/_split}
QWEN_PY=${QWEN_PY:-$HERE/../../training/.venv-qwen/bin/python}; LATEON_PY=${LATEON_PY:-$MAIN/.venv-lateon/bin/python}
SPECTER2_PY=${SPECTER2_PY:-$MAIN/.venv-specter2/bin/python}; M3_PY=${M3_PY:-$MAIN/.venv-m3/bin/python}; INFRET_PY=${INFRET_PY:-$MAIN/.venv-infret/bin/python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ENV_FILE=${ENV_FILE:-$HERE/../../.env}; [ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }   # default: <repo>/.env
py() { case $1 in qwen) echo $QWEN_PY;; lateon) echo $LATEON_PY;; specter2) echo $SPECTER2_PY;; m3) echo $M3_PY;; infret) echo $INFRET_PY;; esac; }
env_of() { $M3_PY -c "import json,sys;d=json.load(open('$REGISTRY'));print((d['_shared'].get(sys.argv[1]) or d[sys.argv[1]])['env'])" "$1"; }
done_() { [ "$1" = bge_m3 ] && set -- bge_m3_multivec; [ -f "$WORK/$1/results/scores.json" ]; }
GPU_MODELS=${MODELS:-"qwen3_emb_4b bge_m3 qwen3_emb_0.6b openscholar_retriever lateon specter2"}; CPU_MODELS=${CPU_MODELS-"bm25 gemini_emb_001"}
LASTTOKEN_MODELS=${LASTTOKEN_MODELS-"inf_retriever_v1_pro"}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}; read -ra G <<< "$GPUS"
L=$WORK/_logs; mkdir -p "$L"
[ -f "$WORK/_data/pools.json" ] || $M3_PY -I "$HERE/prepare.py" --work "$WORK" --loo-dir "$LOO_DIR" ${EVAL:+--eval "$EVAL"} > "$L/prepare.txt" 2>&1 \
  || { echo "prepare failed, see $L/prepare.txt"; exit 1; }
Q=$L/todo; : > "$Q"; for m in $GPU_MODELS; do done_ $m || echo "$m" >> "$Q"; done
take() { local m; m=$(head -n 1 "$Q"); [ -n "$m" ] && sed -i 1d "$Q"; echo "$m"; }
export -f take; export Q
run1() {  # model device label
  echo "[$3] $1 start $(date -u +%FT%TZ)" >> "$L/schedule.txt"
  $(py $(env_of $1)) -I "$MAIN/encode.py" "$1" --work "$WORK" --device "$2" --registry "$REGISTRY" >> "$L/$1.txt" 2>&1 && s=done || s=FAILED
  echo "[$3] $1 $s $(date -u +%FT%TZ)" >> "$L/schedule.txt"
}
worker() { while true; do m=$(flock "$Q.lock" bash -c take); [ -z "$m" ] && break; CUDA_VISIBLE_DEVICES=$1 run1 "$m" cuda:0 "gpu$1"; done; }
for m in $CPU_MODELS; do done_ $m || run1 "$m" cpu cpu & done
for g in "${G[@]}"; do worker "$g" & done; wait
DEVS=$(for g in "${G[@]}"; do printf "cuda:%s," $g; done); DEVS=${DEVS%,}
for m in $LASTTOKEN_MODELS; do
  done_ $m && continue
  echo "[all] $m start $(date -u +%FT%TZ)" >> "$L/schedule.txt"
  $(py $(env_of $m)) -I "$MAIN/encode_lasttoken.py" "$m" --work "$WORK" --devices "$DEVS" --registry "$REGISTRY" >> "$L/$m.txt" 2>&1 && s=done || s=FAILED
  echo "[all] $m $s $(date -u +%FT%TZ)" >> "$L/schedule.txt"
done
grep FAILED "$L/schedule.txt" && echo "some models failed, see $L/<model>.txt"
$M3_PY -I "$HERE/split.py" --work "$WORK" --out "$OUT" --registry "$REGISTRY" || exit 1
[ -n "${ORIG:-}" ] && $M3_PY -I "$HERE/compare.py" --orig "$ORIG" --loo "$OUT" --out "$OUT/compare.json"
