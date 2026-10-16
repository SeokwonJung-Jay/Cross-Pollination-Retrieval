#!/usr/bin/env bash
# Held-out domain evaluation on CPR-eval, max length 1024: every model encodes all 187,411 documents and 1,300 queries
# (evaluation/main/encode.py, documents over all GPUs), evaluation/main/score.py scores every query against its pool of 151, and
# tables.py keeps the queries of each domain (fold model = the one trained without that domain).
# Models = keys of models.json: vanilla qwen3_0.6b / qwen3_4b (pinned HF revisions), the all-domain runs abl2v2_<base>_hardneg_ours (the same
# entries as ablation/dataset, so a WORK shared with the dataset ablation reuses their encodings; checkpoints under DATASET_RUNS = OUT of
# ablation/dataset/run.sh) and the 8 fold runs domabl_<base>_<domain> (checkpoints under RUNS = OUT of run.sh). Each fine-tuned model = the
# last finalized checkpoint-<N> of its run dir, resolved by ../dataset/resolve_runs.py into $WORK/_logs/models_domain.json; a run that is
# not found (or its variable unset) is reported and skipped.
# Finished models (WORK/ours/<model>/results/scores.json) are skipped.
#   RUNS=/path/runs DATASET_RUNS=/path/dataset_runs EVAL_DIR=/path/eval_release WORK=/path/eval [GPUS="0 1 ..."] [FOLDS_JSON=FOLDS/folds.json] bash eval.sh [MODEL ...]     (logs: $WORK/_logs/)
# EVAL_DIR (else RELEASE_DIR/eval): CPR-eval release dir with eval_texts.jsonl (evaluation/main/prepare.py).
# Python environment: QWEN_PY (the Qwen training environment). Tables: results/ours/ (score.py) and results/{eval.json, RES.md}.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); EV=$(cd "$HERE/../../evaluation" && pwd); : ${WORK:?WORK (output dir) is required}
REGISTRY_SRC=${REGISTRY:-$HERE/models.json}; R=${RES_DIR:-$HERE/results}
QWEN_PY=${QWEN_PY:-$HERE/../../training/.venv-qwen/bin/python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}; read -ra G <<< "$GPUS"
L=$WORK/_logs; mkdir -p $L $R/ours
REGISTRY=$L/models_domain.json   # models.json with the fine-tuned runs resolved to their checkpoint dirs
$QWEN_PY -I $HERE/../dataset/resolve_runs.py "$REGISTRY_SRC" "$REGISTRY" > /dev/null || { echo "resolve_runs.py failed"; exit 1; }

[ -f $WORK/ours/_data/pools.json ] || $QWEN_PY -I $EV/main/prepare.py --work $WORK/ours || exit 1
MODELS=${*:-$($QWEN_PY -c "import json, sys; print(' '.join(k for k in json.load(open(sys.argv[1])) if not k.startswith('_')))" "$REGISTRY")}
DEVS=$(for g in "${G[@]}"; do printf "cuda:%s," $g; done); DEVS=${DEVS%,}
for m in $MODELS; do
  $QWEN_PY -c "import json, sys; sys.exit(sys.argv[2] not in json.load(open(sys.argv[1])))" "$REGISTRY" "$m" || { echo "skip $m: no checkpoint"; continue; }
  [ -f $WORK/ours/$m/results/scores.json ] && { echo "skip $m"; continue; }
  echo "[all] ours $m start $(date -u +%FT%TZ)" >> $L/schedule.txt
  $QWEN_PY -I $EV/main/encode.py $m --work $WORK/ours --registry $REGISTRY --device cuda:${G[0]} --devices $DEVS < /dev/null >> $L/${m}_ours.txt 2>&1 && st=done || st=FAILED
  echo "[all] ours $m $st $(date -u +%FT%TZ)" >> $L/schedule.txt; [ $st = FAILED ] && echo "FAILED ours $m (log $L/${m}_ours.txt)"
done

$QWEN_PY -I $EV/main/score.py --work $WORK/ours --registry $REGISTRY --out $R/ours --title "Held-out domain: CPR-eval (max length 1024)" \
  > $L/score_ours.txt 2>&1 || echo "ours scoring FAILED"
$QWEN_PY -I $HERE/tables.py --work $WORK/ours --out $R ${FOLDS_JSON:+--folds $FOLDS_JSON}
