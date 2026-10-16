#!/usr/bin/env bash
# Dataset ablation evaluation, max length 1024: BRIGHT (7 StackExchange splits, all 749 queries), AnaloBench, CPR-eval.
# Models = keys of models.json: the vanilla bases (qwen3_0.6b, qwen3_4b, nemotron8b; pinned HF revisions) and the 24 runs
# abl2v2_<base>_<neg>_<set>, your own run.sh outputs under RUNS (= OUT of run.sh): Qwen = the last finalized checkpoint-<N> of
# $RUNS/<base>/<neg>/<set>/, nemotron8b = its eval_model/ (export.sh). resolve_runs.py writes the resolved registry to $WORK/_logs/models.json;
# a run that is not found (or RUNS unset) is reported and skipped.
# Jobs go through one queue over the GPUs, largest first: CPR-eval (187k documents over all GPUs at once, one model at a time), BRIGHT
# document chunks (each split in parts, encode.py --shard), AnaloBench; then BRIGHT queries + ranking; then scoring of every model in WORK.
# Every encoder skips finished outputs, so a rerun resumes.
#   RUNS=/path/runs EVAL_DIR=/path/eval_release WORK=/path/eval [GPUS="0 1 ..."] bash eval.sh [MODEL ...]      (default: every model in models.json; logs: $WORK/_logs/)
# EVAL_DIR (else RELEASE_DIR/eval): CPR-eval release dir with eval_texts.jsonl (evaluation/main/prepare.py).
# Python environments: QWEN_PY (Qwen; the Qwen training environment), NEMO_PY (llama-embed-nemotron-8b; training/nemotron/setup_env_eval.sh).
# Tables: results/{bright,analobench,ours}/ and results/RES.md.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); EV=$(cd "$HERE/../../evaluation" && pwd); : ${WORK:?WORK (output dir) is required}
REGISTRY_SRC=${REGISTRY:-$HERE/models.json}; R=${RES_DIR:-$HERE/results}
QWEN_PY=${QWEN_PY:-$HERE/../../training/.venv-qwen/bin/python}; NEMO_PY=${NEMO_PY:-$HERE/../../training/.venv-nemotron-eval/bin/python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false QWEN_EVAL_MAXLEN=1024
SE="biology earth_science economics psychology robotics stackoverflow sustainable_living"
# documents per split: earth_science 121k, stackoverflow 107k, robotics 62k, sustainable_living 61k, biology 57k, psychology 52k, economics 50k
SHARDS="earth_science:5 stackoverflow:4 robotics:3 sustainable_living:3 biology:2 psychology:2 economics:2"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}; read -ra G <<< "$GPUS"
L=$WORK/_logs; mkdir -p $L $R/{bright,analobench,ours}
REGISTRY=$L/models.json   # models.json with the fine-tuned runs resolved to their checkpoint dirs
$QWEN_PY -I $HERE/resolve_runs.py "$REGISTRY_SRC" "$REGISTRY" > /dev/null || { echo "resolve_runs.py failed"; exit 1; }

[ -f $WORK/bright/data/documents_economics.parquet ] || $QWEN_PY -I $EV/bright/prepare.py --out $WORK/bright/data --splits $SE || exit 1
[ -f $WORK/analobench/data/L30.json ] || $QWEN_PY -I $EV/analobench/prepare.py --work $WORK/analobench || exit 1
[ -f $WORK/ours/_data/pools.json ] || $QWEN_PY -I $EV/main/prepare.py --work $WORK/ours || exit 1

# one line per model: name kind batch ckpt(- = vanilla) bright_name analobench_name
PLAN=$L/plan.txt
$QWEN_PY -I - "$REGISTRY" "$@" > $PLAN <<'EOF' || { echo "plan failed"; exit 1; }
import json, sys
REG = json.load(open(sys.argv[1])); names = sys.argv[2:] or [k for k in REG if not k.startswith('_')]
VAN = {'qwen3_0.6b': ('qwen06b_qp', 'qwen06b'), 'qwen3_4b': ('qwen4b_qp', 'qwen4b'), 'nemotron8b': ('nemotron8b_qp', 'nemotron8b_qp')}
for n in names:
    if n not in REG: print(f'skip {n}: not in the resolved registry (see the resolve_runs.py messages above)', file=sys.stderr); continue
    e = REG[n]; kind = 'nemo' if e.get('remote_code') else 'qwen'
    if n in VAN: print(n, kind, e['batch'], '-', *VAN[n]); continue
    print(n, kind, e['batch'], e['hf'], n, n)
EOF
cat $PLAN

Q=$L/todo; : > $Q.1; : > $Q.2; : > $Q.3
while read -r name kind bs ck nb na; do
  for x in $SHARDS; do s=${x%:*}; n=${x#*:}; for ((i = 0; i < n; i++)); do echo "bright:$s:$i/$n $name $kind $bs $ck $nb" >> $Q.1; done
    echo "bright:$s $name $kind $bs $ck $nb" >> $Q.3; done   # phase 2: queries + ranking once every chunk of the split is saved
  echo "analobench $name $kind $bs $ck $na" >> $Q.2
done < $PLAN
cat $Q.1 $Q.2 > $Q; rm -f $Q.1 $Q.2
take() { local j; j=$(head -n 1 "$Q"); [ -n "$j" ] && sed -i 1d "$Q"; echo "$j"; }
export -f take; export Q
job() {  # bench name kind batch ckpt bench_name   (one GPU: CUDA_VISIBLE_DEVICES)
  local b=$1 kind=$3 bs=$4 ck=$5 n=$6 PY=$QWEN_PY CKA=""; [ $kind = nemo ] && PY=$NEMO_PY; [ $ck = - ] || CKA="--ckpt $ck"
  case $b in
    bright:*:*) b=${b#bright:}; $PY -I $EV/bright/encode.py $n --data $WORK/bright/data --out $WORK/bright --splits ${b%%:*} --shard ${b#*:} --batch $bs $CKA;;
    bright:*) $PY -I $EV/bright/encode.py $n --data $WORK/bright/data --out $WORK/bright --splits ${b#bright:} --batch $bs $CKA;;
    analobench) $PY -I $EV/analobench/encode.py $n --work $WORK/analobench --batch $bs $CKA;;
  esac
}
worker() { while true; do j=$(flock "$Q.lock" bash -c take); [ -z "$j" ] && break; set -- $j; lg=$L/${2}_${1//[:\/]/_}.txt
  echo "[gpu$g] $1 $2 start $(date -u +%FT%TZ)" >> $L/schedule.txt
  CUDA_VISIBLE_DEVICES=$g job "$@" >> $lg 2>&1 && st=done || st=FAILED; echo "[gpu$g] $1 $2 $st $(date -u +%FT%TZ)" >> $L/schedule.txt
  [ $st = FAILED ] && echo "FAILED $1 $2 (log $lg)"; done; }
DEVS=$(for g in "${G[@]}"; do printf "cuda:%s," $g; done); DEVS=${DEVS%,}
while read -r name kind bs ck nb na; do   # CPR-eval first, documents encoded on all GPUs at once
  [ ! -f $WORK/ours/$name/results/scores.json ] || continue
  PY=$QWEN_PY; [ $kind = nemo ] && PY=$NEMO_PY
  echo "[all] ours $name start $(date -u +%FT%TZ)" >> $L/schedule.txt
  $PY -I $EV/main/encode.py $name --work $WORK/ours --registry $REGISTRY --device cuda:${G[0]} --devices $DEVS < /dev/null >> $L/${name}_ours.txt 2>&1 && st=done || st=FAILED
  echo "[all] ours $name $st $(date -u +%FT%TZ)" >> $L/schedule.txt; [ $st = FAILED ] && echo "FAILED ours $name"
done < $PLAN
for g in "${G[@]}"; do worker & done; wait
cp $Q.3 $Q; rm -f $Q.3; for g in "${G[@]}"; do worker & done; wait   # BRIGHT phase 2

# scoring: every model evaluated so far
M_AN=$(cd $WORK/analobench/emb && ls -d qwen06b qwen4b nemotron8b_qp abl2v2_* 2>/dev/null | while read m; do ls $m/m1024/L30_d.npy >/dev/null 2>&1 && echo $m; done)
$QWEN_PY -I $EV/bright/score.py --data $WORK/bright/data --out $WORK/bright --splits $SE > $L/score_bright.txt 2>&1 \
  && cp $WORK/bright/eval_full.{json,md} $R/bright/ || echo "bright scoring FAILED"
$QWEN_PY -I $EV/analobench/score.py --work $WORK/analobench --models $M_AN --out $R/analobench > $L/score_analobench.txt 2>&1 || echo "analobench scoring FAILED"
$QWEN_PY -I $EV/main/score.py --work $WORK/ours --registry $REGISTRY --out $R/ours --title "Dataset ablation: CPR-eval (max length 1024)" \
  > $L/score_ours.txt 2>&1 || echo "ours scoring FAILED"
$QWEN_PY -I $HERE/tables.py --eval $WORK --res-dir $R --out $R/RES.md
