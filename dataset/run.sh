#!/bin/bash
# One CPR-Dataset run (config/eval.json or config/train.json) after the corpus steps (README "Corpus"):
#   bash run.sh CONFIG [STEP ...]      STEP in: prep stage2 s3 s4 s5 s5b s7b select pool s7 export   (default: all, in this order)
# select runs only for a config with queries_per_field (eval). export writes $WORK/<name>.jsonl (eval: export_eval.py; train:
# export_train.py, excluding papers of $WORK/eval.jsonl when it exists). Every step is resumable; LLM steps stop at their config caps.
# Env: S2AG S2ORC OPENALEX DATA WORK, OPENAI_API_KEY, optional S2_API_KEY (or a .env file); PY (python, default python3);
# QN (query shards per pooling retriever, default 6).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); CF=$(realpath "$1"); shift; PY=${PY:-python3}; QN=${QN:-6}
STEPS=${*:-"prep stage2 s3 s4 s5 s5b s7b select pool s7 export"}
cfg() { $PY -I -c "import sys; sys.path.insert(0, '$HERE'); from common import load_config; c = load_config('$CF'); print(c.get('$1') or '')"; }
NAME=$(cfg name); WD=$(cfg work); PER=$(cfg queries_per_field)
for s in $STEPS; do
  echo "== $s $(date -u +%FT%TZ)"
  case $s in
    prep)   $PY -I $HERE/positives/stage2_input.py $CF
            $PY -I $HERE/positives/openalex_tags.py $CF
            $PY -I $HERE/positives/pmcid_lookup.py $CF
            $PY -I $HERE/positives/jats_prefetch.py $CF
            $PY -I $HERE/positives/s2orc_extract.py $CF ;;
    stage2) for c in plan run merge; do $PY -I $HERE/positives/stage2_parallel.py $CF $c; done ;;
    s3)     $PY -I $HERE/positives/stage3_analogy.py $CF ;;
    s4)     $PY -I $HERE/positives/stage4_transfer.py $CF ;;
    s5)     $PY -I $HERE/positives/stage5_query.py $CF ;;
    s5b|s7b) $PY -I $HERE/positives/dedup_usefulness.py $CF $s ;;
    select) [ -n "$PER" ] && $PY -I $HERE/export/select_eval.py $CF || echo "no queries_per_field: skipped" ;;
    pool)   PAIRS=$WD/final_pairs.json; [ -n "$PER" ] && PAIRS=$WD/eval_pairs.json
            $PY -I $HERE/negatives/pool_universe.py spec $CF --pairs $PAIRS
            for m in e5 splade scincl; do
              for i in $(seq 0 $((QN - 1))); do QSHARD=$i/$QN THREADS=4 $PY -I $HERE/negatives/pool_universe.py score $CF $m > $WD/s6/score_${m}_$i.log 2>&1 & done
            done; wait
            for m in e5 splade scincl; do $PY -I $HERE/negatives/pool_universe.py join $CF $m; done
            $PY -I $HERE/negatives/pool_universe.py merge $CF ;;
    s7)     $PY -I $HERE/negatives/judge_negatives.py $CF ;;
    export) if [ -n "$PER" ]; then $PY -I $HERE/export/export_eval.py $CF $(dirname $WD)/$NAME.jsonl
            else EX=$(dirname $WD)/eval.jsonl; $PY -I $HERE/export/export_train.py $CF $(dirname $WD)/$NAME.jsonl --neg 9 $([ -f $EX ] && echo --exclude-eval $EX); fi ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
