#!/bin/bash
# Leave-one-out pools for one eval pooling run (paper Section 4.3): for each pooling retriever left out (e5, splade, scincl), rebuild the
# candidate pools of the eval queries from the other two retrievers and judge them as in the dataset pipeline (dataset/negatives: pool_universe.py merge + judge_negatives.py).
#   depth 300   : merge the other two retrievers' top-300 lists per stratum (dataset/negatives/pool_universe.py merge, round-robin by rank,
#                 the base paper's other cited papers first as in Stage 7), judge with dataset/negatives/judge_negatives.py (same prompt / model / Batch API,
#                 50 negatives per stratum). Every judgment of the source run is reused (its s7/ directory is shared), so only candidates
#                 that were never judged cost an API call.
#   deeper K    : queries with a stratum still short of 50 in any of the three pools are re-scored with all three retrievers at POOL_K=K
#                 (pool_universe.py score), merged without the left-out retriever at POOL_K=K and judged again (DEPTHS, default 300 1000 3000).
#   final       : LOO_WORK/<TAG>/pools_loo_<r>.json = per query the pool of the deepest depth that pooled it.
# Run once per pooling run of the eval set (the eval queries of 2010-19 were pooled on the <=2019 universe, those of 2020-25 on the
# <=2025 universe; CORPUS / VECTORS = the corpus and vector dirs of that run, as in the dataset config keys `corpus` / `vectors`):
#   SRC=/path/eval_run_work TAG=eval2025 CORPUS=/path/corpus_2025 VECTORS=/path/vectors_2025 BASE_CONFIG=/path/eval_config.json EVAL=/path/eval_release/eval_texts.jsonl LOO_WORK=/path/loo bash build_pools.sh
# SRC = work dir of the dataset pipeline's eval run (s6/{queries.json, top300_<m>.json}, s7/ judgments, s7_texts.json). The run writes its new
# judgments, pool and short files into SRC/s7/ (shared judgment directory) and its own files into LOO_WORK/<TAG>/.
# Env: CAP (USD per Stage 7 run, default 200), PY (python with the dataset/ requirements), THREADS / QSHARDS for scoring. Keys: OPENAI_API_KEY.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); DS=$REPO/dataset
: ${SRC:?} ${TAG:?} ${CORPUS:?} ${VECTORS:?} ${BASE_CONFIG:?} ${EVAL:?} ${LOO_WORK:?}
PY=${PY:-python}; CAP=${CAP:-200}; DEPTHS=${DEPTHS:-"300 1000 3000"}; QSHARDS=${QSHARDS:-1}
ALL="scincl splade e5"   # merge order of pool_universe.py (round-robin over retrievers by rank)
T=$LOO_WORK/$TAG; mkdir -p "$T"; ln -sfn "$SRC/s7_texts.json" "$T/s7_texts.json"
others() { local o=""; for m in $ALL; do if [ $m != $1 ]; then o="$o${o:+,}$m"; fi; done; echo $o; }
prev=""
for K in $DEPTHS; do
  D=$T/k$K; mkdir -p "$D/s6"
  if [ -z "$prev" ]; then   # first depth: the source run's own top-K lists
    $PY -I "$HERE/pools.py" init --src "$SRC" --eval "$EVAL" --out "$D/s6/queries.json"
    for m in $ALL; do ln -sfn "$SRC/s6/top300_$m.json" "$D/s6/top300_$m.json"; done
  else                      # deeper: only the queries still short, re-scored at POOL_K=K
    $PY -I "$HERE/pools.py" short --runs $(for r in $ALL; do echo "$T/k${prev}_$r"; done) --queries "$T/k$prev/s6/queries.json" --out "$D/s6/queries.json"
    if [ "$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$D/s6/queries.json")" = 0 ]; then break; fi
    $PY -I "$HERE/pools.py" config --base "$BASE_CONFIG" --work "$D" --name "loo_${TAG}_k${K}_score" --corpus "$CORPUS" --vectors "$VECTORS" --tag "$TAG" --left-out none --depth $K --cap 0 --out "$D/config.json"
    for m in $ALL; do
      for i in $(seq 0 $((QSHARDS - 1))); do POOL_K=$K QSHARD=$i/$QSHARDS $PY -I "$DS/negatives/pool_universe.py" score "$D/config.json" $m & done; wait
      if [ "$QSHARDS" -gt 1 ]; then $PY -I "$DS/negatives/pool_universe.py" join "$D/config.json" $m; fi
    done
  fi
  for r in $ALL; do   # one Stage 7 run at a time: they share the judgment directory and the text cache
    W=$T/k${K}_$r; mkdir -p "$W/s6"; cp "$D/s6/queries.json" "$W/s6/"
    for m in $ALL; do if [ $m != $r ]; then ln -sfn "$(readlink -f "$D/s6/top300_$m.json")" "$W/s6/top300_$m.json"; fi; done
    ln -sfn "$SRC/s7" "$W/s7"; ln -sfn "$SRC/s7_texts.json" "$W/s7_texts.json"
    $PY -I "$HERE/pools.py" config --base "$BASE_CONFIG" --work "$W" --name "loo_${TAG}_k${K}_$r" --corpus "$CORPUS" --vectors "$VECTORS" --tag "$TAG" --left-out $r --depth $K --cap $CAP --out "$W/config.json"
    POOL_K=$K $PY -I "$DS/negatives/pool_universe.py" merge "$W/config.json" --retrievers $(others $r)
    $PY -I "$DS/negatives/judge_negatives.py" "$W/config.json"
  done
  prev=$K
done
for r in $ALL; do $PY -I "$HERE/pools.py" final --tag-dir "$T" --left-out $r --depths "$DEPTHS"; done
