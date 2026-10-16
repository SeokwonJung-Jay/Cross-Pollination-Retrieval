#!/usr/bin/env python3
"""Export the leave-one-out pools in the eval_texts.jsonl format (dataset/release/fetch_abstracts.py): one file per left-out pooling
retriever, OUT/eval_loo_<r>.jsonl (r = e5, splade, scincl), same rows and order as the eval file, with the negatives replaced:
{query_id, query, field, pos: {cid, text}, negs: [{cid, stratum, text}]}; negatives = the first 50 usable ones per stratum
(q_side, pos_side, other) in judge order, text = title + '. ' + abstract[:2000] (the text the Stage 7 judge saw).
Pools come from build_pools.sh: LOO_WORK/<tag>/pools_loo_<r>.json, texts from LOO_WORK/<tag>/s7_texts.json (the run's Stage 7 text cache).
An eval query is matched to its pool by (query text, positive corpus id). Texts missing from the caches are read from the corpus parquet
shards given by --corpus (columns corpusid, title, abstract), same formatting.
  python -I export.py --eval EVAL_DIR/eval_texts.jsonl --loo-work LOO_WORK --out OUT [--corpus 'CORPUS/export/universe_*.parquet' ...]"""
import argparse, collections, glob, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--eval', required=True); ap.add_argument('--loo-work', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--corpus', nargs='*', default=[]); ap.add_argument('--per-stratum', type=int, default=50); a = ap.parse_args()
E = [json.loads(l) for l in open(a.eval)]; O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
tags = sorted(p for p in Path(a.loo_work).iterdir() if (p / 's7_texts.json').exists())
TX = {t.name: json.load(open(t / 's7_texts.json')) for t in tags}
FILL = {}
def fill(cids):
    need = sorted(set(cids) - set(FILL))
    if not need: return
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    for sh in sorted(f for g in a.corpus for f in glob.glob(g)):
        rows = np.flatnonzero(np.isin(pq.read_table(sh, columns=['corpusid']).column(0).to_numpy(), need))
        if not len(rows): continue
        t = pq.read_table(sh, columns=['corpusid', 'title', 'abstract']).take(pa.array(rows))
        for c, ti, ab in zip(t.column('corpusid').to_pylist(), t.column('title').to_pylist(), t.column('abstract').to_pylist()):
            FILL.setdefault(c, f"{ti}. {' '.join((ab or '').split())[:2000]}")
K = a.per_stratum
for r in ('e5', 'splade', 'scincl'):
    pool = {}   # (query, pos_cid) -> (tag, pool)
    for t in tags:
        f = t / f'pools_loo_{r}.json'
        if f.exists():
            for p, x in json.load(open(f)).items(): pool[(x['query'], int(x['pos_cid']))] = (t.name, x)
    rows = [(e, *pool[(e['query'], int(e['pos']['cid']))]) for e in E]
    fill([int(c) for e, t, x in rows for st in x['main'].values() for c in st[:K] if str(c) not in TX[t]])
    sizes = collections.Counter(); miss = 0
    with open(O / f'eval_loo_{r}.jsonl', 'w') as f:
        for e, t, x in rows:
            negs = [{'cid': int(c), 'stratum': st, 'text': TX[t].get(str(c)) or FILL.get(int(c))} for st in ('q_side', 'pos_side', 'other') for c in x['main'][st][:K]]
            miss += sum(n['text'] is None for n in negs); sizes[len(negs)] += 1
            f.write(json.dumps({**{k: e[k] for k in ('query_id', 'query', 'field', 'pos')}, 'negs': negs}, ensure_ascii=False) + '\n')
    print(f'without {r}: queries {len(rows)} | pool sizes {dict(sizes)} | texts missing {miss}')
