#!/usr/bin/env python3
"""Leave-one-out evaluation input: the three leave-one-out eval files (export.py) as ONE evaluation/main work dir, so every model encodes each
document once (union of the three pools) and scores 3 x 1,300 pseudo-queries, each against its own pool.
Query ids become '<query_id>@loo_<pool>' (pool = e5 / splade / scincl). One text per corpus id: the text of the original eval set
(eval_texts.jsonl) when the id is there, else the first text seen (pool order e5, splade, scincl).
Writes WORK/_data/{pools.json, texts.json, source.json} in the format of evaluation/main/prepare.py (same checks: 150 distinct negatives per
query, 50 per type, positive not among its negatives). split.py later splits the scores per pool.
  python -I prepare.py --work WORK --loo-dir DIR [--eval EVAL_DIR/eval_texts.jsonl]
DIR/eval_loo_<pool>.jsonl from export.py. --eval defaults to eval_texts.jsonl in $EVAL_DIR, else $RELEASE_DIR/eval (the CPR-eval release dir
after dataset/release/fetch_abstracts.py)."""
import argparse, collections, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; POOLS = ('e5', 'splade', 'scincl')
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--loo-dir', required=True); ap.add_argument('--eval')
ap.add_argument('--n', type=int, default=1300); a = ap.parse_args()
d = os.environ.get('EVAL_DIR') or (os.environ.get('RELEASE_DIR') and os.path.join(os.environ['RELEASE_DIR'], 'eval')); path = a.eval or (d and str(Path(d) / 'eval_texts.jsonl'))
if not path: ap.error('give --eval or set EVAL_DIR / RELEASE_DIR')
T = {}
for l in open(path):
    e = json.loads(l)
    for x in [e['pos']] + e['negs']: T.setdefault(str(x['cid']), x['text'])
n_orig_docs = len(T); pools = {}; replaced = collections.Counter()
for p in POOLS:
    R = [json.loads(l) for l in open(Path(a.loo_dir) / f'eval_loo_{p}.jsonl')]
    assert len(R) == a.n and len({r['query_id'] for r in R}) == a.n, (p, len(R))
    for r in R:
        cids = [str(n['cid']) for n in r['negs']]; pos = str(r['pos']['cid'])
        assert len(cids) == len(set(cids)) == 150 and pos not in cids, (p, r['query_id'])
        assert collections.Counter(n['stratum'] for n in r['negs']) == {'q_side': 50, 'pos_side': 50, 'other': 50}, (p, r['query_id'])
        for x in [r['pos']] + r['negs']:
            t = T.setdefault(str(x['cid']), x['text']); replaced[p] += t != x['text']
        main = {'q_side': [], 'pos_side': [], 'other': []}
        for n in r['negs']: main[n['stratum']].append(str(n['cid']))
        pools[f"{r['query_id']}@loo_{p}"] = {'query': r['query'], 'pos_cid': pos, 'field': r['field'], 'main': main}
need = {c for x in pools.values() for c in [x['pos_cid']] + [c for s in x['main'].values() for c in s]}
T = {c: T[c] for c in sorted(need)}
D = Path(a.work) / '_data'; D.mkdir(parents=True, exist_ok=True)
json.dump(pools, open(D / 'pools.json', 'w')); json.dump(T, open(D / 'texts.json', 'w'))
src = {'file': str(Path(path).resolve()), 'loo_dir': str(Path(a.loo_dir).resolve()),
       'pools': [f'loo_{p}' for p in POOLS], 'queries': len(pools), 'docs': len(T), 'fields': len({x['field'] for x in pools.values()}),
       'texts_replaced_by_original': dict(replaced)}
json.dump(src, open(D / 'source.json', 'w'), indent=1); print(src)
