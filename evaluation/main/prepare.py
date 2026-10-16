#!/usr/bin/env python3
"""Our eval set (CPR-eval, 26 fields x 50 = 1,300 queries) from the id-only release: EVAL_DIR/eval_texts.jsonl, written by
dataset/release/fetch_abstracts.py from the release's eval.jsonl + papers.jsonl (fields query_id, query, field, pos {cid, text},
negs [{cid, stratum, text}]; text = "title. abstract", abstract cut at 2,000 characters). Each query's pool = positive + 150 negatives
(q_side 50, pos_side 50, other 50).
EVAL_DIR = --eval-dir, else the environment variable EVAL_DIR, else RELEASE_DIR/eval (RELEASE_DIR = the downloaded CPR-Dataset directory;
EVAL_DIR holds eval.jsonl and the fetch_abstracts.py output eval_texts.jsonl).
Writes WORK/_data/pools.json {query_id: {query, pos_cid, field, main: {q_side: [cid], pos_side: [...], other: [...]}}} (query_id = the
release id scaleup_<query corpus id>_<positive corpus id>), WORK/_data/texts.json {cid: text}, WORK/_data/source.json (file, sha256,
counts, documents whose fetched abstract does not match the release hash).
Checks: --n distinct queries, 150 distinct negatives per query (50 per type), positive not among its negatives, one text per corpus id.
  python -I prepare.py --work DIR [--eval-dir DIR] [--n N]"""
import argparse, collections, hashlib, json, os
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True)
ap.add_argument('--eval-dir', default=os.environ.get('EVAL_DIR') or (os.environ.get('RELEASE_DIR') and os.path.join(os.environ['RELEASE_DIR'], 'eval')),
                help='default: $EVAL_DIR, else $RELEASE_DIR/eval')
ap.add_argument('--n', type=int, default=1300); a = ap.parse_args()
if not a.eval_dir: ap.error('give --eval-dir or set EVAL_DIR (the directory with eval_texts.jsonl) / RELEASE_DIR (the CPR-Dataset directory)')
F = Path(a.eval_dir) / 'eval_texts.jsonl'
assert F.exists(), f'{F} not found: run dataset/release/fetch_abstracts.py on the release first'
D = Path(a.work) / '_data'; D.mkdir(parents=True, exist_ok=True)
R = [json.loads(l) for l in open(F)]
assert len(R) == a.n and len({r['query_id'] for r in R}) == a.n, len(R)
R = [{**r, 'pos': [r['pos']['text']], 'pos_cid': r['pos']['cid'], 'neg': [n['text'] for n in r['negs']], 'neg_cids': [n['cid'] for n in r['negs']],
      'neg_types': [n['stratum'] for n in r['negs']]} for r in R]
T, pools = {}, {}
def put(c, t):
    c = str(c); assert T.setdefault(c, t) == t, f'two texts for {c}'
for r in R:
    assert len(r['pos']) == 1 and len(r['neg']) == len(r['neg_cids']) == len(r['neg_types']) == 150 and len(set(r['neg_cids'])) == 150, r['query_id']
    assert collections.Counter(r['neg_types']) == {'q_side': 50, 'pos_side': 50, 'other': 50}, r['query_id']
    assert r['pos_cid'] not in r['neg_cids'], r['query_id']
    put(r['pos_cid'], r['pos'][0]); main = {'q_side': [], 'pos_side': [], 'other': []}
    for c, t, ty in zip(r['neg_cids'], r['neg'], r['neg_types']): put(c, t); main[ty].append(str(c))
    pools[r['query_id']] = {'query': r['query'], 'pos_cid': str(r['pos_cid']), 'field': r['field'], 'main': main}
json.dump(pools, open(D / 'pools.json', 'w')); json.dump(T, open(D / 'texts.json', 'w'))
mm = Path(a.eval_dir) / 'hash_mismatch.json'; bad = set(map(str, json.load(open(mm)))) & set(T) if mm.exists() else set()
src = {'file': F.name, 'sha256': hashlib.sha256(F.read_bytes()).hexdigest(), 'queries': len(pools), 'docs': len(T),
       'fields': len({p['field'] for p in pools.values()}), 'docs_hash_mismatch': len(bad)}
json.dump(src, open(D / 'source.json', 'w'), indent=1); print(src)
if bad: print(f'warning: {len(bad)} documents have an abstract that differs from the release hash (hash_mismatch.json of fetch_abstracts.py)')
