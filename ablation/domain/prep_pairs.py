#!/usr/bin/env python3
"""Corpus ids and field of every training and eval pair (no texts): the input of fetch_openalex.py and build_domains.py.
Reads the id-only release: TRAIN_DIR/train.jsonl and EVAL_DIR/eval.jsonl (fields query_id, q_corpusid, positive.corpusid, field).
TRAIN_DIR / EVAL_DIR = --train-dir / --eval-dir, else the environment variables TRAIN_DIR / EVAL_DIR, else RELEASE_DIR/train, RELEASE_DIR/eval.
Writes OUT/train.jsonl and OUT/test.jsonl, one row per pair in release order: {query_id, q_cid, pos_cid, field} (query_id = the release id,
the key of evaluation/main/prepare.py's pools.json for the eval pairs).
  python -I prep_pairs.py [--train-dir DIR] [--eval-dir DIR] --out PAIRS"""
import argparse, json, os
E = lambda k, sub: os.environ.get(k) or (os.environ.get('RELEASE_DIR') and os.path.join(os.environ['RELEASE_DIR'], sub))
ap = argparse.ArgumentParser(); ap.add_argument('--train-dir', default=E('TRAIN_DIR', 'train')); ap.add_argument('--eval-dir', default=E('EVAL_DIR', 'eval'))
ap.add_argument('--out', required=True); A = ap.parse_args()
if not (A.train_dir and A.eval_dir): ap.error('give --train-dir / --eval-dir or set TRAIN_DIR / EVAL_DIR / RELEASE_DIR')
os.makedirs(A.out, exist_ok=True)
for sp, f in (('train', f'{A.train_dir}/train.jsonl'), ('test', f'{A.eval_dir}/eval.jsonl')):
    n = 0
    with open(f'{A.out}/{sp}.jsonl', 'w') as o:
        for l in open(f):
            r = json.loads(l); q, p = str(r['q_corpusid']), str(r['positive']['corpusid'])
            assert r['query_id'] == f'scaleup_{q}_{p}', r['query_id']
            o.write(json.dumps({'query_id': r['query_id'], 'q_cid': q, 'pos_cid': p, 'field': r['field']}) + '\n'); n += 1
    print(sp, n)
