#!/usr/bin/env python3
"""Eval query selection before pooling: per OpenAlex field of the base paper (the field it was sampled under in config q_pool), the
final pairs (<work>/final_pairs.json, Stage 7b u >= 2) in base-paper pool order, the first queries_per_field (config, 50) plus
--spares more per field (pooled and judged too, to replace queries whose groups cannot be filled with 50 negatives).
Writes <work>/eval_pairs.json [pair_id] (input of pool_universe.py spec --pairs) and <work>/eval_selection.json {field: {selected, spares}}.
  python -I select_eval.py CONFIG [--spares 10]"""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, base_papers, records
ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--spares', type=int, default=10); A = ap.parse_args()
C = load_config(A.config); W = Path(C['work']); N = C['queries_per_field']
pool = base_papers(C); pos = {c: (f, i) for f, v in pool.items() for i, c in enumerate(v)}
S3 = records(W / 's3'); fin = json.load(open(W / 'final_pairs.json')); by = collections.defaultdict(list)
for p in fin:
    q = int(S3[p]['source']['base_id'])
    if q in pos: by[pos[q][0]].append(p)
sel = {f: sorted(by[f], key=lambda p: (pos[int(S3[p]['source']['base_id'])][1], p)) for f in pool}
out = {f: {'selected': v[:N], 'spares': v[N:N + A.spares]} for f, v in sel.items()}
json.dump(out, open(W / 'eval_selection.json', 'w'), indent=1)
json.dump(sorted(p for v in out.values() for p in v['selected'] + v['spares']), open(W / 'eval_pairs.json', 'w'), indent=1)
print('fields', len(out), '| selected', sum(len(v['selected']) for v in out.values()), '| short fields', {f: len(v['selected']) for f, v in out.items() if len(v['selected']) < N})
