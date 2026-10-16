#!/usr/bin/env python3
"""Release step 1: every paper of the eval and training sets (base paper q, positive d, negatives n) with its roles and its abstract
license ($DATA/license/abstracts, license_table.py) -> OUT_DIR/release_ids.json.
  python -I collect_ids.py EVAL.jsonl TRAIN.meta.jsonl OUT_DIR"""
import collections, glob, json, sys
from pathlib import Path
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env
E = [json.loads(l) for l in open(sys.argv[1])]; M = [json.loads(l) for l in open(sys.argv[2])]; OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
role = collections.defaultdict(set)
for e in E:
    role[int(e['query_id'].split('_')[-2])].add('q'); role[e['pos']['cid']].add('d')
    for n in e['negs']: role[n['cid']].add('n')
for m in M:
    role[int(m['q_cid'])].add('q'); role[m['pos_cid']].add('d')
    for s, c in m['neg']: role[c].add('n')
ids = sorted(role); lic = {}
for f in sorted(glob.glob(str(env('DATA') / 'license' / 'abstracts' / '*.parquet'))):
    t = pq.read_table(f); t = t.filter(pc.is_in(t.column('corpusid'), value_set=pa.array(ids, pa.int64())))
    for c, l in zip(t.column('corpusid').to_pylist(), t.column('license').to_pylist()): lic[c] = l
json.dump({'ids': ids, 'roles': {c: sorted(v) for c, v in role.items()}, 'license': {c: lic.get(c) for c in ids}}, open(OUT / 'release_ids.json', 'w'))
print('papers', len(ids))
