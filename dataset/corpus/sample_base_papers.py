#!/usr/bin/env python3
"""Base-paper (q) pool: per OpenAlex field, a random order of the eligible papers.
Eligible q (universe_f.parquet of build_universe.py): a seed paper (top-20% venue of its field, abstract, langid English, publication type
JournalArticle / Conference / Review or none), year in [--ymin, --ymax], a non-empty s2orc_v2 body and Semantic Scholar
referencecount >= 15 (s2_index.py), field = OpenAlex field of its rank-1 topic. Papers in --exclude pools are left out. Per field the
eligible q are shuffled with --seed and the first --per-field are kept; the pipeline processes them in this order.
Output {per_field: {field: [corpusid, ...]}, ...}.
Paper runs: 2010-2019, 800 per field, seed 20261005 on the <=2019 universe (the 2010-19 eval queries); 2010-2025, seed 20261006 on
the <=2025 universe, excluding the first pool (training queries and the 2020-25 eval queries).
  python -I sample_base_papers.py --universe $DATA/universe_2025 --ymin 2010 --ymax 2025 --per-field 1600 --seed 20261006 \
      --out $DATA/base_papers/pool_2010_2025.json [--exclude POOL.json ...]"""
import argparse, collections, json, random, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env
ap = argparse.ArgumentParser(); ap.add_argument('--universe', required=True); ap.add_argument('--ymin', type=int, default=2010); ap.add_argument('--ymax', type=int, default=2025)
ap.add_argument('--per-field', type=int, required=True); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--min-refs', type=int, default=15)
ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--out', required=True); A = ap.parse_args()
IDX = env('DATA') / 's2index'
old = collections.defaultdict(set)
for f in A.exclude:
    for fl, v in json.load(open(f))['per_field'].items(): old[fl] |= {int(c) for c in v}
t = pq.read_table(Path(A.universe) / 'universe_f.parquet', columns=['corpusid', 'year', 'is_seed', 'field'])
cid, yr, sd = t.column('corpusid').to_numpy(), t.column('year').to_numpy(), t.column('is_seed').to_numpy(zero_copy_only=False)
fl = np.array(t.column('field').to_pylist(), dtype=object)
m = sd & (yr >= A.ymin) & (yr <= A.ymax); cid, fl, yr = cid[m], fl[m], yr[m]
s2 = np.unique(np.concatenate([np.load(f) for f in (IDX / 's2orc_ids').glob('*.npy')]))
rc = np.concatenate([np.load(f) for f in (IDX / 'refcount').glob('*.npy')]); rc = rc[rc[:, 1] >= A.min_refs, 0]
keep = np.isin(cid, s2) & np.isin(cid, rc)
print(f'seed {A.ymin}-{A.ymax} {len(cid):,} | eligible {keep.sum():,}', flush=True)
by = collections.defaultdict(list)
for c, f in zip(cid[keep].tolist(), fl[keep].tolist()):
    if f and c not in old[f]: by[f].append(c)
rng = random.Random(A.seed); out = {f: rng.sample(sorted(v), min(A.per_field, len(v))) for f, v in sorted(by.items())}
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({'years': [A.ymin, A.ymax], 'seed': A.seed, 'per_field_n': A.per_field, 'excluded': A.exclude, 'per_field': out,
           'eligible_per_field': {f: len(v) for f, v in by.items()}}, open(A.out, 'w'), indent=1)
print('fields', len(out), '| q', sum(map(len, out.values())), '| min per field', min(map(len, out.values())))
