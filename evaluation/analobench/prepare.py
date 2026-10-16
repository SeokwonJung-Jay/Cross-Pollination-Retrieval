#!/usr/bin/env python3
"""AnaloBench (HF jhu-clsp/AnaloBench, pinned REV) -> one JSON per story length. Every task of a length draws its texts from one pool of 340
stories (checked here), so each length is
  WORK/data/L<n>.json  {"stories": [340 texts], "t2": [{"q", "bank", "gold"}], "t1_full": [{"q", "opts", "gold"}], "t1_sub": [...]}
with story indices into "stories". T2: query story vs its own 200-story bank (Options, numbered 1..200), gold = Indices (1-based into the bank).
T1: query vs 4 options (A-D), gold = Label. Kept as published: 5 T1-Subset rows (152, 163, 168, 188, 192, every length) list the query
story itself as a wrong option (never in T1-Full); counted in "t1_sub_self". The S1 files have no Story column: the query is Sentence.
Raw CSVs: WORK/raw.
  python -I prepare.py --work WORK"""
import argparse, json, re
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download
REV = '58c09414e35f38dd010e7cc87d3e2a5d24f151cc'   # jhu-clsp/AnaloBench main
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); a = ap.parse_args()
W = Path(a.work); raw = W / 'raw'; (W / 'data').mkdir(parents=True, exist_ok=True)
snapshot_download('jhu-clsp/AnaloBench', repo_type='dataset', revision=REV, local_dir=raw, allow_patterns=['*.csv'])
num = lambda o: [x.strip() for x in re.split(r'(?:^|\n)\d+\. ', o)[1:]]
abc = lambda o: [x.strip() for x in re.split(r'(?:^|\n)[A-D]\. ', o)[1:]]
qcol = lambda df: (df['Story'] if 'Story' in df else df['Sentence']).str.strip()
for L in (1, 10, 30):
    t2 = pd.read_csv(raw / f'AnaloBench-T2-S{L}.csv'); banks = [num(o) for o in t2['Options']]
    stories = sorted({x for b in banks for x in b}); ix = {s: i for i, s in enumerate(stories)}; assert len(stories) == 340, len(stories)
    T2 = []
    for q, b, g in zip(qcol(t2), banks, t2['Indices']):
        g = [int(x) for x in str(g).split(',')]; assert len(b) == 200 and q in ix and q not in b and len(set(g)) == len(g) and all(1 <= x <= 200 for x in g)
        T2.append({'q': ix[q], 'bank': [ix[x] for x in b], 'gold': [ix[b[x - 1]] for x in g]})
    out = {'stories': stories, 't2': T2}
    for k, f in (('t1_full', 'Full'), ('t1_sub', 'Subset')):
        t1 = pd.read_csv(raw / f'AnaloBench-T1-{f}-S{L}.csv'); R = []; self_ = 0
        for q, o, lab in zip(qcol(t1), t1['Options'], t1['Label']):
            o = abc(o); assert len(o) == 4 and q in ix and all(x in ix for x in o) and o['ABCD'.index(lab.strip())] != q
            self_ += q in o
            R.append({'q': ix[q], 'opts': [ix[x] for x in o], 'gold': ix[o['ABCD'.index(lab.strip())]]})
        out[k] = R; assert self_ == (5 if k == 't1_sub' else 0), self_
    out['t1_sub_self'] = 5
    json.dump(out, open(W / 'data' / f'L{L}.json', 'w'))
    print(f'S{L}: stories 340, t2 {len(T2)} (gold/query {sum(len(r["gold"]) for r in T2) / len(T2):.2f}), t1_full {len(out["t1_full"])}, t1_sub {len(out["t1_sub"])}')
