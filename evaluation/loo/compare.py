#!/usr/bin/env python3
"""Leave-one-out pooling comparison (paper Section 4.3, Figure robust (b)).
Reads the per-query ranks written by evaluation/main/score.py (`per_query.json`, {variant: {model: {query_id: rank}}}) for the original
pools and for the three leave-one-out pools, and reports per retriever Recall@10 (mean over all queries, main pool of 151 documents) in each
pool, its change from the original pool, the largest absolute change, and Kendall's tau-b between the retriever order on the original pool
and on each leave-one-out pool (tau-b over the unrounded Recall@10 values).
Models = those present in all four pools (or --models). Query ids must agree between the original and the leave-one-out runs.
  python -I compare.py --orig ORIG_RESULTS/per_query.json --loo LOO_ROOT [--metric-k 10] [--models m1,m2,...] [--out compare.json]
LOO_ROOT/<pool>/results/per_query.json for pool in loo_e5, loo_splade, loo_scincl (split.py output)."""
import argparse, itertools, json, math
from pathlib import Path
POOLS = ('loo_e5', 'loo_splade', 'loo_scincl')
ap = argparse.ArgumentParser(); ap.add_argument('--orig', required=True); ap.add_argument('--loo', required=True)
ap.add_argument('--metric-k', type=int, default=10); ap.add_argument('--models'); ap.add_argument('--variant', default='main'); ap.add_argument('--out')
a = ap.parse_args()
R = {'orig': json.load(open(a.orig))[a.variant]} | {p: json.load(open(Path(a.loo) / p / 'results' / 'per_query.json'))[a.variant] for p in POOLS}
models = a.models.split(',') if a.models else [m for m in R['orig'] if all(m in R[p] for p in POOLS)]
for p in POOLS:
    for m in models: assert set(R[p][m]) == set(R['orig'][m]), f'{p} {m}: query ids differ from the original run'
rec = {p: {m: sum(r <= a.metric_k for r in R[p][m].values()) / len(R[p][m]) for m in models} for p in R}

def tau_b(x, y):
    c = d = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0: continue
        if dx == 0: tx += 1
        elif dy == 0: ty += 1
        elif dx * dy > 0: c += 1
        else: d += 1
    return (c - d) / math.sqrt((c + d + tx) * (c + d + ty))

k = f'R@{a.metric_k}'
out = {'metric': k, 'models': models, 'recall': rec,
       'change': {p: {m: rec[p][m] - rec['orig'][m] for m in models} for p in POOLS},
       'kendall_tau_b_vs_orig': {p: tau_b([rec['orig'][m] for m in models], [rec[p][m] for m in models]) for p in POOLS}}
mx = max(((abs(v), p, m) for p in POOLS for m, v in out['change'][p].items()))
out['max_abs_change'] = {'value': mx[0], 'pool': mx[1], 'model': mx[2]}
print(f"| model | orig {k} | " + ' | '.join(POOLS) + ' |'); print('|---' * (2 + len(POOLS)) + '|')
for m in models:
    print(f"| {m} | {rec['orig'][m]:.3f} | " + ' | '.join(f"{rec[p][m]:.3f} ({rec[p][m] - rec['orig'][m]:+.3f})" for p in POOLS) + ' |')
print(f"largest |change| {mx[0]:.4f} ({mx[2]}, {mx[1]}); Kendall tau-b vs original: " + ', '.join(f"{p} {v:.3f}" for p, v in out['kendall_tau_b_vs_orig'].items()))
if a.out: json.dump(out, open(a.out, 'w'), indent=1)
