#!/usr/bin/env python3
"""Metrics of the LLM reranker from the rank of the positive per query, with the conventions of evaluation/main/score.py:
R@1/5/10, MRR@10, nDCG@10 over all queries of WORK/_data/pools.json (a query missing from the ranks file counts as a miss, rank 152);
per OpenAlex domain of the query field (evaluation/main/field_domain.json) and Avg = mean over all queries; 95% CI = bootstrap over queries
(2,000 resamples, random.Random(0), sorted index 50 / 1949), stratified within each domain for the domain table.
--ranks: {query_id: rank} (rerank.py results/ranks.json) or a WORK/<name>/results/scores.json of rerank.py (ranked like main/score.py).
Writes OUT/metrics.json and prints the table.
  python -I score.py --work WORK --ranks WORK/llm_reranker_mimo/results/ranks.json [--out WORK/llm_reranker_mimo/results]"""
import argparse, json, math, random, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--ranks', required=True); ap.add_argument('--out')
ap.add_argument('--field-domain', default=str(HERE.parent / 'main' / 'field_domain.json')); a = ap.parse_args()
pools = json.load(open(Path(a.work) / '_data' / 'pools.json')); qids = sorted(pools)
P = {q: [x['pos_cid']] + [c for s in ('q_side', 'pos_side', 'other') for c in x['main'][s]] for q, x in pools.items()}; N = len(P[qids[0]])
X = json.load(open(a.ranks)); first = next(iter(X.values()))
R = {q: (1 + sum(X[q][d] >= X[q][P[q][0]] for d in P[q][1:]) if isinstance(first, dict) else X[q]) if q in X else N + 1 for q in qids}
extra = sorted(set(X) - set(qids)); assert not extra, f'{len(extra)} ids not in the eval set, e.g. {extra[:3]}'
FD = json.load(open(a.field_domain)); F2D = {f: d.split()[0] for d, fs in FD.items() if not d.startswith('_') for f in fs}
DOMS = ['Health', 'Life', 'Physical', 'Social']; DQ = {d: [q for q in qids if F2D[pools[q]['field']] == d] for d in DOMS}
KS = ('R@1', 'R@5', 'R@10', 'MRR@10', 'nDCG@10')
def met(rs): return {'R@1': statistics.mean(r <= 1 for r in rs), 'R@5': statistics.mean(r <= 5 for r in rs), 'R@10': statistics.mean(r <= 10 for r in rs),
                     'MRR@10': statistics.mean(1 / r if r <= 10 else 0 for r in rs), 'nDCG@10': statistics.mean(1 / math.log2(r + 1) if r <= 10 else 0 for r in rs)}
ci = lambda pt, bs: {k: [round(pt[k], 4), round(sorted(x[k] for x in bs)[50], 4), round(sorted(x[k] for x in bs)[1949], 4)] for k in KS}
rng = random.Random(0); B = [[rng.choice(qids) for _ in qids] for _ in range(2000)]
out = {'n': len(qids), 'pool_size': N, 'missing': len(qids) - sum(q in X for q in qids), 'all': ci(met(list(R.values())), [met([R[q] for q in b]) for b in B])}
rng = random.Random(0); BD = [{d: [rng.choice(DQ[d]) for _ in DQ[d]] for d in DOMS} for _ in range(2000)]
bs = [{**{d: met([R[q] for q in b[d]]) for d in DOMS}, 'Avg': met([R[q] for d in DOMS for q in b[d]])} for b in BD]
pt = {**{d: met([R[q] for q in DQ[d]]) for d in DOMS}, 'Avg': met(list(R.values()))}
out['by_domain'] = {d: ci(pt[d], [x[d] for x in bs]) for d in DOMS + ['Avg']}; out['n_by_domain'] = {d: len(DQ[d]) for d in DOMS}
if a.out: Path(a.out).mkdir(parents=True, exist_ok=True); json.dump(out, open(Path(a.out) / 'metrics.json', 'w'), indent=1)
print(f"{out['n']} queries (missing {out['missing']}), pool {N} | " + ' | '.join(f"{k} {out['all'][k][0]:.4f} [{out['all'][k][1]:.3f}, {out['all'][k][2]:.3f}]" for k in KS))
for d in DOMS + ['Avg']: print(f"{d:9s} n {len(DQ[d]) if d in DQ else len(qids):4d} | " + ' | '.join(f"{k} {out['by_domain'][d][k][0]:.4f}" for k in KS))
