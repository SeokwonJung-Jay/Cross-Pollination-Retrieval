#!/usr/bin/env python3
"""Metrics for bright/encode.py runs. Per split: nDCG@10 (binary gains on gold_ids, as BRIGHT's pytrec_eval), R@1/5/10 (any gold in the top k),
MRR@10 (first gold), query bootstrap 95% CI (2,000 resamples, random.Random(0)), exact random baseline (N = corpus size minus the query's
excluded_ids). A gold outside the saved top 1000 counts as rank 1001. Macro = mean over splits. Output OUT/eval_full.json and OUT/eval_full.md.
Default: all queries of each split. Optional --sample N --sample_splits biology economics ... scores random.Random(20261007).sample(sorted query ids, N)
of each listed split instead (all queries if <= N); retrieval still searches the whole corpus. Sampled ids go to OUT/sample_<split>.json.
Per-query scores: OUT/per_query_<split>.json {model: {query id: {nDCG@10, R@1, R@5, R@10, MRR@10, gold_ranks}}} (1001 = outside the top 1000).
  python3 bright/score.py --data DIR --out DIR --splits biology ... [--sample N --sample_splits ...]"""
import sys, json, math, random, statistics, argparse
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common')); from random_baseline import random_metrics
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--splits', nargs='+', required=True)
ap.add_argument('--sample', type=int); ap.add_argument('--sample_splits', nargs='*', default=[]); ap.add_argument('--seed', type=int, default=20261007); a = ap.parse_args()
O = Path(a.out); ORDER = ['qwen06b_qp', 'qwen4b_qp', 'nemotron8b_qp']   # vanilla (encode.py)
ORDER += sorted({p.name for p in O.glob('abl2v2_*')})   # dataset ablation final checkpoints
models = [m for m in ORDER if all((O / m / s / 'run.json').exists() for s in a.splits)]
K = ('nDCG@10', 'R@1', 'R@5', 'R@10', 'MRR@10')
def ndcg10(rs):
    return sum(1 / math.log2(r + 1) for r in rs if r <= 10) / sum(1 / math.log2(i + 2) for i in range(min(len(rs), 10)))
def met(gr):
    f = [g[0] for g in gr]
    return {'nDCG@10': statistics.mean(ndcg10(g) for g in gr), 'R@1': statistics.mean(r <= 1 for r in f), 'R@5': statistics.mean(r <= 5 for r in f),
            'R@10': statistics.mean(r <= 10 for r in f), 'MRR@10': statistics.mean(1 / r if r <= 10 else 0 for r in f)}
out = {'splits': {}, 'macro': {}}
for s in a.splits:
    ex = pd.read_parquet(Path(a.data) / f'examples_{s}.parquet'); nd = len(pd.read_parquet(Path(a.data) / f'documents_{s}.parquet', columns=['id']))
    G = {str(i): {str(x) for x in g} for i, g in zip(ex['id'], ex['gold_ids'])}
    NX = {str(i): nd - len({str(x) for x in e} - G[str(i)]) for i, e in zip(ex['id'], ex['excluded_ids'])}
    qids = sorted(G)
    if a.sample and s in a.sample_splits and len(qids) > a.sample: qids = sorted(random.Random(a.seed).sample(qids, a.sample))
    if s in a.sample_splits: json.dump(qids, open(O / f'sample_{s}.json', 'w'))
    rng = random.Random(0); Bs = [[rng.choice(qids) for _ in qids] for _ in range(2000)]
    rnd = {k: [] for k in K}
    for q in qids:
        N, g = NX[q], len(G[q]); rm = random_metrics(N, g)
        rnd['R@1'].append(rm['recall1']); rnd['R@5'].append(rm['recall5']); rnd['R@10'].append(rm['recall10'])
        rnd['MRR@10'].append(sum(math.comb(N - r, g - 1) / math.comb(N, g) / r for r in range(1, 11)))
        rnd['nDCG@10'].append(g / N * sum(1 / math.log2(r + 1) for r in range(1, 11)) / sum(1 / math.log2(i + 2) for i in range(min(g, 10))))
    S = {'n': len(qids), 'corpus': nd, 'gold_mean': round(statistics.mean(len(G[q]) for q in qids), 2), 'random': {k: round(statistics.mean(v), 4) for k, v in rnd.items()}}
    PQ = {}
    for m in models:
        run = json.load(open(O / m / s / 'run.json'))
        R = {q: sorted([i + 1 for i, (d, _) in enumerate(run[q]) if d in G[q]] + [1001] * len(G[q] - {d for d, _ in run[q]})) for q in qids}
        PQ[m] = {q: {**{k: round(v, 4) for k, v in met([R[q]]).items()}, 'gold_ranks': R[q]} for q in qids}
        pt = met([R[q] for q in qids]); bs = [met([R[q] for q in b]) for b in Bs]
        S[m] = {k: [round(pt[k], 3), round(sorted(x[k] for x in bs)[50], 3), round(sorted(x[k] for x in bs)[1949], 3)] for k in K}
    out['splits'][s] = S; json.dump(PQ, open(O / f'per_query_{s}.json', 'w'))
for m in models: out['macro'][m] = {k: round(statistics.mean(out['splits'][s][m][k][0] for s in a.splits), 3) for k in K}
out['macro']['random'] = {k: round(statistics.mean(out['splits'][s]['random'][k] for s in a.splits), 4) for k in K}
json.dump(out, open(O / 'eval_full.json', 'w'), indent=1)
L = [f"## nDCG@10 (official BRIGHT metric), per split + macro mean\n", '| Model | ' + ' | '.join(a.splits) + ' | macro |', '|---|' + '---|' * (len(a.splits) + 1)]
bs_ = {s: max(out['splits'][s][m]['nDCG@10'][0] for m in models) for s in a.splits}; bm_ = max(out['macro'][m]['nDCG@10'] for m in models)   # bold = best per column
for m in models + ['random']:
    cell = lambda s: (f"{out['splits'][s]['random']['nDCG@10']:.3f}" if m == 'random' else f"{out['splits'][s][m]['nDCG@10'][0]:.3f} [{out['splits'][s][m]['nDCG@10'][1]:.3f}, {out['splits'][s][m]['nDCG@10'][2]:.3f}]")
    bc = lambda s: f'**{cell(s)}**' if m != 'random' and out['splits'][s][m]['nDCG@10'][0] == bs_[s] else cell(s)
    top = m != 'random' and out['macro'][m]['nDCG@10'] == bm_
    L.append(f"| {'**' + m + '**' if top else m} | " + ' | '.join(bc(s) for s in a.splits) + (f" | **{out['macro'][m]['nDCG@10']:.3f}** |" if top else f" | {out['macro'][m]['nDCG@10']:.3f} |"))
for s in a.splits:
    S = out['splits'][s]; L += [f"\n## {s} (query {S['n']}, corpus {S['corpus']:,}, gold/query {S['gold_mean']})\n", '| Model | ' + ' | '.join(K) + ' |', '|---|' + '---|' * len(K)]
    bk = {k: max(S[m][k][0] for m in models) for k in K}
    for m in models: L.append(f"| {'**' + m + '**' if S[m]['nDCG@10'][0] == bk['nDCG@10'] else m} | " + ' | '.join((lambda t: f'**{t}**' if S[m][k][0] == bk[k] else t)(f'{S[m][k][0]:.3f} [{S[m][k][1]:.3f}, {S[m][k][2]:.3f}]') for k in K) + ' |')
    L.append('| random | ' + ' | '.join(f"{S['random'][k]:.4f}" for k in K) + ' |')
(O / 'eval_full.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
