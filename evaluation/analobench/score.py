#!/usr/bin/env python3
"""Score the AnaloBench embeddings (encode.py). Score = cosine (normalized vectors), query vector = 'Query:' + story.
T2 (query vs its 200-story bank, 7.24 gold on average): nDCG@10 (binary gains, as BRIGHT) + the paper's T2 metrics (Table "P@3 P@5 R@3 R@5 MAP
MRR", LLMs return a ranked top-10 list): P@k = gold in top k / k, R@k = gold in top k / |gold|, MAP = AP over the top 10 = sum over gold at rank
r <= 10 of P@r, divided by |gold| (trec_eval map_cut_10; the paper does not give its normalization), MRR = 1 / rank of the first gold within the
top 10 (0 if none). T1 (4 options): accuracy = argmax option is the gold (ties never happen with float scores). T1-Full is the main T1,
T1-Subset the paper's subset (reference).
95% CI = bootstrap (2,000 resamples, random.Random(0)) over T2 queries; for T1-Full over query stories (each story is the query of ~7 rows),
for T1-Subset over rows. Random = exact expected value (T2: N = 200 bank stories, per query gold count; T1: 1/4).
Output results/eval.json {"m<max_len>": {"S<n>": {model: {metric: [value, lo, hi]}}, "mean": {model: {metric: mean over S1/S10/S30}}}},
results/eval.md, results/per_query.json {"m<max_len>": {"S<n>": {model: {"t2": [gold ranks in the bank per query], "t1_full": "0/1 per row",
"t1_sub": ...}}}}.
  python -I score.py --work WORK --models MODEL ... --out DIR"""
import argparse, json, math, random, statistics
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True)
ap.add_argument('--models', nargs='+', required=True, help='models to score (encode.py outputs in WORK/emb/), in table order')
ap.add_argument('--out', help='results dir (default: results/ next to this file)'); a = ap.parse_args(); W = Path(a.work)
R = Path(a.out) if a.out else Path(__file__).resolve().parent / 'results'
MODELS = a.models; LENS, MAXLEN = (1, 10, 30), (512, 1024)
T2K = ['nDCG@10', 'MAP', 'MRR', 'P@3', 'P@5', 'R@3', 'R@5']; K = T2K + ['T1 acc', 'T1-sub acc']
missing = [m for m in MODELS if not all((W / 'emb' / m / f'm{x}' / f'L{n}_d.npy').exists() for x in MAXLEN for n in LENS)]; assert not missing, missing
def t2m(rk):   # rk: sorted 1-based ranks of the gold stories in the bank
    g = len(rk); top = lambda k: sum(r <= k for r in rk)
    return {'nDCG@10': sum(1 / math.log2(r + 1) for r in rk if r <= 10) / sum(1 / math.log2(i + 2) for i in range(min(g, 10))),
            'MAP': sum((i + 1) / r for i, r in enumerate(rk) if r <= 10) / g, 'MRR': 1 / rk[0] if rk[0] <= 10 else 0.0,
            'P@3': top(3) / 3, 'P@5': top(5) / 5, 'R@3': top(3) / g, 'R@5': top(5) / g}
def rand_t2(g, N=200):
    C = math.comb(N, g)
    return {'nDCG@10': g / N * sum(1 / math.log2(r + 1) for r in range(1, 11)) / sum(1 / math.log2(i + 2) for i in range(min(g, 10))),
            'MAP': sum(g / N * (1 + (k - 1) * (g - 1) / (N - 1)) / k for k in range(1, 11)) / g,
            'MRR': sum(math.comb(N - r, g - 1) / C / r for r in range(1, 11)), 'P@3': g / N, 'P@5': g / N, 'R@3': 3 / N, 'R@5': 5 / N}
mean = lambda xs: statistics.mean(xs)
def ci(vals, groups, B):   # vals: per row; groups: list of row-index lists (bootstrap unit); B: (2000, n units) resampled unit indices
    v = np.asarray(vals, np.float64); gs = np.array([v[g].sum() for g in groups]); gn = np.array([len(g) for g in groups], np.float64)
    s = np.sort(gs[B].sum(1) / gn[B].sum(1)); return [round(float(v.mean()), 4), round(float(s[50]), 4), round(float(s[1949]), 4)]
out, PQ = {}, {}
for x in MAXLEN:
    out[f'm{x}'] = {}; PQ[f'm{x}'] = {}
    for n in LENS:
        D = json.load(open(W / 'data' / f'L{n}.json')); t2, t1f, t1s = D['t2'], D['t1_full'], D['t1_sub']
        stq = sorted({r['q'] for r in t1f}); gf = [[i for i, r in enumerate(t1f) if r['q'] == q] for q in stq]
        assert sorted(i for g in gf for i in g) == list(range(len(t1f)))
        rng = random.Random(0); B2 = np.array([[rng.randrange(len(t2)) for _ in t2] for _ in range(2000)])
        Bf = np.array([[rng.randrange(len(gf)) for _ in gf] for _ in range(2000)]); Bs = np.array([[rng.randrange(len(t1s)) for _ in t1s] for _ in range(2000)])
        g2 = [[i] for i in range(len(t2))]; gs = [[i] for i in range(len(t1s))]
        rd = [rand_t2(len(r['gold'])) for r in t2]
        res = {'n': {'t2': len(t2), 't1_full': len(t1f), 't1_full_stories': len(stq), 't1_sub': len(t1s)},
               'random': {**{k: round(mean([d[k] for d in rd]), 4) for k in T2K}, 'T1 acc': 0.25, 'T1-sub acc': 0.25}}
        PQ[f'm{x}'][f'S{n}'] = {}
        for m in MODELS:
            E = W / 'emb' / m / f'm{x}'
            S = np.load(E / f'L{n}_q.npy') @ np.load(E / f'L{n}_d.npy').T
            ranks = []
            for r in t2:
                order = sorted(r['bank'], key=lambda j: -S[r['q'], j]); pos = {j: i + 1 for i, j in enumerate(order)}
                ranks.append(sorted(pos[j] for j in r['gold']))
            mt = [t2m(rk) for rk in ranks]
            acc = lambda rows: [int(max(r['opts'], key=lambda j: S[r['q'], j]) == r['gold']) for r in rows]
            af, as_ = acc(t1f), acc(t1s)
            res[m] = {**{k: ci([d[k] for d in mt], g2, B2) for k in T2K}, 'T1 acc': ci(af, gf, Bf), 'T1-sub acc': ci(as_, gs, Bs)}
            PQ[f'm{x}'][f'S{n}'][m] = {'t2': ranks, 't1_full': ''.join(map(str, af)), 't1_sub': ''.join(map(str, as_))}
        out[f'm{x}'][f'S{n}'] = res; print(f'max {x} S{n} done', flush=True)
    out[f'm{x}']['mean'] = {m: {k: round(mean([out[f'm{x}'][f'S{n}'][m][k][0] for n in LENS]), 4) for k in K} for m in MODELS}
    out[f'm{x}']['mean']['random'] = {k: round(mean([out[f'm{x}'][f'S{n}']['random'][k] for n in LENS]), 4) for k in K}
R.mkdir(parents=True, exist_ok=True); json.dump(out, open(R / 'eval.json', 'w'), indent=1); json.dump(PQ, open(R / 'per_query.json', 'w'))
L = ['# AnaloBench (`score.py`)', '',
     'Query = `Query:` + story, document = story. T2: a 200-story bank per query, 7.24 gold on average. T1: 4 options. Values [95% bootstrap CI].',
     'S1 / S10 / S30 = story length (1 / ~10 / ~30 sentences, the same 340 stories). T1 acc = T1-Full accuracy (2,460 four-option items), T1-sub acc = the paper\'s T1 Subset (340).',
     'Bold = best value per column (random excluded); bold model name = best T2 nDCG@10 or T1 acc.', '']
for x in MAXLEN:
    for n in LENS + ('mean',):
        sec = out[f'm{x}'][n if n == 'mean' else f'S{n}']
        L += [f'## max {x} tokens — ' + ('mean of S1 / S10 / S30' if n == 'mean' else f'S{n}'), '', '| Model | ' + ' | '.join(K) + ' |', '|---|' + '---|' * len(K)]
        pt = lambda m, k: round(sec[m][k] if n == 'mean' else sec[m][k][0], 3)   # bold the best value per column, and the
        best = {k: max(pt(m, k) for m in MODELS) for k in K}                     # model best on the two headline metrics (T2 nDCG@10, T1 acc)
        for m in MODELS + ['random']:
            v = sec[m]; f = (lambda k: f'{v[k]:.3f}') if n == 'mean' or m == 'random' else (lambda k: f'{v[k][0]:.3f} [{v[k][1]:.3f}, {v[k][2]:.3f}]')
            b = lambda k: f'**{f(k)}**' if m != 'random' and pt(m, k) == best[k] else f(k)
            name = f'**{m}**' if m != 'random' and any(pt(m, k) == best[k] for k in ('nDCG@10', 'T1 acc')) else m
            L.append(f'| {name} | ' + ' | '.join(b(k) for k in K) + ' |')
        L.append('')
open(R / 'eval.md', 'w').write('\n'.join(L)); print('wrote', R / 'eval.json')
