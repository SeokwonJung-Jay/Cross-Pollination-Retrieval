#!/usr/bin/env python3
"""Main experiment scoring.
Pool variants: main = pos + q_side 50 + pos_side 50 + other 50 = 151 docs; cross_domain = q_side dropped = 101 docs.
Per query the rank of the positive inside the pool (descending score; ties -> the positive ranks after every doc with an equal score);
R@1/5/10 (positive in top k), MRR@10 (1/rank if rank <= 10 else 0), nDCG@10 (one positive: 1/log2(rank+1) if rank <= 10 else 0), mean over queries;
95% CI = bootstrap over queries, 2,000 resamples, random.Random(0) per variant, 2.5/97.5 percentiles (sorted index 50 / 1949);
random = exact expectation for one positive in N docs (R@k = k/N, MRR@10 = sum_{r<=10} 1/(N r), nDCG@10 = sum_{r<=10} 1/(N log2(r+1))).
Domain tables (paper main table): query domain = OpenAlex domain of the query's `field` (field_domain.json);
per domain Health / Life / Physical / Social the mean over that domain's queries, Avg = mean over all queries;
CI = stratified bootstrap (resample queries within each domain, 2,000x, random.Random(0) per variant). Main table R@1, R@10; appendix R@5, nDCG@10.
Reads WORK/<model>/results/scores.json (encode.py / encode_lasttoken.py) for every models.json entry that has one. Writes
  WORK/<model>/results/{metrics.json, per_query.json}   and   results/{eval.json, eval.md, per_query.json} next to this file.
  python -I score.py --work WORK [--registry models.json --out results --title ...]"""
import argparse, json, math, random, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--field-domain', default=str(HERE / 'field_domain.json'))
ap.add_argument('--registry', default=str(HERE / 'models.json')); ap.add_argument('--out', default=str(HERE / 'results')); ap.add_argument('--title', default='Main results')
a = ap.parse_args(); W = Path(a.work); REG = json.load(open(a.registry))
pools = json.load(open(W / '_data' / 'pools.json')); qids = sorted(pools)
VAR = {'main': ('q_side', 'pos_side', 'other'), 'cross_domain': ('pos_side', 'other')}
P = {v: {q: [x['pos_cid']] + [c for s in st for c in x['main'][s]] for q, x in pools.items()} for v, st in VAR.items()}
KS = ('R@1', 'R@5', 'R@10', 'MRR@10', 'nDCG@10')
FD = json.load(open(a.field_domain)); F2D = {f: d.split()[0] for d, fs in FD.items() if not d.startswith('_') for f in fs}
DOMS = ['Health', 'Life', 'Physical', 'Social']; QD = {q: F2D[x['field']] for q, x in pools.items()}; DQ = {d: [q for q in qids if QD[q] == d] for d in DOMS}
assert sum(map(len, DQ.values())) == len(qids), 'query without domain'
def rank(S, pool): return 1 + sum(S[d] >= S[pool[0]] for d in pool[1:])   # pessimistic on ties
def met(rs): return {'R@1': statistics.mean(r <= 1 for r in rs), 'R@5': statistics.mean(r <= 5 for r in rs), 'R@10': statistics.mean(r <= 10 for r in rs),
                     'MRR@10': statistics.mean(1 / r if r <= 10 else 0 for r in rs), 'nDCG@10': statistics.mean(1 / math.log2(r + 1) if r <= 10 else 0 for r in rs)}
models = [m for m in REG if not m.startswith('_') and (W / m / 'results' / 'scores.json').exists()]
missing = [m for m in REG if not m.startswith('_') and m not in models]
out = {'eval_set': json.load(open(W / '_data' / 'source.json')), 'work': str(W), 'models': models, 'missing': missing, 'summary': {}}; per_q = {v: {} for v in VAR}
for v in VAR:
    rng = random.Random(0); B = [[rng.choice(qids) for _ in qids] for _ in range(2000)]; N = len(P[v][qids[0]])
    s = {'n': len(qids), 'pool_size': N, 'random': {'R@1': 1 / N, 'R@5': 5 / N, 'R@10': 10 / N, 'MRR@10': sum(1 / (N * r) for r in range(1, 11)), 'nDCG@10': sum(1 / (N * math.log2(r + 1)) for r in range(1, 11))}}
    for m in models:
        S = json.load(open(W / m / 'results' / 'scores.json')); S = S[v] if REG[m]['kind'] == 'bm25' else S
        r = {q: rank(S[q], P[v][q]) for q in qids}; pt = met(list(r.values())); bs = [met([r[q] for q in b]) for b in B]
        s[m] = {k: [round(pt[k], 4), round(sorted(x[k] for x in bs)[50], 4), round(sorted(x[k] for x in bs)[1949], 4)] for k in KS}; per_q[v][m] = r
    out['summary'][v] = s
    rng = random.Random(0); BD = [{d: [rng.choice(DQ[d]) for _ in DQ[d]] for d in DOMS} for _ in range(2000)]; dm = {'n': {d: len(DQ[d]) for d in DOMS}}
    for m in models:
        r = per_q[v][m]; pt = {d: met([r[q] for q in DQ[d]]) for d in DOMS}; pt['Avg'] = met(list(r.values()))
        bs = []
        for b in BD: x = {d: met([r[q] for q in b[d]]) for d in DOMS}; x['Avg'] = met([r[q] for d in DOMS for q in b[d]]); bs.append(x)
        dm[m] = {d: {k: [round(pt[d][k], 4), round(sorted(x[d][k] for x in bs)[50], 4), round(sorted(x[d][k] for x in bs)[1949], 4)] for k in KS} for d in DOMS + ['Avg']}
    out.setdefault('by_domain', {})[v] = dm
for m in models:
    json.dump({v: {'all': out['summary'][v][m], 'by_domain': out['by_domain'][v][m]} for v in VAR}, open(W / m / 'results' / 'metrics.json', 'w'), indent=1)
    json.dump({v: per_q[v][m] for v in VAR}, open(W / m / 'results' / 'per_query.json', 'w'))
R = Path(a.out); R.mkdir(parents=True, exist_ok=True)
json.dump(out, open(R / 'eval.json', 'w'), indent=1); json.dump(per_q, open(R / 'per_query.json', 'w'))
src = out['eval_set']; L = [f'# {a.title}', '', f"Eval set `{src['repo']}` ({src['file']}, rev `{src['revision'][:8]}`), {src['queries']} queries, {src['docs']:,} documents. "
     'Bold = best value per column (domain tables: also the model name best on an Avg column). Made by `score.py` from `' + str(W) + '/<model>/`; 95% CIs in `eval.json`.', '',
     f"Domain = OpenAlex domain of the query field (queries: " + ', '.join(f"{d} {len(DQ[d])}" for d in DOMS) + "). Avg = mean over all queries.", '']
def dtab(v, ks, title):   # domain header row, metric row under it (HTML for the merged header), Avg once at the right
    dm = out['by_domain'][v]; cols = [(d, k) for d in DOMS + ['Avg'] for k in ks]; best = {c: max(dm[m][c[0]][c[1]][0] for m in models) for c in cols}
    cell = lambda m, d, k: f"<b>{dm[m][d][k][0]:.3f}</b>" if dm[m][d][k][0] == best[(d, k)] else f"{dm[m][d][k][0]:.3f}"
    T = [title, '', '<table>', '<tr><th rowspan="2">Family</th><th rowspan="2">Model</th>' + ''.join(f'<th colspan="{len(ks)}">{d}</th>' for d in DOMS + ['Avg']) + '</tr>',
         '<tr>' + ''.join(f'<th>{k}</th>' for _ in DOMS + ['Avg'] for k in ks) + '</tr>']
    win = {m for m in models for k in ks if dm[m]['Avg'][k][0] == best[('Avg', k)]}   # the model best on an Avg column is bold too
    for m in models: T.append(f"<tr><td>{REG[m]['family']}</td><td>{'<b>' + REG[m]['label'] + '</b>' if m in win else REG[m]['label']}</td>" + ''.join(f'<td>{cell(m, d, k)}</td>' for d, k in cols) + '</tr>')
    return T + ['</table>', '']
L += dtab('main', ('R@1', 'R@10'), '## Main table — R@1, R@10 per domain (main pool 151)')
L += dtab('main', ('R@5', 'nDCG@10'), '## Appendix — R@5, nDCG@10 per domain (main pool 151)')
L += ['## All queries (no domain split)', '']
for v in VAR:
    s = out['summary'][v]; best = {k: max(s[m][k][0] for m in models) for k in KS}
    L += [f"### {v} ({s['n']} queries, pool {s['pool_size']}) — point estimate [95% CI]", '', '| Family | Model | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |', '|---|---|---|---|---|---|---|']
    for m in models:
        L.append(f"| {REG[m]['family']} | {REG[m]['label']} | " + ' | '.join((f"**{s[m][k][0]:.3f}**" if s[m][k][0] == best[k] else f"{s[m][k][0]:.3f}") + f" [{s[m][k][1]:.3f}, {s[m][k][2]:.3f}]" for k in KS) + ' |')
    L += ['| — | random | ' + ' | '.join(f"{s['random'][k]:.3f}" for k in KS) + ' |', '']
if missing: L += [f"not run: {', '.join(missing)}", '']
open(R / 'eval.md', 'w').write('\n'.join(L)); print('\n'.join(L))
