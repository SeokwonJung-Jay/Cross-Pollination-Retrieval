#!/usr/bin/env python3
"""Held-out domain result tables. Per base model (Qwen3-Embedding-0.6B / 4B) three rows on the CPR-eval queries of each domain
(Health 250 / Life 250 / Physical 500 / Social 300, pool 151):
  before training  = the vanilla model
  without domain   = for each domain column the fold model trained without that domain (domabl_<base>_<domain>, 5,896 pairs)
  all domains      = the dataset-ablation run on all 10,573 training pairs (abl2v2_<base>_hardneg_ours)
Values and 95% CIs per domain are evaluation/main/score.py's (WORK/<model>/results/metrics.json, main pool, by_domain: queries resampled
within each domain, 2,000 resamples, random.Random(0)). Avg = unweighted mean of the 4 domain values (point estimate only).
Also prints the Recall@10 gain of the fold models over the vanilla model per domain (paper: Figure domain, Section 4.4).
--folds adds the training-pair counts per fold (build_folds.py folds.json).
Writes OUT/eval.json and OUT/RES.md.
  python -I tables.py --work WORK/ours --out results [--folds FOLDS/folds.json]"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--out', required=True); ap.add_argument('--folds'); A = ap.parse_args()
DOMS = ['Health', 'Life', 'Physical', 'Social']; KS = ('R@1', 'R@5', 'R@10', 'nDCG@10')
BASES = [('qwen3_0.6b', 'Qwen3-Embedding-0.6B'), ('qwen3_4b', 'Qwen3-Embedding-4B')]
ROWS = [('before training', lambda b: {d: b for d in DOMS}), ('without domain', lambda b: {d: f'domabl_{b}_{d.lower()}' for d in DOMS}),
        ('all domains', lambda b: {d: f'abl2v2_{b}_hardneg_ours' for d in DOMS})]
def metrics(m):
    p = os.path.join(A.work, m, 'results', 'metrics.json')
    return json.load(open(p))['main']['by_domain'] if os.path.exists(p) else None
out = {'tables': {}}
for b, _ in BASES:
    T = {}
    for name, f in ROWS:
        ms = f(b); M = {d: metrics(ms[d]) for d in DOMS}
        if any(M[d] is None for d in DOMS): continue
        T[name] = {'models': ms, **{d: {k: M[d][d][k] for k in KS} for d in DOMS}}
        T[name]['Avg'] = {k: round(sum(T[name][d][k][0] for d in DOMS) / 4, 4) for k in KS}
    out['tables'][b] = T
os.makedirs(A.out, exist_ok=True); json.dump(out, open(f'{A.out}/eval.json', 'w'), indent=1)

L = ['# Held-out domain results (max length 1024)', '',
     'Rows per base: before training (vanilla), without domain (each domain column: the fold model trained without that domain, 5,896 pairs), '
     'all domains (all 10,573 training pairs). Columns: CPR-eval queries of the domain (Health 250, Life 250, Physical 500, Social 300), pool 151. '
     'Avg = unweighted mean of the 4 domains.', '']
if A.folds:
    F = json.load(open(A.folds)); before = next(iter(F['folds'].values()))['train_before_leak_removal']
    L += [f"## Training pairs per fold ({F['train_total']:,} pairs, {F['train_no_domain_dropped']} without a positive domain dropped -> {before:,}; seed {F['seed']})", '',
          '| held-out | query in domain | only positive in domain | remaining | used |', '|---|---|---|---|---|']
    for d in DOMS:
        x = F['folds'][d.lower()]; L.append(f"| {d} | {x['removed_q_in_D']:,} | {x['removed_pos_only_in_D']:,} | {x['train_after_leak_removal']:,} | {x['train_final']:,} |")
    L.append('')
L += ['## Recall@10 [95% CI]', '', '| base | training data | ' + ' | '.join(DOMS) + ' |', '|---|---|' + '---|' * len(DOMS)]
for b, label in BASES:
    for name, x in out['tables'][b].items():
        L.append(f'| {label} | {name} | ' + ' | '.join(f"{x[d]['R@10'][0]:.3f} [{x[d]['R@10'][1]:.3f}, {x[d]['R@10'][2]:.3f}]" for d in DOMS) + ' |')
L.append('')
for ks, title in ((('R@1', 'R@10'), 'R@1, R@10'), (('R@5', 'nDCG@10'), 'R@5, nDCG@10')):
    L += [f'## {title}', '', '| base | training data | ' + ' | '.join(f'{d} {k}' for d in DOMS + ['Avg'] for k in ks) + ' |', '|---|---|' + '---|' * (len(ks) * 5)]
    for b, label in BASES:
        for name, x in out['tables'][b].items():
            L.append(f'| {label} | {name} | ' + ' | '.join(f"{(x[d][k] if d == 'Avg' else x[d][k][0]):.3f}" for d in DOMS + ['Avg'] for k in ks) + ' |')
    L.append('')
L += ['## Recall@10 gain of the model trained without the domain', '', '| base | over | ' + ' | '.join(DOMS) + ' | range |', '|---|---|' + '---|' * (len(DOMS) + 1)]
for b, label in BASES:
    T = out['tables'][b]
    for ref in ('before training', 'all domains'):
        if 'without domain' not in T or ref not in T: continue
        g = [T['without domain'][d]['R@10'][0] - T[ref][d]['R@10'][0] for d in DOMS]
        L.append(f'| {label} | {ref} | ' + ' | '.join(f'{x:+.3f}' for x in g) + f' | {min(g):+.2f} to {max(g):+.2f} |')
L.append('')
open(f'{A.out}/RES.md', 'w').write('\n'.join(L)); print('\n'.join(L))
