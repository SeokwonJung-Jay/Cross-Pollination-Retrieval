#!/usr/bin/env python3
"""Dataset ablation result tables (markdown): one summary row per model (all benchmarks), then one detail table per benchmark. Rows are grouped
by base model; bold = best per column within a base (base models are not compared); vanilla = the base model without training.
  Summary: BRIGHT macro nDCG@10 | AnaloBench T2 nDCG@10 and T1 acc (mean over S1/S10/S30) | CPR-eval R@1 and R@10 (all 1,300 queries)
  BRIGHT = StackExchange 7 splits nDCG@10 + macro | AnaloBench (max 1024) = T2 nDCG@10 S1/S10/S30/mean + T1 acc mean
  | CPR-eval = R@1 / R@10 per query domain + Avg over all queries (pool 151).
Rows = models evaluated so far (eval.sh calls this after scoring).
  python -I tables.py --eval WORK --res-dir results --out results/RES.md"""
import argparse, json, os
BASES = [('qwen3_0.6b', 'Qwen3-Embedding-0.6B', ('qwen06b_qp', 'qwen06b', 'qwen3_0.6b')),   # base, title, vanilla names (BRIGHT, AnaloBench, CPR-eval)
         ('qwen3_4b', 'Qwen3-Embedding-4B', ('qwen4b_qp', 'qwen4b', 'qwen3_4b')),
         ('nemotron8b', 'llama-embed-nemotron-8b', ('nemotron8b_qp', 'nemotron8b_qp', 'nemotron8b'))]
RUNS = [(n, d) for n in ('hardneg', 'inbatch') for d in ('ours', 'msmarco', 'reasonir', 'specter')]
SE = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 'stackoverflow', 'sustainable_living']
DOMS = ['Health', 'Life', 'Physical', 'Social', 'Avg']
ld = lambda p: json.load(open(p)) if os.path.exists(p) else {}

def table(head, groups):   # groups: [(base title, [(label, [float | None])])]; bold = best per column within the base group (ties too)
    L = ['| base | model | ' + ' | '.join(head) + ' |', '|---|---|' + '---|' * len(head)]
    for title, rows in groups:
        best = [max((r[1][j] for r in rows if r[1][j] is not None), default=None) for j in range(len(head))]
        f = lambda v, b: '-' if v is None else (f'**{v:.3f}**' if b is not None and round(v, 3) == round(b, 3) else f'{v:.3f}')
        L += [f'| {title} | {lab} | ' + ' | '.join(f(v, best[j]) for j, v in enumerate(vals)) + ' |' for lab, vals in rows]
    return L

def build(E, R):
    BR, ANA = ld(f'{E}/bright/eval_full.json'), ld(f'{R}/analobench/eval.json')
    AN = ANA.get('m1024', {})
    G = {k: [] for k in ('summary', 'bright', 'analobench', 'ours')}
    for base, title, (vb, va, vo) in BASES:
        models = [('vanilla', vb, va, vo)] + [(f'{n} / {d}',) + (f'abl2v2_{base}_{n}_{d}',) * 3 for n, d in RUNS]
        rows = []
        for lab, b, a, o in models:
            m = ld(f'{E}/ours/{o}/results/metrics.json'); av = m['main']['all'] if m else None
            v = [BR['macro'][b]['nDCG@10'] if b in BR.get('macro', {}) else None] + \
                ([AN['mean'][a]['nDCG@10'], AN['mean'][a]['T1 acc']] if a in AN.get('mean', {}) else [None, None]) + \
                ([av['R@1'][0], av['R@10'][0]] if av else [None, None])
            if any(x is not None for x in v): rows.append((lab, v))
        if rows: G['summary'].append((title, rows))
        rows = [(lab, [BR['splits'][s][b]['nDCG@10'][0] for s in SE] + [BR['macro'][b]['nDCG@10']]) for lab, b, _, _ in models if b in BR.get('macro', {})]
        if rows: G['bright'].append((title, rows))
        rows = [(lab, [AN[f'S{n}'][a]['nDCG@10'][0] for n in (1, 10, 30)] + [AN['mean'][a]['nDCG@10'], AN['mean'][a]['T1 acc']])
                for lab, _, a, _ in models if a in AN.get('mean', {})]
        if rows: G['analobench'].append((title, rows))
        rows = []
        for lab, _, _, o in models:
            m = ld(f'{E}/ours/{o}/results/metrics.json')
            if m: dm = m['main']['by_domain']; rows.append((lab, [dm[d][k][0] for d in DOMS for k in ('R@1', 'R@10')]))
        if rows: G['ours'].append((title, rows))
    L = ['# Dataset ablation results (max length 1024)', '',
         'Rows grouped by base model; bold = best per column within a base model (base models are not compared). vanilla = the base model '
         'without training; other rows = <negatives> / <training set>. 95% CIs and per-query results: `results/{bright,analobench,ours}/`.', '']
    if G['summary']: L += ['## Summary', '',
                           'BRIGHT = StackExchange 7 splits macro nDCG@10 | AnaloBench = T2 nDCG@10 / T1 acc (mean over S1, S10, S30) | '
                           'CPR-eval = R@1 / R@10 (mean over all 1,300 queries)', ''] + \
                          table(['BRIGHT nDCG@10', 'AnaloBench T2 nDCG@10', 'AnaloBench T1 acc', 'ours R@1', 'ours R@10'], G['summary']) + ['']
    if G['bright']: L += ['## BRIGHT StackExchange: nDCG@10 (whole split corpus searched, all 749 queries; macro = mean of the 7 splits)', ''] + \
                         table(SE + ['macro'], G['bright']) + ['']
    if G['analobench']: L += ['## AnaloBench: T2 nDCG@10 (story length S1 / S10 / S30), T1 acc (4 options, mean over S1, S10, S30)', ''] + \
                             table(['T2 S1', 'T2 S10', 'T2 S30', 'T2 mean', 'T1 acc mean'], G['analobench']) + ['']
    if G['ours']: L += ['## CPR-eval: R@1 / R@10 per domain (pool 151, Avg = mean over all queries)', ''] + \
                       table([f'{d} {k}' for d in DOMS for k in ('R@1', 'R@10')], G['ours']) + ['']
    return '\n'.join(L)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); [ap.add_argument(x, required=True) for x in ('--eval', '--res-dir', '--out')]; a = ap.parse_args()
    open(a.out, 'w').write(build(a.eval, a.res_dir))
