#!/usr/bin/env python3
"""Helpers of build_pools.sh (leave-one-out pools). Subcommands:
  init   --src SRC --eval EVAL --out QUERIES   the eval queries of one pooling run: rows of SRC/s6/queries.json (dataset/negatives/pool_universe.py
                                               spec) whose (query text, positive corpus id) is a query of the eval file
  config --base CFG --work W --name N --corpus C --vectors V --tag T --left-out R --depth K --cap USD --out CFG_OUT
                                               a Stage 7 run config: the eval config with this run's work dir, corpus / vector dirs, pool / short file names
                                               (_pools_loo_<R>_<T>_k<K>.json / _short_loo_...) and spend cap; s7_only dropped
  short  --runs W1 W2 W3 --queries Q --out Q2  queries.json of the next depth: the queries of Q with a stratum short of s7_per_stratum
                                               negatives in any of the runs W1..W3 (their s7/<s7_short_name>)
  final  --tag-dir T --left-out R --depths "300 1000 3000"
                                               T/pools_loo_<R>.json: per query the pool of the deepest depth that pooled it
  python -I pools.py SUBCOMMAND ..."""
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('init', 'config', 'short', 'final'))
for k in ('src', 'eval', 'out', 'base', 'work', 'name', 'corpus', 'vectors', 'tag', 'left-out', 'queries', 'tag-dir', 'depths'): ap.add_argument('--' + k)
ap.add_argument('--depth', type=int); ap.add_argument('--cap', type=float); ap.add_argument('--runs', nargs='+'); a = ap.parse_args()
ld = lambda p: json.load(open(p))
def cfg_of(w): return ld(Path(w) / 'config.json')
def names(tag, out, k): return f'_pools_loo_{out}_{tag}_k{k}.json', f'_short_loo_{out}_{tag}_k{k}.json'

if a.cmd == 'init':
    E = {(json.loads(l)['query'], int(json.loads(l)['pos']['cid'])) for l in open(a.eval)}
    Q = [q for q in ld(Path(a.src) / 's6' / 'queries.json') if (q['query'], int(q['pos_cid'])) in E]
    keys = [(q['query'], int(q['pos_cid'])) for q in Q]; assert len(keys) == len(set(keys)), 'two pooling rows for one eval query'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(Q, open(a.out, 'w'), indent=1)
    print(f'{len(Q)} eval queries of {len(E)} pooled in {a.src}')
elif a.cmd == 'config':
    c = ld(os.path.expandvars(a.base)); c.pop('s7_only', None); pn, sn = names(a.tag, a.left_out, a.depth)
    c.update({'name': a.name, 'work': str(Path(a.work).resolve()), 'corpus': a.corpus, 'vectors': a.vectors, 's7_pools_name': pn, 's7_short_name': sn})
    c.setdefault('caps', {})['s7'] = a.cap
    json.dump(c, open(a.out, 'w'), indent=1)
elif a.cmd == 'short':
    need = set()
    for w in a.runs:
        c = cfg_of(w); need |= {p for p, st, n in ld(Path(w) / 's7' / c['s7_short_name'])}
    Q = [q for q in ld(a.queries) if q['pair_id'] in need]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(Q, open(a.out, 'w'), indent=1); print(f'{len(Q)} queries short in some pool')
elif a.cmd == 'final':
    T = Path(a.tag_dir); P = {}; src = {}
    for k in a.depths.split():
        w = T / f'k{k}_{a.left_out}'
        if not (w / 'config.json').exists(): continue
        x = ld(w / 's7' / cfg_of(w)['s7_pools_name']); P.update(x); src.update({p: int(k) for p in x})
    json.dump(P, open(T / f'pools_loo_{a.left_out}.json', 'w'))
    short = sum(any(len(v) < 50 for v in x['main'].values()) for x in P.values())
    print(f'{a.left_out}: {len(P)} queries | pool depth ' + ', '.join(f'K={k}: {sum(v == int(k) for v in src.values())}' for k in a.depths.split())
          + f' | still short {short}')
