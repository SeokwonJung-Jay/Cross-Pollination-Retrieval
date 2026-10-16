#!/usr/bin/env python3
"""Split the combined leave-one-out run (prepare.py + evaluation/main/encode.py) into one work dir per pool and score each with
evaluation/main/score.py. Scores are copied as they are; nothing is re-encoded.
For pool in loo_e5 / loo_splade / loo_scincl writes OUT/<pool>/_data/{pools,texts,source}.json (original query ids) and
OUT/<pool>/<model>/results/scores.json (that pool's rows), then runs score.py -> OUT/<pool>/results/{eval.json, eval.md, per_query.json}
and OUT/<pool>/<model>/results/{metrics.json, per_query.json}.
  python -I split.py --work WORK --out OUT [--registry ../main/models.json] [--n 1300]"""
import argparse, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; MAIN = HERE.parent / 'main'
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--registry', default=str(MAIN / 'models.json')); ap.add_argument('--n', type=int, default=1300); a = ap.parse_args()
W, O = Path(a.work), Path(a.out)
pools = json.load(open(W / '_data' / 'pools.json')); texts = json.load(open(W / '_data' / 'texts.json')); src = json.load(open(W / '_data' / 'source.json'))
models = sorted(p.parent.parent.name for p in W.glob('*/results/scores.json'))
strip = lambda q, pool: q[:-len(pool) - 1] if q.endswith('@' + pool) else None
rc = 0
for pool in ('loo_e5', 'loo_splade', 'loo_scincl'):
    D = O / pool; (D / '_data').mkdir(parents=True, exist_ok=True)
    sub = {strip(q, pool): x for q, x in pools.items() if strip(q, pool)}
    assert len(sub) == a.n, (pool, len(sub))
    need = {c for x in sub.values() for c in [x['pos_cid']] + [c for s in x['main'].values() for c in s]}
    json.dump(sub, open(D / '_data' / 'pools.json', 'w')); json.dump({c: texts[c] for c in sorted(need)}, open(D / '_data' / 'texts.json', 'w'))
    json.dump({**src, 'queries': len(sub), 'docs': len(need), 'leave_one_out_pool': pool, 'split_from': str(W.resolve())},
              open(D / '_data' / 'source.json', 'w'), indent=1)
    for m in models:
        S = json.load(open(W / m / 'results' / 'scores.json'))
        if set(S) <= {'main', 'cross_domain'}:   # bm25: {variant: {query_id: {cid: score}}} (idf depends on the pool)
            out = {v: {strip(q, pool): s for q, s in S[v].items() if strip(q, pool)} for v in S}
        else:
            out = {strip(q, pool): s for q, s in S.items() if strip(q, pool)}
        (D / m / 'results').mkdir(parents=True, exist_ok=True); json.dump(out, open(D / m / 'results' / 'scores.json', 'w'))
        if (W / m / 'config.json').exists(): (D / m / 'config.json').write_text((W / m / 'config.json').read_text())
    r = subprocess.run([sys.executable, '-I', str(MAIN / 'score.py'), '--work', str(D), '--registry', a.registry, '--out', str(D / 'results'),
                        '--title', f'Leave-one-out pool {pool}'])
    print(pool, 'models', len(models), 'score.py exit', r.returncode, flush=True); rc |= r.returncode
sys.exit(rc)
