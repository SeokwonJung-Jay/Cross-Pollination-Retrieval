#!/usr/bin/env python3
"""Stage 2 (cross-topic citation mining with citing sentences) on K shards of the base papers, in parallel.
stage2/stage2_candidate_preparation.py runs unchanged on every shard (own papers/references files and output dir) with the OpenAlex
tags, the JATS cache, the s2orc_v2 records and the PDF cache; `merge` concatenates the shard outputs.
  plan  : base papers in pool order per field (config q_pool / q_per_field) that have references; round-robin over K shards
          (config workers.stage2_shards)                     -> <work>/stage2/shard_NN/{papers,references}.json
  run   : all shards in parallel (resumable per shard)
  merge : -> <work>/stage2/{prepared_pairs.json, bases.jsonl, candidate_pairs.jsonl, pair_audit.jsonl}
Run jats_prefetch.py first: a q with a PMCID whose JATS is neither cached nor marked missing would fall through to s2orc_v2 / PDF.
  python -I stage2_parallel.py CONFIG plan|run|merge"""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, base_papers, stage2_dirs
C = load_config(sys.argv[1]); CMD = sys.argv[2]; I, CA = stage2_dirs(C); W = Path(C['work']) / 'stage2'; W.mkdir(exist_ok=True)
STAGE2 = Path(__file__).resolve().parent / 'stage2' / 'stage2_candidate_preparation.py'


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


if CMD == 'plan':
    P = json.load(open(I / 'papers.json')); R = json.load(open(I / 'references.json'))
    per = {f: [str(c) for c in qs if str(c) in R] for f, qs in base_papers(C).items()}
    Q = [c for v in per.values() for c in v]; K = C['workers'].get('stage2_shards', 16)
    json.dump({'per_field': per}, open(W / 'plan.json', 'w'), indent=1)
    for k in range(K):   # round-robin keeps field mix and pool order similar across shards
        sub = Q[k::K]; d = W / f'shard_{k:02d}'; d.mkdir(exist_ok=True)
        refs = {q: R[q] for q in sub}; need = set(sub) | {r['citedPaper']['paperId'] for v in refs.values() for r in v}
        json.dump({c: P[c] for c in need if c in P}, open(d / 'papers.json', 'w')); json.dump(refs, open(d / 'references.json', 'w'))
    log(f'{len(Q)} q in {K} shards')
elif CMD == 'run':
    procs = []
    for d in sorted(W.glob('shard_*')):
        out = d / 'out'; res = ['--resume'] if (out / 'bases.jsonl').exists() else []
        cmd = [sys.executable, '-I', STAGE2, '--papers', d / 'papers.json', '--references', d / 'references.json', '--tags', I / 'tags_openalex.jsonl', '--out', out,
               '--fulltext-cache', CA / 'fulltext_cache', '--jats-cache', CA / 'jats_cache', '--s2orc-dir', CA / 's2orc_q', '--pair-prefix', C.get('pair_prefix', 'scaleup'), *res]
        procs.append((d.name, subprocess.Popen(list(map(str, cmd)), stdout=open(d / 'stage2.log', 'a'), stderr=subprocess.STDOUT)))
    t0 = time.time(); bad = [n for n, p in procs if p.wait() != 0]
    log(f'{len(procs)} shards done in {time.time() - t0:.0f}s | failed {bad}'); sys.exit(1 if bad else 0)
elif CMD == 'merge':
    pairs = []
    for d in sorted(W.glob('shard_*')): pairs += json.load(open(d / 'out/prepared_pairs.json'))
    json.dump(pairs, open(W / 'prepared_pairs.json', 'w'))
    for name in ('bases.jsonl', 'candidate_pairs.jsonl', 'pair_audit.jsonl'):
        with open(W / name, 'w') as fo:
            for d in sorted(W.glob('shard_*')):
                if (d / 'out' / name).exists(): fo.write(open(d / 'out' / name).read())
    log(f'merged {len(pairs)} prepared pairs from {len({p["base_id"] for p in pairs})} q')
else: sys.exit(__doc__)
