#!/usr/bin/env python3
"""Compact indexes of the Semantic Scholar releases, one output file per input file (resumable), in $DATA/s2index/:
  absids   : corpusids that have an abstract ($S2AG/abstracts)                     -> abs_ids/<f>.npy
  edges    : citation edges (citing, cited) as int32 pairs ($S2AG/citations)       -> edges/<f>.npy
  refcount : (corpusid, referencecount) of every paper ($S2AG/papers; -1 = null)   -> refcount/<f>.npy
  s2orc    : corpusids with a non-empty s2orc_v2 body ($S2ORC)                      -> s2orc_ids/<f>.npy
build_universe.py reads abs_ids and edges; sample_base_papers.py reads refcount and s2orc_ids. The paper used S2AG 2026-09-09 and
s2orc_v2 2026-09-22.
  python -I s2_index.py absids|edges|refcount|s2orc [PROCS]"""
import gzip, sys, numpy as np
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env
OUT = env('DATA') / 's2index'


def abs_one(f):
    o = OUT / 'abs_ids' / (f.name.replace('.gz', '') + '.npy')
    if o.exists(): return f.name, 'skip'
    ids = []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"corpusid":'); j = l.find(',', k)
            if (l.find('"abstract":"') > 0 or l.find('"abstract": "') > 0): ids.append(int(l[k + 11:j]))
    np.save(o, np.array(ids, dtype=np.int64)); return f.name, len(ids)


def edges_one(f):
    o = OUT / 'edges' / (f.name + '.npy')
    if o.exists(): return f.name, 'skip'
    a = []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"citingcorpusid":'); k2 = l.find(',', k); m = l.find('"citedcorpusid":'); m2 = l.find(',', m)
            try: a.append((int(l[k + 17:k2]), int(l[m + 16:m2])))
            except ValueError: continue
    np.save(o, np.array(a, np.int32).reshape(-1, 2)); return f.name, len(a)


def refcount_one(f):
    o = OUT / 'refcount' / (f.name + '.npy')
    if o.exists(): return f.name, 'skip'
    a = []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"corpusid":'); j = l.find(',', k); m = l.find('"referencecount":'); n = l.find(',', m)
            try: a.append((int(l[k + 11:j]), int(l[m + 17:n]) if m > 0 and l[m + 17:n] != 'null' else -1))
            except ValueError: continue
    np.save(o, np.array(a, np.int64).reshape(-1, 2)); return f.name, len(a)


def s2orc_one(f):
    """corpusid is the first key of each s2orc_v2 record"""
    o = OUT / 's2orc_ids' / (f.name + '.npy')
    if o.exists(): return f.name, 'skip'
    ids = []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            j = l.find(',')
            try: c = int(l[12:j])
            except ValueError: continue
            k = l.find('"body":')
            if k > 0 and not l.startswith('"body":null', k) and not l.startswith('"body":{"text":""', k): ids.append(c)
    np.save(o, np.array(ids, np.int64)); return f.name, len(ids)


if __name__ == '__main__':
    cmd = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    fn, sub, src = {'absids': (abs_one, 'abs_ids', lambda: env('S2AG') / 'abstracts'), 'edges': (edges_one, 'edges', lambda: env('S2AG') / 'citations'),
                    'refcount': (refcount_one, 'refcount', lambda: env('S2AG') / 'papers'), 's2orc': (s2orc_one, 's2orc_ids', lambda: env('S2ORC'))}[cmd]
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    with Pool(procs) as p:
        for x in p.imap_unordered(fn, sorted(src().glob('*.gz'))): print(*x, flush=True)
    print(cmd, 'done', flush=True)
