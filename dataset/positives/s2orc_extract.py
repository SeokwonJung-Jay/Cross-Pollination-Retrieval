#!/usr/bin/env python3
"""s2orc_v2 records ($S2ORC) of the base papers -> <stage2_cache>/s2orc_q/<corpusid>.json (one line each). Resumable: one done
marker per input file and q set.
  python -I s2orc_extract.py CONFIG"""
import gzip, sys
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env, load_config, base_papers, stage2_dirs
C = load_config(sys.argv[1]); _, CA = stage2_dirs(C); OUT = CA / 's2orc_q'; DONE = OUT / f"_done_{C['name']}"; DONE.mkdir(parents=True, exist_ok=True)
Q = {c for v in base_papers(C).values() for c in v} - {int(f.stem) for f in OUT.glob('*.json')}


def one(f):
    if (DONE / f.name).exists(): return 0
    n = 0
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            j = l.find(",", 12)
            try: c = int(l[12:j])   # '{"corpusid": ' is 12 characters
            except ValueError: continue
            if c in Q: (OUT / f'{c}.json').write_text(l); n += 1
    (DONE / f.name).write_text(str(n)); return n


if __name__ == '__main__':
    fs = sorted(env('S2ORC').glob('*.gz')); tot = 0
    with Pool(24) as p:
        for i, n in enumerate(p.imap_unordered(one, fs), 1):
            tot += n
            if i % 25 == 0: print(i, '/', len(fs), 'found', tot, flush=True)
    print('done found', len(list(OUT.glob('*.json'))), flush=True)
