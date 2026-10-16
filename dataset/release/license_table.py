#!/usr/bin/env python3
"""Abstract licenses: one pass over the Semantic Scholar abstracts files ($S2AG/abstracts); every record's openaccessinfo (license,
status) -> $DATA/license/abstracts/<file>.parquet (corpusid, license, status). Resumable per file. cls() maps a license string to
'open' (CC-BY or CC0 / public domain: the abstract may be redistributed), 'nc_or_nd' (another CC license), 'other' or None.
  python -I license_table.py [PROCS]"""
import gzip, json, os, re, sys
from pathlib import Path
from multiprocessing import Pool


def cls(x):
    x = (x or '').lower().replace(' ', '')
    if not x or x == 'none': return None
    if re.fullmatch(r'(cc-?by|ccby)(-?4\.0|-?3\.0|-?2\.0)?|cc0.*|cc-?zero|publicdomain|public-?domain|pd|cc0/pd.*', x): return 'open'
    if x.startswith(('cc', 'cc-')): return 'nc_or_nd'
    return 'other'


def scan(a):
    f, out = a
    import pyarrow as pa, pyarrow.parquet as pq
    o = out / (f.name + '.parquet')
    if o.exists(): return f.name, 'skip'
    ids, lic, st = [], [], []
    with gzip.open(f, 'rt') as g:
        for l in g:
            i = l.find(',"openaccessinfo"')
            if i < 0: continue
            try: c = int(l[12:i])
            except ValueError: continue
            try: x = json.loads(l).get('openaccessinfo') or {}
            except Exception: x = {}
            ids.append(c); lic.append(x.get('license')); st.append(x.get('status'))
    pq.write_table(pa.table({'corpusid': pa.array(ids, pa.int64()), 'license': pa.array(lic, pa.string()), 'status': pa.array(st, pa.string())}), str(o) + '.tmp')
    os.replace(str(o) + '.tmp', o); return f.name, len(ids)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env
    OUT = env('DATA') / 'license' / 'abstracts'; OUT.mkdir(parents=True, exist_ok=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        for r in p.imap_unordered(scan, [(f, OUT) for f in sorted((env('S2AG') / 'abstracts').glob('*.gz'))]): print(*r, flush=True)
