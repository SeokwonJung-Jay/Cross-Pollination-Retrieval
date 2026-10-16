#!/usr/bin/env python3
"""Venue 2-year impact factor from the Semantic Scholar Academic Graph ($S2AG; the paper used release 2026-09-09).
IF(V, 2025) = citations made by 2025 papers to V's 2023-2024 papers / number of V's 2023-2024 papers; venue = publicationvenueid.
  papers    : per papers file -> $DATA/venue_if/papers/<f>.npz (corpusid, year, venue) for papers with a venue and year 2023-2025,
              plus $DATA/venue_if/venue_papers/<f>.tsv (venueid, year, doi) rows of 2015-2025 for the field assignment (venue_field.py)
  lookup    : sorted id arrays for the citation scan
  citations : per citations file -> $DATA/venue_if/cites/<f>.npy citation counts per cited venue (citing year 2025, cited year 2023-2024)
  combine   : $DATA/venue_if/venue_if.parquet (venueid, name, type, n_2023_24, cites_2025, if_2025)
Resumable per file. Run the commands in this order:
  python -I venue_if.py papers|lookup|citations|combine"""
import gzip, json, sys, numpy as np
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env
SNAP = env('S2AG'); OUT = env('DATA') / 'venue_if'; Y = 2025
for d in ('papers', 'venue_papers', 'cites'): (OUT / d).mkdir(parents=True, exist_ok=True)


def papers_one(f):
    o = OUT / 'papers' / (f.name + '.npz')
    if o.exists(): return f.name, 'skip'
    cid, yr, vid = [], [], []; rows = []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            r = json.loads(l); v = r.get('publicationvenueid'); y = r.get('year')
            if not v: continue
            doi = (r.get('externalids') or {}).get('DOI')
            if doi and y and 2015 <= y <= 2025: rows.append(f"{v}\t{y}\t{doi}\n")
            if y in (Y - 2, Y - 1, Y): cid.append(r['corpusid']); yr.append(y); vid.append(v)
    np.savez(o, cid=np.array(cid, np.int64), yr=np.array(yr, np.int16), vid=np.array(vid))
    open(OUT / 'venue_papers' / (f.name + '.tsv'), 'w').writelines(rows); return f.name, len(cid)


_L = {}


def _load():
    if _L: return
    z = np.load(OUT / 'lookup.npz', allow_pickle=True); _L.update(c25=z['c25'], c2324=z['c2324'], v2324=z['v2324'], nv=int(z['nv']))


def cites_one(f):
    o = OUT / 'cites' / (f.name + '.npy')
    if o.exists(): return f.name, 'skip'
    _load(); c25, c2324, v2324 = _L['c25'], _L['c2324'], _L['v2324']; cnt = np.zeros(_L['nv'], np.int64); a = []; b = []; n = 0

    def flush():
        if not a: return
        A = np.array(a, np.int64); Bc = np.array(b, np.int64)
        i = np.searchsorted(c25, A); ok = (i < len(c25)) & (c25[np.minimum(i, len(c25) - 1)] == A); A, Bc = A[ok], Bc[ok]
        j = np.searchsorted(c2324, Bc); ok = (j < len(c2324)) & (c2324[np.minimum(j, len(c2324) - 1)] == Bc)
        np.add.at(cnt, v2324[j[ok]], 1); a.clear(); b.clear()
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"citingcorpusid":'); k2 = l.find(',', k); m = l.find('"citedcorpusid":'); m2 = l.find(',', m)
            try: x, y = int(l[k + 17:k2]), int(l[m + 16:m2])
            except ValueError: continue
            a.append(x); b.append(y); n += 1
            if len(a) >= 2_000_000: flush()
    flush(); np.save(o, cnt); return f.name, n


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'papers':
        with Pool(16) as p:
            for x in p.imap_unordered(papers_one, sorted((SNAP / 'papers').glob('*.gz'))): print(*x, flush=True)
    elif cmd == 'lookup':
        cid, yr, vid = [], [], []
        for f in sorted((OUT / 'papers').glob('*.npz')):
            z = np.load(f); cid.append(z['cid']); yr.append(z['yr']); vid.append(z['vid'])
        cid, yr, vid = np.concatenate(cid), np.concatenate(yr), np.concatenate(vid)
        venues, vix = np.unique(vid, return_inverse=True)
        c25 = np.unique(cid[yr == Y]); m = yr < Y; o = np.argsort(cid[m]); c2324 = cid[m][o]; v2324 = vix[m][o]
        np.savez(OUT / 'lookup.npz', c25=c25, c2324=c2324, v2324=v2324, nv=len(venues)); np.save(OUT / 'venues.npy', venues)
        np.save(OUT / 'n2324.npy', np.bincount(v2324, minlength=len(venues)))
        print('venues', len(venues), '2025 papers', len(c25), '2023-24 papers', len(c2324), flush=True)
    elif cmd == 'citations':
        with Pool(16) as p:
            for x in p.imap_unordered(cites_one, sorted((SNAP / 'citations').glob('*.gz'))): print(*x, flush=True)
    elif cmd == 'combine':
        venues = np.load(OUT / 'venues.npy', allow_pickle=True); n = np.load(OUT / 'n2324.npy'); c = sum(np.load(f) for f in (OUT / 'cites').glob('*.npy'))
        meta = {}
        for f in (SNAP / 'publication-venues').glob('*.gz'):
            for l in gzip.open(f, 'rt'):
                r = json.loads(l); meta[r['id']] = (r.get('name'), r.get('type'))
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.table({'venueid': venues.astype(str), 'name': [meta.get(v, (None, None))[0] for v in venues], 'type': [meta.get(v, (None, None))[1] for v in venues],
                                 'n_2023_24': n, 'cites_2025': c, 'if_2025': c / np.maximum(n, 1)}), OUT / 'venue_if.parquet')
        print('combined', len(venues), flush=True)
    else: sys.exit(__doc__)
