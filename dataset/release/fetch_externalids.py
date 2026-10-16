#!/usr/bin/env python3
"""Release step 2: external ids (DOI, PubMed, PubMedCentral, ArXiv, ...) and title of every release paper from the Semantic Scholar Graph
API /paper/batch (500 ids per call; S2_API_KEY from the environment; 4 calls in flight, starts spaced 1.05 s apart). Incremental: one
JSON line per answered id; a rerun resumes.
  python -I fetch_externalids.py OUT_DIR/release_ids.json OUT_DIR/release_externalids.jsonl"""
import json, os, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import common  # noqa: F401  (.env)
IDS, OUT = sys.argv[1], Path(sys.argv[2])
ids = json.load(open(IDS))['ids']; done = set()
if OUT.exists():
    for l in open(OUT): done.add(json.loads(l)['corpusid'])
todo = [c for c in ids if c not in done]; print('todo', len(todo), 'done', len(done), flush=True)
H = {'x-api-key': os.environ['S2_API_KEY']} if os.environ.get('S2_API_KEY') else {}
lock = threading.Lock(); gate = threading.Lock(); last = [0.0]; fo = open(OUT, 'a'); n = [0]


def one(i):
    chunk = todo[i:i + 500]; r = None
    for t in range(8):
        with gate:
            w = last[0] + 1.05 - time.time()
            if w > 0: time.sleep(w)
            last[0] = time.time()
        try:
            r = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch', params={'fields': 'externalIds,title,year'},
                              json={'ids': [f'CorpusId:{c}' for c in chunk]}, headers=H, timeout=120)
            if r.status_code == 200: break
        except Exception: r = None
        time.sleep(2 * (t + 1))
    if r is None or r.status_code != 200: print('failed chunk', i, getattr(r, 'status_code', None), flush=True); return
    with lock:
        for c, x in zip(chunk, r.json()):
            fo.write(json.dumps({'corpusid': c, 'externalIds': (x or {}).get('externalIds'), 'title_api': (x or {}).get('title'), 'found': x is not None}) + '\n')
        fo.flush(); n[0] += len(chunk)


with ThreadPoolExecutor(4) as ex: list(ex.map(one, range(0, len(todo), 500)))
print('done', n[0], flush=True)
