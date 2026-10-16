#!/usr/bin/env python3
"""Prefetch the JATS XML of every base paper with a PMCID (Semantic Scholar PubMedCentral id or pmcid_lookup.py) into
<stage2_cache>/jats_cache/. The PMC Open Access copy on AWS (pmc-oa-opendata, latest version PMCxxx.N/PMCxxx.N.xml) is tried first,
then Europe PMC (fullTextXML); --no-s3 uses Europe PMC only. Threads with one requests.Session each; cached and resumable.
  python -I jats_prefetch.py CONFIG [--threads 64] [--no-s3]"""
import argparse, collections, json, re, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, base_papers, stage2_dirs
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stage2')); import _jats
ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--threads', type=int, default=64); ap.add_argument('--no-s3', action='store_true')
ap.add_argument('--min-interval', type=float, default=0.0); A = ap.parse_args()
C = load_config(A.config); I, CA = stage2_dirs(C); J = CA / 'jats_cache'
P = json.load(open(I / 'papers.json')); L = I / 'parts/pmcid_lookup.json'; look = json.load(open(L)) if L.exists() else {}
ids = []
for c in (str(c) for v in base_papers(C).values() for c in v if str(c) in P):
    p = (P[c].get('externalIds') or {}).get('PubMedCentral') or P[c].get('pmcid') or look.get(c)
    if p: ids.append(p)
TL = threading.local()


def sess():
    if getattr(TL, 's', None) is None: TL.s = requests.Session(); TL.s.mount('https://', HTTPAdapter(max_retries=3))
    return TL.s


def s3(p):
    p = 'PMC' + str(p).upper().removeprefix('PMC'); f = J / f'{p}.xml'
    if f.exists() or (J / f'{p}.missing').exists(): return
    r = sess().get('https://pmc-oa-opendata.s3.amazonaws.com/', params={'list-type': '2', 'prefix': p + '.', 'delimiter': '/'}, timeout=60)
    vs = sorted({int(v) for v in re.findall(rf'<Prefix>{p}\.(\d+)/</Prefix>', r.text)}) if r.status_code == 200 else []
    if not vs: return   # not in the AWS copy: Europe PMC decides
    r = sess().get(f'https://pmc-oa-opendata.s3.amazonaws.com/{p}.{vs[-1]}/{p}.{vs[-1]}.xml', timeout=60)
    if r.status_code == 200 and b'<article' in r.content[:3000]:
        tmp = f.with_suffix('.tmp'); tmp.write_bytes(r.content); tmp.replace(f)


def one(p):
    try:
        if not A.no_s3: s3(p)
        _jats.fetch(p, J, session=sess(), min_interval=A.min_interval); return 'ok'
    except _jats.JatsUnavailable as e: return str(e)
    except Exception as e: return type(e).__name__


res = collections.Counter()
with ThreadPoolExecutor(A.threads) as ex:
    for i, r in enumerate(ex.map(one, ids), 1):
        res[r] += 1
        if i % 500 == 0: print(i, '/', len(ids), dict(res), flush=True)
print('done', len(ids), dict(res), flush=True)
