#!/usr/bin/env python3
"""PMCIDs (Europe PMC search, free) for base papers without a Semantic Scholar PubMedCentral id, by DOI or PMID: one search per 100
DOIs (DOI:"a" OR DOI:"b" ...) or 100 PMIDs, matched back by the returned doi / pmid, 8 parallel callers. Writes
<work>/stage2_input/parts/pmcid_lookup.json {q: pmcid or null} (resumable) and adds q['pmcid'] to papers.json.
  python -I pmcid_lookup.py CONFIG"""
import json, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, base_papers, stage2_dirs
C = load_config(sys.argv[1]); I, _ = stage2_dirs(C); OUT = I / 'parts/pmcid_lookup.json'; OUT.parent.mkdir(exist_ok=True)
P = json.load(open(I / 'papers.json')); Q = [str(c) for v in base_papers(C).values() for c in v if str(c) in P]
done = json.load(open(OUT)) if OUT.exists() else {}
todo = [q for q in Q if q not in done and not (P[q].get('externalIds') or {}).get('PubMedCentral')]
print('to look up', len(todo), flush=True)
_l = threading.Lock()
by_doi = [q for q in todo if (P[q].get('externalIds') or {}).get('DOI')]; sd = set(by_doi)
by_pmid = [q for q in todo if q not in sd and (P[q].get('externalIds') or {}).get('PubMed')]; sp = set(by_pmid)
for q in todo:
    if q not in sd and q not in sp: done[q] = None


def group(qs, kind):
    s = requests.Session()
    key = {str(P[q]['externalIds']['DOI' if kind == 'doi' else 'PubMed']).lower(): q for q in qs}
    query = ' OR '.join(f'DOI:"{k}"' for k in key) if kind == 'doi' else '(' + ' OR '.join(f'EXT_ID:{k}' for k in key) + ') AND SRC:MED'
    got = {}
    for k in range(4):
        try:
            r = s.get('https://www.ebi.ac.uk/europepmc/webservices/rest/search', params={'query': query, 'format': 'json', 'resultType': 'lite', 'pageSize': 1000}, timeout=120)
            if r.status_code == 200:
                for x in r.json().get('resultList', {}).get('result', []):
                    m = str(x.get('doi' if kind == 'doi' else 'pmid') or '').lower()
                    if m in key and x.get('pmcid'): got.setdefault(key[m], x['pmcid'])
                break
        except requests.RequestException: pass
        time.sleep(10 * (k + 1))
    else: return   # not served: left for a rerun
    with _l:
        for q in qs: done[q] = got.get(q)
        OUT.write_text(json.dumps(done)); print(len(done), 'found', sum(1 for v in done.values() if v), flush=True)


G = [(by_doi[i:i + 100], 'doi') for i in range(0, len(by_doi), 100)] + [(by_pmid[i:i + 100], 'pmid') for i in range(0, len(by_pmid), 100)]
with ThreadPoolExecutor(8) as ex: list(ex.map(lambda g: group(*g), G))
OUT.write_text(json.dumps(done))
for q, v in done.items():
    if v: P[q]['pmcid'] = v
json.dump(P, open(I / 'papers.json', 'w')); print('papers.json: pmcid for', sum(1 for v in done.values() if v), flush=True)
