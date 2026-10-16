#!/usr/bin/env python3
"""OpenAlex topics of every query paper and positive paper (train + eval; hard negatives are not looked up).
Semantic Scholar corpus id -> externalIds (S2 paper/batch, 500 ids per request) -> OpenAlex work by DOI, else PMID, else MAG
(50 per request) -> primary topic + top-3 topics, each with subfield / field / domain.
Resumable caches OUT/{s2_ids.json, openalex.json}; final OUT/paper_domains.jsonl, one row per paper:
  {cid, openalex, primary: {topic, subfield, field, domain, score} | null, topics: [...top 3], via: DOI | PubMed | MAG}.
Optional environment: SEMANTIC_SCHOLAR_API_KEY (S2 requests only), OPENALEX_MAILTO (contact address in the User-Agent).
  python -I fetch_openalex.py --pairs PAIRS --out META      (PAIRS/{train,test}.jsonl from prep_pairs.py)"""
import argparse, json, os, time
import requests

ap = argparse.ArgumentParser(); ap.add_argument('--pairs', required=True); ap.add_argument('--out', required=True); A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
mail = os.environ.get('OPENALEX_MAILTO'); H = {'User-Agent': 'cpr-dataset domain lookup' + (f' (mailto:{mail})' if mail else '')}
ids = set()
for sp in ('train', 'test'):
    for l in open(f'{A.pairs}/{sp}.jsonl'):
        r = json.loads(l); ids |= {str(r['q_cid']), str(r['pos_cid'])}
ids = sorted(ids); print('papers', len(ids), flush=True)
load = lambda p: json.load(open(p)) if os.path.exists(p) else {}
def save(o, p): json.dump(o, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)

def get(method, url, **kw):
    for a in range(10):
        try:
            k = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
            r = requests.request(method, url, headers={**H, 'x-api-key': k} if k and 'semanticscholar' in url else H, timeout=90, **kw)
            if r.status_code == 200: return r.json()
            print('HTTP', r.status_code, url[:80], flush=True)
        except Exception as e: print('ERR', e, flush=True)
        time.sleep(min(60, 3 * 2 ** a))
    raise SystemExit('giving up: ' + url[:120])

P = f'{A.out}/s2_ids.json'; s2 = load(P); todo = [c for c in ids if c not in s2]
for i in range(0, len(todo), 500):
    ch = todo[i:i + 500]
    res = get('POST', 'https://api.semanticscholar.org/graph/v1/paper/batch', params={'fields': 'externalIds'}, json={'ids': [f'CorpusId:{c}' for c in ch]})
    for c, x in zip(ch, res): s2[c] = (x or {}).get('externalIds') or {}
    save(s2, P); print('s2', i + len(ch), '/', len(todo), flush=True); time.sleep(1.5)

def oa_rec(w):
    t = lambda x: {'topic': x['display_name'], 'subfield': x['subfield']['display_name'], 'field': x['field']['display_name'], 'domain': x['domain']['display_name'], 'score': x.get('score')}
    return {'openalex': w['id'].rsplit('/', 1)[1], 'primary': t(w['primary_topic']) if w.get('primary_topic') else None, 'topics': [t(x) for x in (w.get('topics') or [])[:3]]}
P2 = f'{A.out}/openalex.json'; oa = load(P2)
for key, flt, norm in (('DOI', 'doi', lambda v: v.lower()), ('PubMed', 'ids.pmid', str), ('MAG', 'ids.mag', str)):
    todo = [c for c in ids if c not in oa and s2.get(c, {}).get(key)]
    print('openalex by', key, len(todo), flush=True)
    for i in range(0, len(todo), 50):
        ch = todo[i:i + 50]; want = {norm(s2[c][key]): c for c in ch}
        res = get('GET', 'https://api.openalex.org/works', params={'filter': f'{flt}:' + '|'.join(want), 'per-page': 200, 'select': 'id,doi,ids,primary_topic,topics'})
        for w in res.get('results', []):
            ix = w.get('ids') or {}
            k = {'doi': (w.get('doi') or '').replace('https://doi.org/', '').lower(), 'ids.pmid': (ix.get('pmid') or '').rsplit('/', 1)[-1], 'ids.mag': str(ix.get('mag') or '')}[flt]
            if k in want and want[k] not in oa: oa[want[k]] = dict(oa_rec(w), via=key)
        if (i // 50) % 20 == 0: save(oa, P2); print(' ', key, i + len(ch), '/', len(todo), flush=True)
        time.sleep(0.12)
    save(oa, P2)
with open(f'{A.out}/paper_domains.jsonl', 'w') as f:
    for c in ids: f.write(json.dumps({'cid': c, **(oa.get(c) or {'openalex': None, 'primary': None, 'topics': []})}) + '\n')
print('done', len(ids), '| with primary_topic', sum(1 for c in ids if (oa.get(c) or {}).get('primary')), '| no s2 ids', sum(1 for c in ids if not s2.get(c)))
