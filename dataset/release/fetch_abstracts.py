#!/usr/bin/env python3
"""Fill in the abstracts this release does not redistribute, check them against the published SHA-256, and build the text files.

The release ships abstracts verbatim only for papers whose abstract license is CC-BY or CC0 / public domain. For every other paper
papers.jsonl carries the Semantic Scholar corpus id, the title, the external ids and abstract_sha256. This script gets those
abstracts from one of two sources and writes the texts the benchmark uses:

  --snapshot DIR   the Semantic Scholar Academic Graph "abstracts" dataset files (*.gz, release 2026-09-09 gives exact matches)
  --api            the Semantic Scholar Graph API /paper/batch (set S2_API_KEY for the 1 request/s limit; newer or missing
                   abstracts are reported as hash mismatches)

  python3 fetch_abstracts.py --release DIR (--snapshot DIR | --api) [--out DIR]

Writes OUT/papers_full.jsonl (papers.jsonl with every abstract it could match), OUT/eval_texts.jsonl (if eval.jsonl is in DIR),
OUT/train_texts.jsonl (if train.jsonl is in DIR), and
prints how many abstracts matched, mismatched, or were missing. Text rule: title + '. ' + whitespace-normalized abstract; the eval
set cuts the abstract part at 2,000 characters. A paper whose abstract does not match its hash keeps the fetched text and is listed
in OUT/hash_mismatch.json."""
import argparse, glob, gzip, hashlib, json, os, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--release', required=True); ap.add_argument('--snapshot'); ap.add_argument('--api', action='store_true')
ap.add_argument('--out'); A = ap.parse_args()
R = Path(A.release); OUT = Path(A.out or A.release); OUT.mkdir(parents=True, exist_ok=True)
norm = lambda s: ' '.join((s or '').split()); sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
P = {}
for l in open(R / 'papers.jsonl'): x = json.loads(l); P[x['corpusid']] = x
need = {c for c, x in P.items() if x['abstract'] is None and x['abstract_sha256']}
print(f'papers {len(P)} | abstracts included {sum(1 for x in P.values() if x["abstract"])} | to fetch {len(need)}', flush=True)
got = {}
if A.snapshot:
    for f in sorted(glob.glob(os.path.join(A.snapshot, '*.gz'))):
        with gzip.open(f, 'rt') as g:
            for l in g:
                i = l.find(',"openaccessinfo"')
                try: c = int(l[12:i])
                except ValueError: continue
                if c in need: got[c] = json.loads(l).get('abstract')
        print(f, len(got), flush=True)
elif A.api:
    import requests
    H = {'x-api-key': os.environ['S2_API_KEY']} if os.environ.get('S2_API_KEY') else {}; ids = sorted(need)
    for i in range(0, len(ids), 500):
        ch = ids[i:i + 500]
        for t in range(8):
            r = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch', params={'fields': 'abstract'},
                              json={'ids': [f'CorpusId:{c}' for c in ch]}, headers=H, timeout=120)
            if r.status_code == 200: break
            time.sleep(2 * (t + 1))
        for c, x in zip(ch, r.json() if r.status_code == 200 else []): got[c] = (x or {}).get('abstract')
        time.sleep(1.05)
else: ap.error('give --snapshot or --api')
ok, bad, miss = 0, [], 0
for c in need:
    a = norm(got.get(c))
    if not a: miss += 1; continue
    P[c]['abstract'] = a
    if sha(a) == P[c]['abstract_sha256']: ok += 1
    else: bad.append(c)
with open(OUT / 'papers_full.jsonl', 'w') as f:
    for x in P.values(): f.write(json.dumps(x, ensure_ascii=False) + '\n')
json.dump(bad, open(OUT / 'hash_mismatch.json', 'w'))
def text(c, cut): x = P[c]; a = x['abstract'] or ''; return f"{x['title']}. {a[:cut] if cut else a}"
if (R / 'eval.jsonl').exists():
  with open(OUT / 'eval_texts.jsonl', 'w') as f:
    for l in open(R / 'eval.jsonl'):
        e = json.loads(l)
        f.write(json.dumps({'query_id': e['query_id'], 'query': e['query'], 'field': e['field'], 'pos': {'cid': e['positive']['corpusid'], 'text': text(e['positive']['corpusid'], 2000)},
                            'negs': [{'cid': n['corpusid'], 'stratum': n['stratum'], 'text': text(n['corpusid'], 2000)} for n in e['negatives']]}, ensure_ascii=False) + '\n')
if (R / 'train.jsonl').exists():
  with open(OUT / 'train_texts.jsonl', 'w') as f:
    for l in open(R / 'train.jsonl'):
        e = json.loads(l)
        f.write(json.dumps({'query_id': e['query_id'], 'query': e['query'], 'pos': [text(e['positive']['corpusid'], 0)],
                            'neg': [text(n['corpusid'], 0) for n in e['negatives']]}, ensure_ascii=False) + '\n')
print(f'fetched and matched {ok} | hash mismatch {len(bad)} | not found {miss} -> {OUT}')
