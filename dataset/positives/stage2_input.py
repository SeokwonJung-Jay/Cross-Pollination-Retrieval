#!/usr/bin/env python3
"""Stage 2 inputs for the base papers of a run (config q_pool), from the Semantic Scholar Academic Graph ($S2AG: papers, abstracts,
citations) and the Semantic Scholar Graph API (openAccessPdf only; S2_API_KEY optional). paperId = str(corpusid).
  references : every citation row whose citing paper is a q (all references, so Stage 2's parse ratio uses the full reference count),
               with the Semantic Scholar contexts and intents; one row per (q, d)
  papers     : q + every cited d: title, year, authors, externalIds, s2FieldsOfStudy, venue, whitespace-normalized abstract;
               q also get openAccessPdf
Each scan writes one part file per input file (resumable). Output: <work>/stage2_input/{papers.json, references.json}
  python -I stage2_input.py CONFIG"""
import gzip, json, os, sys, time
from pathlib import Path
from multiprocessing import Pool
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env, load_config, base_papers, stage2_dirs
C = load_config(sys.argv[1]); SNAP = env('S2AG'); OUT, _ = stage2_dirs(C); PARTS = OUT / 'parts'; PARTS.mkdir(parents=True, exist_ok=True)
Q = {c for v in base_papers(C).values() for c in v}


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def cit_one(f):
    out = PARTS / ('cit_' + f.name + '.jsonl')
    if out.exists(): return
    rows = []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"citingcorpusid":'); j = l.find(',', k)
            try: c = int(l[k + 17:j])
            except ValueError: continue
            if c in Q:
                r = json.loads(l)
                if r.get('citedcorpusid'): rows.append({'q': c, 'd': r['citedcorpusid'], 'contexts': r.get('contexts') or [], 'intents': r.get('intents') or []})
    tmp = out.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows)); tmp.replace(out)


def scan_one(a):
    f, kind, ids = a; out = PARTS / (f'{kind}_' + f.name + '.json')
    if out.exists(): return
    res = {}
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"corpusid":'); j = l.find(',', k)
            try: c = int(l[k + 11:j])
            except ValueError: continue
            if c in ids:
                r = json.loads(l)
                if kind == 'abstract': res[c] = ' '.join((r.get('abstract') or '').split())
                else: res[c] = {'title': r.get('title') or '', 'year': r.get('year'), 'authors': [{'name': x.get('name')} for x in r.get('authors') or []],
                                'externalIds': r.get('externalids') or {}, 's2FieldsOfStudy': r.get('s2fieldsofstudy') or [], 'venue': r.get('venue') or ''}
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(res)); tmp.replace(out)


if __name__ == '__main__':
    log('q', len(Q))
    cf = sorted((SNAP / 'citations').glob('*.gz'))
    with Pool(24) as p:
        for i, _ in enumerate(p.imap_unordered(cit_one, cf), 1):
            if i % 20 == 0: log(f'citations {i}/{len(cf)}')
    refs = {}
    for f in PARTS.glob('cit_*.jsonl'):
        for l in open(f):
            r = json.loads(l); refs.setdefault(r['q'], {}).setdefault(r['d'], r)   # one row per (q, d)
    ids = Q | {d for v in refs.values() for d in v}
    log(f'q with refs {len(refs)} | ref rows {sum(len(v) for v in refs.values()):,} | papers to fetch {len(ids):,}')
    with Pool(24) as p:
        p.map(scan_one, [(f, 'paper', ids) for f in sorted((SNAP / 'papers').glob('*.gz'))]); log('papers scanned')
        p.map(scan_one, [(f, 'abstract', ids) for f in sorted((SNAP / 'abstracts').glob('*.gz'))]); log('abstracts scanned')
    P = {}
    for f in PARTS.glob('paper_*.json'):
        for c, v in json.load(open(f)).items(): P[int(c)] = v
    for f in PARTS.glob('abstract_*.json'):
        for c, v in json.load(open(f)).items():
            if int(c) in P: P[int(c)]['abstract'] = v
    # openAccessPdf of q (PDF fallback of Stage 2) from the Graph API
    oaf = PARTS / 'oa_pdf.json'; oa = json.load(open(oaf)) if oaf.exists() else {}
    key = os.environ.get('S2_API_KEY', ''); todo = sorted(c for c in Q if str(c) not in oa)
    for i in range(0, len(todo), 500):
        chunk = todo[i:i + 500]
        for attempt in range(5):
            r = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch', params={'fields': 'corpusId,openAccessPdf,externalIds'},
                              json={'ids': [f'CorpusId:{c}' for c in chunk]}, headers={'x-api-key': key} if key else {}, timeout=120)
            if r.status_code == 200: break
            log('paper/batch', r.status_code, 'wait 30 s'); time.sleep(30)
        r.raise_for_status()
        for c, x in zip(chunk, r.json()): oa[str(c)] = (x or {}).get('openAccessPdf') or None
        oaf.write_text(json.dumps(oa)); time.sleep(1.1)
    log(f'openAccessPdf url for {sum(1 for c in Q if (oa.get(str(c)) or {}).get("url"))} / {len(Q)} q')
    papers = {}
    for c, v in P.items():
        rec = {'paperId': str(c), **v}
        if c in Q and oa.get(str(c)): rec['openAccessPdf'] = oa[str(c)]
        papers[str(c)] = rec
    references = {str(q): [{'citedPaper': {'paperId': str(d)}, 'contexts': r['contexts'], 'intents': r['intents']} for d, r in v.items()] for q, v in refs.items()}
    json.dump(papers, open(OUT / 'papers.json', 'w')); json.dump(references, open(OUT / 'references.json', 'w'))
    log(f'written: papers {len(papers):,} | q with abstract {sum(1 for c in Q if papers.get(str(c), {}).get("abstract"))}')
