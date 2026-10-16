#!/usr/bin/env python3
"""Release step 3: the id-only public release. Abstracts are included verbatim only when the paper's abstract license (Semantic Scholar
openaccessinfo) is CC-BY or CC0 / public domain; every other paper ships as corpus id + title + external ids + the SHA-256 of its
whitespace-normalized abstract, and fetch_abstracts.py (copied into OUT_DIR) fills the abstracts back in for the user.
Inputs: the exported eval / training files, the Stage 7 judgment dirs of the runs (judge_score of every negative), the universe export
shards (title + abstract), OUT_DIR/release_ids.json (collect_ids.py), OUT_DIR/release_externalids.jsonl (fetch_externalids.py).
Outputs in OUT_DIR:
  papers.jsonl  {corpusid, title, year, externalIds, roles [q/d/n], abstract_license, abstract_redistributable, abstract, abstract_sha256}
  eval.jsonl    {query_id, query, field, q_corpusid, q_year, positive {corpusid}, negatives [{corpusid, stratum, judge_score}]}
  train.jsonl   same fields as eval.jsonl
  fetch_abstracts.py
and prints a check: texts rebuilt with the release rule (title + '. ' + abstract; eval cut at 2,000 characters) against the exported texts.
  python -I build_release.py --eval EVAL.jsonl --train TRAIN.jsonl --judgments WORK/eval/s7 WORK/train/s7 --corpus $DATA/universe_2025 [...] --out OUT_DIR"""
import argparse, collections, glob, hashlib, json, shutil, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).resolve().parent)); from license_table import cls
ap = argparse.ArgumentParser(); ap.add_argument('--eval', required=True); ap.add_argument('--train', required=True)
ap.add_argument('--judgments', nargs='+', required=True); ap.add_argument('--corpus', nargs='+', required=True, help='universe dirs, first match wins')
ap.add_argument('--out', required=True); A = ap.parse_args()
OUT = Path(A.out); R = json.load(open(OUT / 'release_ids.json')); ids = R['ids']
roles = {int(k): v for k, v in R['roles'].items()}; lic = {int(k): v for k, v in R['license'].items()}
E = [json.loads(l) for l in open(A.eval)]; TR = [json.loads(l) for l in open(A.train)]; TM = [json.loads(l) for l in open(A.train.rsplit('.', 1)[0] + '.meta.jsonl')]
norm = lambda s: ' '.join((s or '').split())
TA = {}
for d in A.corpus:
    for f in sorted(glob.glob(str(Path(d) / 'export' / 'universe_*.parquet'))):
        t = pq.read_table(f, columns=['corpusid', 'title', 'abstract', 'year']); t = t.filter(pc.is_in(t.column('corpusid'), value_set=pa.array(ids, pa.int64())))
        for c, ti, ab, y in zip(*[t.column(k).to_pylist() for k in ('corpusid', 'title', 'abstract', 'year')]): TA.setdefault(c, (ti, ab, y))
def text(c, split): ti, ab, _ = TA.get(c, ('', '', None)); a = norm(ab); return f"{(ti or '').strip()}. {a[:2000] if split == 'eval' else a}"
X = {}
for l in open(OUT / 'release_externalids.jsonl'): x = json.loads(l); X[x['corpusid']] = x
used = {c for c, r in roles.items() if set(r) & {'d', 'n'}}; chk = collections.Counter()
with open(OUT / 'papers.jsonl', 'w') as f:
    for c in ids:
        ti, ab, y = TA.get(c, (None, None, None)); red = cls(lic.get(c)) == 'open' and bool(norm(ab))
        f.write(json.dumps({'corpusid': c, 'title': (ti or '').strip() or X.get(c, {}).get('title_api'), 'year': y,
                            'externalIds': {k: v for k, v in ((X.get(c) or {}).get('externalIds') or {}).items() if k != 'CorpusId'},
                            'roles': roles[c], 'abstract_license': lic.get(c), 'abstract_redistributable': red,
                            'abstract': norm(ab) if red and c in used else None,
                            'abstract_sha256': hashlib.sha256(norm(ab).encode()).hexdigest() if (c in used and norm(ab)) else None}, ensure_ascii=False) + '\n')
        chk['redistributable' if red and c in used else 'hash only' if c in used else 'q paper (metadata only)'] += 1
JD = [Path(d) for d in A.judgments]
def score(p, c):
    for d in JD:
        f = d / f'{p}__{c}.json'
        if f.exists(): return json.load(open(f))['score']
    return None
with open(OUT / 'eval.jsonl', 'w') as f:
    for e in E:
        negs = []
        for n in e['negs']:
            s = score(e['query_id'], n['cid']); chk['eval score missing'] += s is None
            negs.append({'corpusid': n['cid'], 'stratum': n['stratum'], 'judge_score': s})
            chk['eval text rebuild ok' if text(n['cid'], 'eval') == n['text'] else 'eval text rebuild differs'] += 1
        f.write(json.dumps({'query_id': e['query_id'], 'query': e['query'], 'field': e['field'], 'q_corpusid': int(e['query_id'].split('_')[-2]), 'q_year': e['q_year'],
                            'positive': {'corpusid': e['pos']['cid']}, 'negatives': negs}, ensure_ascii=False) + '\n')
with open(OUT / 'train.jsonl', 'w') as f:
    for r, m in zip(TR, TM):
        negs = []
        for s, c in m['neg']:
            sc = score(m['query_id'], c); chk['train score missing'] += sc is None
            negs.append({'corpusid': c, 'stratum': s, 'judge_score': sc})
        for c, t in zip([c for _, c in m['neg']] + [m['pos_cid']], r['neg'] + r['pos']): chk['train text rebuild ok' if text(c, 'train') == t else 'train text rebuild differs'] += 1
        f.write(json.dumps({'query_id': m['query_id'], 'query': r['query'], 'field': m['field'], 'q_corpusid': int(m['q_cid']), 'q_year': m['q_year'],
                            'positive': {'corpusid': m['pos_cid']}, 'negatives': negs}, ensure_ascii=False) + '\n')
shutil.copy(Path(__file__).resolve().parent / 'fetch_abstracts.py', OUT / 'fetch_abstracts.py')
print(json.dumps({**chk, 'papers': len(ids), 'eval': len(E), 'train': len(TR)}, indent=1))
