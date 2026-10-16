#!/usr/bin/env python3
"""Export the training set: one JSONL line per final pair, {query_id, query, pos: [text], neg: [text x NEG]} (training/ format), and
OUT.meta.jsonl {query_id, pos_cid, neg: [[group, cid]], q_cid, field, q_year}.
query = Stage 5 query; negatives = per group (q_side, pos_side, other) the first NEG // 3 usable negatives (Stage 7, u <= 1) in judge
order whose text differs from the positive and from the negatives already taken; pairs with fewer than NEG negatives are left out.
Text = title + '. ' + whitespace-normalized full abstract (<corpus>/export).
--exclude-eval EVAL.jsonl (export_eval.py output) drops every pair whose base paper, positive or any negative is a base paper or a
positive of the eval set, so the two splits share none of these papers.
  python -I export_train.py CONFIG OUT.jsonl [--neg 9] [--exclude-eval EVAL.jsonl]"""
import argparse, collections, glob, json, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, records
ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('out'); ap.add_argument('--neg', type=int, default=9); ap.add_argument('--exclude-eval'); A = ap.parse_args()
C = load_config(A.config); W = Path(C['work'])
pools = json.load(open(W / 's7' / C.get('s7_pools_name', '_pools_main.json'))); final = json.load(open(W / 'final_pairs.json'))
FIELD = {str(c): f for f, v in json.load(open(C['q_pool']))['per_field'].items() for c in v}
ban = set()
if A.exclude_eval:
    for l in open(A.exclude_eval):
        e = json.loads(l); ban |= {int(e['query_id'].split('_')[-2]), int(e['pos']['cid'])}
rows, short, banned = [], 0, 0
for pid in sorted(final):
    p = pools.get(pid)
    if not p: short += 1; continue
    src = json.load(open(W / 's3' / f'{pid}.json'))['source']
    rows.append({'pid': pid, 'query': json.load(open(W / 's5' / 'stage5_v1a' / f'{pid}.json'))['rewritten_problem'].strip(), 'pos_cid': int(p['pos_cid']),
                 'negs': [(s, c) for s in ('q_side', 'pos_side', 'other') for c in (p.get('usable') or p['main'])[s]], 'q_cid': str(src['base_id']), 'q_year': src['q'].get('year')})
need = sorted({r['pos_cid'] for r in rows} | {c for r in rows for _, c in r['negs']}); TA = {}
for f in sorted(glob.glob(str(Path(C['corpus']) / 'export' / 'universe_*.parquet'))):
    t = pq.read_table(f, columns=['corpusid', 'title', 'abstract']); t = t.filter(pc.is_in(t.column('corpusid'), value_set=pa.array(need, pa.int64())))
    for c, ti, ab in zip(t.column('corpusid').to_pylist(), t.column('title').to_pylist(), t.column('abstract').to_pylist()):
        TA[c] = f"{(ti or '').strip()}. {' '.join((ab or '').split())}"
out, meta = open(A.out, 'w'), open(A.out.rsplit('.', 1)[0] + '.meta.jsonl', 'w'); n = 0; fields = collections.Counter()
for r in rows:
    if r['pos_cid'] not in TA: continue
    pos = TA[r['pos_cid']]; seen = {pos}; pick = []
    for s in ('q_side', 'pos_side', 'other'):
        k = 0
        for st, c in r['negs']:
            if st != s or k >= A.neg // 3 or c not in TA or TA[c] in seen: continue
            seen.add(TA[c]); pick.append((st, c)); k += 1
    if len(pick) < A.neg: short += 1; continue
    if ban & ({int(r['q_cid']), r['pos_cid']} | {c for _, c in pick}): banned += 1; continue
    out.write(json.dumps({'query_id': r['pid'], 'query': r['query'], 'pos': [pos], 'neg': [TA[c] for _, c in pick]}, ensure_ascii=False) + '\n')
    meta.write(json.dumps({'query_id': r['pid'], 'pos_cid': r['pos_cid'], 'neg': [[s, c] for s, c in pick], 'q_cid': r['q_cid'], 'field': FIELD.get(r['q_cid']), 'q_year': r['q_year']}) + '\n')
    n += 1; fields[FIELD.get(r['q_cid'])] += 1
print(f'finals {len(final)} | written {n} | fewer than {A.neg} negatives {short} | dropped for eval overlap {banned} | fields {len(fields)} -> {A.out}')
