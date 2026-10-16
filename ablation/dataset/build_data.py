#!/usr/bin/env python3
"""Comparison training sets for the dataset ablation: MS MARCO / ReasonIR / SPECTER with as many rows as our training set.
Output format of training/: {"query", "pos": [1], "neg": [...]}.
  specter  : allenai/scirepeval cite_prediction (citation triplets: query paper, cited paper, 1 negative). Text = "title. abstract";
             rows with an empty title / abstract are left out
  reasonir : reasonir/reasonir-data hq + vl pooled, 1 synthetic hard negative each. hq positives are BRIGHT document ids, filled from
             xlangai/BRIGHT documents (the dataset card's script); hq therefore overlaps the BRIGHT corpus
  msmarco  : microsoft/ms_marco v1.1 train, queries with exactly 1 selected passage and >= 9 unselected; neg = the first 9 unselected passages
Eligible rows: non-empty query / pos / negs, pos not among negs, negs distinct. Sampling: random.Random(seed).sample(eligible indices, N),
written in index order -> OUT/<name>/train.jsonl + meta.jsonl (source index).
  python -I build_data.py --raw RAW_DIR --bright BRIGHT_DIR --out OUT_DIR (--n-from OURS_TRAIN.jsonl | --n N) [--seed 42]
RAW_DIR holds the HF dataset snapshots as <org>__<name>/ (allenai__scirepeval, reasonir__reasonir-data, microsoft__ms_marco)."""
import argparse, glob, json, os, random
import pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument('--raw', required=True); ap.add_argument('--bright', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n-from'); ap.add_argument('--n', type=int); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--sets', nargs='+', default=['msmarco', 'reasonir', 'specter']); A = ap.parse_args()
N = A.n or sum(1 for l in open(A.n_from) if l.strip()); print('N', N, flush=True)
def write(name, rows, src):
    d = os.path.join(A.out, name); os.makedirs(d, exist_ok=True)
    with open(f'{d}/train.jsonl', 'w') as f, open(f'{d}/meta.jsonl', 'w') as m:
        for r, s in zip(rows, src): f.write(json.dumps(r, ensure_ascii=False) + '\n'); m.write(json.dumps(s) + '\n')
    print(name, len(rows), 'rows, neg', sorted({len(r['neg']) for r in rows}), flush=True)
sample = lambda eligible: sorted(random.Random(A.seed).sample(eligible, N))
ok = lambda q, pos, neg: q.strip() and pos.strip() and all(t.strip() for t in neg) and pos not in neg and len(set(neg)) == len(neg)
if 'specter' in A.sets:
    ta = lambda d: f"{(d.get('title') or '').strip()}. {' '.join((d.get('abstract') or '').split())}" \
        if (d.get('title') or '').strip() and (d.get('abstract') or '').strip() else ''
    rows = []
    for f in sorted(glob.glob(f'{A.raw}/allenai__scirepeval/cite_prediction/train-*.parquet')):
        rows += [(ta(r['query']), ta(r['pos']), ta(r['neg'])) for r in pq.read_table(f, columns=['query', 'pos', 'neg']).to_pylist()]
    el = [i for i, (q, p, n) in enumerate(rows) if ok(q, p, [n])]; print('specter rows', len(rows), 'eligible', len(el))
    pick = sample(el); write('specter', [{'query': rows[i][0], 'pos': [rows[i][1]], 'neg': [rows[i][2]]} for i in pick], [{'cite_prediction_train_idx': i} for i in pick]); del rows
if 'reasonir' in A.sets:
    id2doc = {}
    for f in sorted(glob.glob(f'{A.bright}/documents_*.parquet') + glob.glob(f'{A.bright}/documents/*.parquet')):
        t = pq.read_table(f, columns=['id', 'content']); id2doc.update(zip(map(str, t.column('id').to_pylist()), t.column('content').to_pylist()))
    assert id2doc, f'no BRIGHT documents under {A.bright}'
    rows = []
    for sub in ('hq', 'vl'):
        for j, r in enumerate(pq.read_table(f'{A.raw}/reasonir__reasonir-data/{sub}/train-00000-of-00001.parquet').to_pylist()):
            q, pos, neg = r['query'][-1], r['pos'][0][1], r['neg'][0][1]
            rows.append((sub, j, q, id2doc.get(pos, '') if sub == 'hq' else pos, neg))
    el = [i for i, (_, _, q, p, n) in enumerate(rows) if ok(q, p, [n])]
    print('reasonir rows', len(rows), 'eligible', len(el), '| hq unresolved/invalid', sum(r[0] == 'hq' for r in rows) - sum(rows[i][0] == 'hq' for i in el))
    pick = sample(el); write('reasonir', [{'query': rows[i][2], 'pos': [rows[i][3]], 'neg': [rows[i][4]]} for i in pick], [{'subset': rows[i][0], 'idx': rows[i][1]} for i in pick])
    print('reasonir picked hq', sum(rows[i][0] == 'hq' for i in pick), 'vl', sum(rows[i][0] == 'vl' for i in pick)); del rows
if 'msmarco' in A.sets:
    rows = []
    for r in pq.read_table(f'{A.raw}/microsoft__ms_marco/v1.1/train-00000-of-00001.parquet', columns=['query', 'query_id', 'passages']).to_pylist():
        sel, txt = r['passages']['is_selected'], [' '.join(x.split()) for x in r['passages']['passage_text']]
        rows.append((r['query'].strip(), r['query_id'], [x for s, x in zip(sel, txt) if s == 1], [x for s, x in zip(sel, txt) if s != 1][:9]))
    el = [i for i, (q, _, p, n) in enumerate(rows) if len(p) == 1 and len(n) == 9 and ok(q, p[0], n)]; print('msmarco rows', len(rows), 'eligible', len(el))
    pick = sample(el); write('msmarco', [{'query': rows[i][0], 'pos': rows[i][2], 'neg': rows[i][3]} for i in pick], [{'query_id': rows[i][1], 'train_idx': i} for i in pick])
