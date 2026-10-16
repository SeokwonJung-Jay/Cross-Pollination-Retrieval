#!/usr/bin/env python3
"""Export the eval set: one JSONL line per query with its positive and 150 negatives (50 per group); the fields of eval_texts.jsonl plus q_year and source:
  {query_id, query, field, q_year, source, pos: {cid, text}, negs: [{cid, stratum, text}]}
query_id = pair id (<prefix>_<q corpusid>_<positive corpusid>), query = Stage 5 query, field = the base paper's OpenAlex field (config
q_pool), text = title + '. ' + whitespace-normalized abstract[:2000] (the text the Stage 7 judge saw). Per field, selected queries
(<work>/eval_selection.json) with a full pool come first, then spares with a full pool replace the others, up to queries_per_field.
Also writes <work>/eval_positives.json (positive corpus ids; the training config leaves out Stage 2 pairs citing them) and
<work>/eval_base_papers.json (every base paper this run processed; the training config skips them), and prints short fields.
  python -I export_eval.py CONFIG OUT.jsonl"""
import json, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, records
C = load_config(sys.argv[1]); W = Path(C['work']); N = C['queries_per_field']; NEG = C['s7_per_stratum']; STRATA = ('q_side', 'pos_side', 'other')
pools = json.load(open(W / 's7' / C.get('s7_pools_name', '_pools_main.json'))); sel = json.load(open(W / 'eval_selection.json')); S3 = records(W / 's3')
full = lambda p: p in pools and all(len(pools[p]['main'][s]) >= NEG for s in STRATA)
keep = {f: ([p for p in v['selected'] if full(p)] + [p for p in v['spares'] if full(p)])[:N] for f, v in sel.items()}
T = json.load(open(W / 's7_texts.json')); need = sorted({int(pools[p]['pos_cid']) for v in keep.values() for p in v} - {int(k) for k in T})
if need:   # positives are not in the Stage 7 text cache: same formatting from the universe shards
    for f in sorted((Path(C['corpus']) / 'export').glob('universe_*.parquet')):
        ids = pq.read_table(f, columns=['corpusid']).column(0).to_numpy(); rr = np.flatnonzero(np.isin(ids, need))
        if not len(rr): continue
        t = pq.read_table(f, columns=['corpusid', 'title', 'abstract']).take(pa.array(rr))
        for c, ti, ab in zip(t.column('corpusid').to_pylist(), t.column('title').to_pylist(), t.column('abstract').to_pylist()):
            T[str(c)] = f"{ti}. {' '.join((ab or '').split())[:2000]}"
n = 0
with open(sys.argv[2], 'w') as fo:
    for f, ps in keep.items():
        for p in ps:
            x = pools[p]; negs = [{'cid': int(c), 'stratum': st, 'text': T[str(c)]} for st in STRATA for c in x['main'][st][:NEG]]
            fo.write(json.dumps({'query_id': p, 'query': x['query'], 'field': f, 'q_year': S3[p]['source']['q'].get('year'), 'source': C['name'],
                                 'pos': {'cid': int(x['pos_cid']), 'text': T[str(x['pos_cid'])]}, 'negs': negs}, ensure_ascii=False) + '\n'); n += 1
json.dump(sorted({int(pools[p]['pos_cid']) for v in keep.values() for p in v}), open(W / 'eval_positives.json', 'w'))
json.dump(sorted({int(c) for v in json.load(open(W / 'stage2' / 'plan.json'))['per_field'].values() for c in v}), open(W / 'eval_base_papers.json', 'w'))   # every q this run processed
print('queries', n, '| short fields', {f: len(v) for f, v in keep.items() if len(v) < N})
