#!/usr/bin/env python3
"""OpenAlex topic tags for Stage 2: for every paper in <work>/stage2_input/papers.json, its OpenAlex topics (rank 1-3, OpenAlex order,
from <corpus>/universe_f.parquet) with names and subfields from the topic catalog, in the tag format stage2_candidate_preparation.py
reads (--tags). Papers without OpenAlex topics in the universe get no tag (Stage 2 then skips them).
Output: <work>/stage2_input/tags_openalex.jsonl
  python -I openalex_tags.py CONFIG"""
import json, sys
from pathlib import Path
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, stage2_dirs, topic_catalog
C = load_config(sys.argv[1]); S, _ = stage2_dirs(C); cat = topic_catalog()
ids = [int(k) for k in json.load(open(S / 'papers.json'))]
t = pq.read_table(Path(C['corpus']) / 'universe_f.parquet', columns=['corpusid', 'topics'])
t = t.filter(pc.is_in(t.column('corpusid'), value_set=pa.array(ids, pa.int64())))
n = miss = 0
with open(S / 'tags_openalex.jsonl', 'w') as fo:
    for c, tp in zip(t.column('corpusid').to_pylist(), t.column('topics').to_pylist()):
        tp = [x for x in (tp or []) if x in cat][:3]
        if not tp: miss += 1; continue
        subs, seen = [], set()
        for x in tp:
            s = cat[x]['subfield']; sid = s['id'].rsplit('/', 1)[1]
            if sid not in seen:
                seen.add(sid); subs.append({'rank': len(subs) + 1, 'id': sid, 'name': s['display_name'], 'field': cat[x]['field']['display_name'], 'domain': cat[x]['domain']['display_name']})
        fo.write(json.dumps({'id': str(c), 'source': 'openalex', 'status': 'ok', 'topic_status': 'ok',
                             'topics': [{'rank': i + 1, 'id': x, 'name': cat[x]['display_name']} for i, x in enumerate(tp)], 'subfields': subs}) + '\n'); n += 1
print('written', n, '| ids without OpenAlex topics', len(ids) - n, '(in the universe without topics:', miss, ')')
