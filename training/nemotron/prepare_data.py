#!/usr/bin/env python3
"""Convert our training data to NeMo AutoModel's inline retrieval JSONL: {"query": "Query:" + query, "pos_doc": [positive], "neg_doc": [negatives]}
(same rows and order; query_id copied when present, NeMo does not read it). Prints the number of rows and of negatives per row.
The query gets exactly the prompt of the Qwen runs, 'Query:' with no space (the collator's query_prefix would add a space, so the prompt is
part of the text and the prefix stays empty). Documents get no prompt. Texts are not cut here: the NeMo collator truncates every query /
document to 1024 Llama tokens (the same budget as sentence-transformers at evaluation).
  python -I prepare_data.py SRC_train.jsonl DST.jsonl"""
import json, sys
src, dst = sys.argv[1:3]; n, negs = 0, set()
with open(src, encoding='utf-8') as f, open(dst, 'w') as out:
    for l in f:
        r = json.loads(l); assert len(r['pos']) == 1, n
        o = {'query_id': r['query_id']} if 'query_id' in r else {}
        out.write(json.dumps({**o, 'query': 'Query:' + r['query'], 'pos_doc': r['pos'], 'neg_doc': r['neg']}, ensure_ascii=False) + '\n')
        n += 1; negs.add(len(r['neg']))
assert len(negs) == 1, f'rows with different numbers of negatives: {sorted(negs)}'
print('rows', n, 'negatives per row', negs.pop())
