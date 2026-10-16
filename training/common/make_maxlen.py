#!/usr/bin/env python3
"""Cut every query / positive / negative to its first MAXLEN-1 Qwen3 tokens (at a token boundary, via offsets), so with the
appended <|endoftext|> each text is at most MAXLEN tokens - the same as sentence-transformers at eval (max_seq_length MAXLEN).
ms-swift applies max_length per text, but truncation 'delete' drops whole rows and 'right' would also cut the final
<|endoftext|> (the pooled token), so the data is cut here instead (training then also passes --max_length MAXLEN).
  python -I make_maxlen.py SRC_DATA_DIR DST_DATA_DIR [MAXLEN=1024]   (SRC/<neg>/<data>/train.jsonl -> DST/<neg>/<data>/train.jsonl + stats.json)"""
import json, os, sys
from transformers import AutoTokenizer
src, dst = sys.argv[1:3]; K = int(sys.argv[3] if len(sys.argv) > 3 else 1024) - 1
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')   # 0.6B and 4B share the tokenizer
def cut(texts):
    enc = tk(texts, add_special_tokens=False, return_offsets_mapping=True); out, n = [], 0
    for t, ids, off in zip(texts, enc['input_ids'], enc['offset_mapping']):
        if len(ids) > K: t = t[:off[K - 1][1]]; n += 1
        out.append(t)
    return out, n
for neg in ('hardneg', 'inbatch'):
    for data in ('ours', 'msmarco', 'reasonir', 'specter'):
        p = f'{src}/{neg}/{data}/train.jsonl'; rows = [json.loads(l) for l in open(p)]; st = {'rows': len(rows), 'cut': {}}
        for key in ('query', 'pos', 'neg'):
            flat = [(i, j, x) for i, r in enumerate(rows) for j, x in enumerate([r[key]] if isinstance(r.get(key), str) else r.get(key, []))]
            new, n = cut([x for _, _, x in flat]) if flat else ([], 0); st['cut'][key] = f'{n}/{len(flat)}'
            for (i, j, _), x in zip(flat, new):
                if isinstance(rows[i][key], str): rows[i][key] = x
                else: rows[i][key][j] = x
        os.makedirs(f'{dst}/{neg}/{data}', exist_ok=True)
        with open(f'{dst}/{neg}/{data}/train.jsonl', 'w') as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        json.dump(st, open(f'{dst}/{neg}/{data}/stats.json', 'w')); print(neg, data, st, flush=True)
