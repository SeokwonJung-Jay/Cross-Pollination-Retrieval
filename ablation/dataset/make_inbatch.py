#!/usr/bin/env python3
"""In-batch-only training sets: the hard-negative sets with every row's negatives removed (same rows, same order).
  python -I make_inbatch.py DATA_DIR   (DATA_DIR/hardneg/<set>/train.jsonl -> DATA_DIR/inbatch/<set>/train.jsonl)"""
import json, os, sys
root = sys.argv[1]
for name in ('ours', 'msmarco', 'reasonir', 'specter'):
    os.makedirs(f'{root}/inbatch/{name}', exist_ok=True)
    with open(f'{root}/hardneg/{name}/train.jsonl', encoding='utf-8') as f, open(f'{root}/inbatch/{name}/train.jsonl', 'w', encoding='utf-8') as out:
        for line in f:
            r = json.loads(line); r['neg'] = []
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(name, 'done')
