"""Pre-training data check used by qwen/train.sh and nemotron/train.sh. Exits 1 on any error.

Input: one JSONL file or a directory of .jsonl/.json files. Row = {"query": str, "pos": [str], "neg": [str, ...]} (other fields ignored).

    python validate_data.py DATA --min-neg 9 --global-batch 128 [--tokenizer NAME --query-prefix '...' --query-max-len 32]

Errors: JSON parse, field types / empty strings, pos not exactly one (ms-swift allows one positive), fewer than --min-neg negatives,
different negative counts across rows, pos text among its negatives, fewer rows than --global-batch (drop_last -> 0 steps).
Warnings: repeated queries, a positive shared by several queries (in-batch false negatives), duplicate negatives in a row.
With --tokenizer, prints query / document token-length stats and the share above --query-max-len / --doc-max-len.
"""
import argparse
import collections
import json
import os
import statistics
import sys


def data_files(path):
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.endswith(('.jsonl', '.json')) and os.path.isfile(os.path.join(path, f)))
        if not files:
            sys.exit(f'error: no .jsonl/.json files in {path}')
        return files
    if not path.endswith(('.jsonl', '.json')):
        sys.exit(f'error: not a .jsonl/.json file: {path}')
    return [path]


def length_report(name, lengths, limit):
    lengths = sorted(lengths)
    q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    over = sum(l > limit for l in lengths) if limit else 0
    msg = f'  {name}: n={len(lengths)} mean={statistics.mean(lengths):.1f} p50={q(.5)} p95={q(.95)} max={lengths[-1]}'
    if limit:
        msg += f' | >{limit} tokens (truncated) {over} ({over / len(lengths):.1%})'
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--min-neg', type=int, default=9, help='minimum negatives per row')
    ap.add_argument('--global-batch', type=int, default=0, help='queries per batch (0 = no check)')
    ap.add_argument('--tokenizer', help='tokenizer for token-length stats (HF name/path)')
    ap.add_argument('--query-prefix', default='', help='string actually prepended to queries')
    ap.add_argument('--query-max-len', type=int, default=0)
    ap.add_argument('--doc-max-len', type=int, default=0)
    args = ap.parse_args()

    errors, warnings = [], []
    rows = []
    for path in data_files(args.data):
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = f'{os.path.basename(path)}:{i}'
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f'{where} JSON parse error: {e}')
                    continue
                if not isinstance(r, dict):
                    errors.append(f'{where} not an object')
                    continue
                q, pos, neg = r.get('query'), r.get('pos'), r.get('neg')
                if not isinstance(q, str) or not q.strip():
                    errors.append(f'{where} query is not a non-empty string')
                if not isinstance(pos, list) or len(pos) != 1 or not all(isinstance(t, str) and t.strip() for t in pos):
                    errors.append(f'{where} pos must be a list of exactly one non-empty string')
                    continue
                if not isinstance(neg, list) or not all(isinstance(t, str) and t.strip() for t in neg):
                    errors.append(f'{where} neg must be a list of non-empty strings')
                    continue
                if len(neg) < args.min_neg:
                    errors.append(f'{where} {len(neg)} negatives < {args.min_neg}')
                if pos[0] in neg:
                    errors.append(f'{where} pos text appears among negatives')
                if len(set(neg)) != len(neg):
                    warnings.append(f'{where} duplicate negative texts')
                rows.append((where, q, pos[0], neg))

    neg_counts = collections.Counter(len(r[3]) for r in rows)
    if len(neg_counts) > 1:
        errors.append(f'negative counts differ across rows: {dict(neg_counts)}')
    if args.global_batch and len(rows) < args.global_batch:
        errors.append(f'{len(rows)} rows < batch {args.global_batch} (drop_last -> 0 training steps)')
    for name, idx in (('query', 1), ('pos', 2)):
        dup = [t for t, c in collections.Counter(r[idx] for r in rows).items() if c > 1]
        if dup:
            warnings.append(f'same {name} in several rows: {len(dup)}' +
                            (' (in-batch false negatives if they share a batch)' if name == 'pos' else ''))

    print(f'{len(rows)} rows, negatives per row {dict(neg_counts)}')
    if args.global_batch:
        print(f'batch {args.global_batch} -> {len(rows) // args.global_batch} batches per epoch ({len(rows) % args.global_batch} rows dropped per epoch)')
    if args.tokenizer and rows:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        n = lambda texts: [len(ids) for ids in tok(texts, add_special_tokens=True)['input_ids']]
        print(f'token lengths ({args.tokenizer}, incl. special tokens):')
        length_report('query', n([args.query_prefix + r[1] for r in rows]), args.query_max_len)
        length_report('doc', n([t for r in rows for t in [r[2]] + r[3]]), args.doc_max_len)

    for w in warnings[:20]:
        print('warning:', w)
    if len(warnings) > 20:
        print(f'{len(warnings) - 20} more warnings')
    for e in errors[:20]:
        print('error:', e)
    if errors:
        sys.exit(f'{len(errors)} errors - not training')
    print('check passed')


if __name__ == '__main__':
    main()
