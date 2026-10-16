"""Convert our training data to the ms-swift embedding (InfoNCE) format.

    python prepare_data.py TRAIN_DATA OUT.jsonl --instruction '...'
    python prepare_data.py TRAIN_DATA OUT.jsonl --query-prompt 'Query:'

Input: one JSONL file of {"query", "pos": [one], "neg": [...]} rows, or a directory of such files (concatenated in name order).
Output row (ms-swift docs/source_en/BestPractices/Embedding.md, "Format for InfoNCE"):
    {"messages": [{"role": "user", "content": "<prompt>{query}"}],
     "positive_messages": [[{"role": "user", "content": pos}]],
     "negative_messages": [[{"role": "user", "content": neg_1}], ...]}
The query prompt goes directly into the content (ms-swift's system field would join with a space and differ from inference).
Documents get no prompt. The trailing <|endoftext|> is added by the ms-swift qwen3_emb template.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--instruction')
    g.add_argument('--query-prompt', help="string prepended to each query as is (no instruction format), e.g. 'Query:'; '' = raw query")
    args = ap.parse_args()

    if os.path.isdir(args.data):
        files = sorted(os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(('.jsonl', '.json')))
    else:
        files = [args.data]
    user = lambda text: [{'role': 'user', 'content': text}]
    prompt = args.query_prompt if args.query_prompt is not None else f'Instruct: {args.instruction}\nQuery:'
    n = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for path in files:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    row = {'messages': user(prompt + r['query']),
                           'positive_messages': [user(r['pos'][0])],
                           'negative_messages': [user(t) for t in r['neg']]}
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    n += 1
    print(f'{args.out}: {n} rows ({len(files)} files)')


if __name__ == '__main__':
    main()
