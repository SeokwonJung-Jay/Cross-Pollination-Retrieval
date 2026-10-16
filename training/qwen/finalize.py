"""Finalize a trained checkpoint for sentence-transformers and check it encodes exactly as in training.

    python finalize.py OUTPUT_DIR/checkpoint-N [--instruction-file OUTPUT_DIR/query_instruction.txt]

1. Set the query prompt in config_sentence_transformers.json to the training prompt
   (ms-swift copies the base model's file with its original "Given a web search query, ..." instruction).
2. Checks (exit 1 on mismatch):
   - tokens: input_ids from the ms-swift qwen3_emb training template == sentence-transformers inference input_ids
     (query: `<prompt>{q}<|endoftext|>`, document: `{doc}<|endoftext|>`);
   - embeddings: sentence-transformers encode == direct computation (hidden state at the final <|endoftext|>, L2-normalized).
Training-resume files (optimizer.pt, ...) are kept.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--instruction-file', help='default: query_instruction.txt next to the checkpoint')
    args = ap.parse_args()
    ckpt = os.path.abspath(args.ckpt)
    # query_prompt.txt written by train.sh = the exact string prepended to queries
    prompt_file = os.path.join(os.path.dirname(ckpt), 'query_prompt.txt')
    if args.instruction_file is None and os.path.exists(prompt_file):
        query_prompt = open(prompt_file, encoding='utf-8').read()
    else:
        instr_file = args.instruction_file or os.path.join(os.path.dirname(ckpt), 'query_instruction.txt')
        instruction = open(instr_file, encoding='utf-8').read()
        query_prompt = f'Instruct: {instruction}\nQuery:'

    cfg_path = os.path.join(ckpt, 'config_sentence_transformers.json')
    cfg = json.load(open(cfg_path))
    cfg['prompts'] = {'query': query_prompt, 'document': ''}
    cfg['default_prompt_name'] = None
    json.dump(cfg, open(cfg_path, 'w'), indent=2, ensure_ascii=False)
    print(f'query prompt → {query_prompt!r}')

    from sentence_transformers import SentenceTransformer
    from swift.model import get_model_processor
    from swift.template import get_template
    from transformers import AutoModel, AutoTokenizer

    # check a few rows actually used in training (prepare_data.py output)
    rows = []
    with open(os.path.join(os.path.dirname(ckpt), 'swift_train.jsonl'), encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == 3:
                break

    # 1) tokens: ms-swift training encode (anchor/positive/negative) vs sentence-transformers tokenizer
    _, processor = get_model_processor(ckpt, model_type='qwen3_emb', task_type='embedding', load_model=False, use_hf=True)
    template = get_template(processor, template_type='qwen3_emb')
    template.set_mode('train')
    # fp32 like the direct computation below (a bf16 load differs by rounding only and would fail the 1e-4 check)
    st = SentenceTransformer(ckpt, device='cpu', processor_kwargs={'padding_side': 'left'}, model_kwargs={'dtype': torch.float32})
    ok = True
    queries, docs = [], []
    for row in rows:
        enc = template.encode(row)
        q = row['messages'][0]['content']
        assert q.startswith(query_prompt), 'query prompt in the training data differs from query_prompt.txt / query_instruction.txt'
        queries.append(q[len(query_prompt):])
        texts = [m[0]['content'] for m in row['positive_messages'] + row['negative_messages']]
        docs += texts[:2]
        pairs = [(enc['anchor_input_ids'], q)] + [(enc['positive_input_ids'], texts[0])] + \
                list(zip(enc.get('negative_input_ids', []), texts[1:]))   # key absent when a row has no hard negatives
        for swift_ids, text in pairs:
            if list(swift_ids) != st.tokenizer(text)['input_ids']:
                ok = False
                print('token mismatch:', repr(text[:60]))
    eos = st.tokenizer.convert_tokens_to_ids('<|endoftext|>')
    print(f'tokens match (ms-swift training == sentence-transformers, final token <|endoftext|>={eos}):', ok,
          f'| {len(queries)} queries, {sum(1 + len(r['negative_messages']) for r in rows)} documents compared')

    # 2) embeddings: sentence-transformers vs direct computation
    tok = AutoTokenizer.from_pretrained(ckpt, padding_side='left')
    model = AutoModel.from_pretrained(ckpt, dtype=torch.float32).eval()
    with torch.no_grad():
        texts = [query_prompt + q for q in queries] + docs
        batch = tok(texts, padding=True, return_tensors='pt')
        ref = F.normalize(model(**batch).last_hidden_state[:, -1], p=2, dim=1)
    st_emb = torch.cat([st.encode(queries, prompt_name='query', convert_to_tensor=True),
                        st.encode(docs, convert_to_tensor=True)]).float()
    diff = (st_emb - ref).abs().max().item()
    print(f'max embedding difference (sentence-transformers vs direct): {diff:.2e}')
    if not ok or diff > 1e-4:
        sys.exit('check failed')
    print(f'done: use SentenceTransformer({ckpt!r})')


if __name__ == '__main__':
    main()
