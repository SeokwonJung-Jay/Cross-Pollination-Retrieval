#!/usr/bin/env python3
"""Encode the AnaloBench story pools (prepare.py). Each of the 340 stories is encoded as query and as document, at max length 512 and 1024
(paper ablation: 1024). Query = 'Query:' + story, document = story without prompt; sentence-transformers, bf16, left padding, L2-normalized,
batch 32 -> WORK/emb/<model>/m<max_len>/L<n>_q.npy, L<n>_d.npy (float32, story order of data/L<n>.json); tok_L<n>.json = token counts.
Models:
  qwen06b / qwen4b                 vanilla Qwen3-Embedding-0.6B / 4B (last-token pooling)
  nemotron8b_qp                    vanilla nvidia/llama-embed-nemotron-8b at NEMOTRON_REV (official remote code, flash_attention_2, mean pooling)
  abl2v2_<base>_<neg>_<data>       a dataset-ablation run's final checkpoint (--ckpt PATH); abl2v2_nemotron8b_* loaded like nemotron8b_qp
llama-embed-nemotron-8b needs the training/nemotron/setup_env_eval.sh environment (transformers 4.51.0, sentence-transformers 5.1.2);
Qwen the training/qwen environment.
  python -I encode.py MODEL [MODEL ...] --work WORK [--device cuda:0] [--batch N] [--ckpt PATH]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer
PROMPT, LENS, MAXLEN = 'Query:', (1, 10, 30), (512, 1024)
NEMOTRON_REV = 'aa3b43a495a9b280d1bdb716da37c54bb495d630'   # nvidia/llama-embed-nemotron-8b
HF = {'qwen06b': 'Qwen/Qwen3-Embedding-0.6B', 'qwen4b': 'Qwen/Qwen3-Embedding-4B', 'nemotron8b_qp': 'nvidia/llama-embed-nemotron-8b'}
ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--work', required=True); ap.add_argument('--device', default='cuda:0')
ap.add_argument('--batch', type=int, default=32)
# --ckpt PATH = weights of the (single) model given (abl2v2_*: the final checkpoint of the run).
ap.add_argument('--ckpt')
a = ap.parse_args(); W = Path(a.work)
def src(m):
    if a.ckpt: assert len(a.models) == 1; return a.ckpt
    assert m in HF, f'{m}: abl2v2_* models need --ckpt'; return HF[m]
for mname in a.models:
    O = W / 'emb' / mname
    if all((O / f'm{x}' / f'L{n}_d.npy').exists() for x in MAXLEN for n in LENS): print(mname, 'already done'); continue
    if mname == 'nemotron8b_qp' or mname.startswith('abl2v2_nemotron8b_'):   # official remote code (transformers 4.51.0 / sentence-transformers 5.1.2)
        r = NEMOTRON_REV if src(mname).startswith('nvidia/') else None
        m = SentenceTransformer(src(mname), device=a.device, trust_remote_code=True, revision=r, tokenizer_kwargs={'padding_side': 'left'},
                                model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'flash_attention_2'})
    else: m = SentenceTransformer(src(mname), device=a.device, model_kwargs={'torch_dtype': torch.bfloat16}, processor_kwargs={'padding_side': 'left'})
    P = PROMPT
    print(mname, src(mname), 'model query prompt:', repr(m.prompts.get('query')), '-> used:', repr(P), flush=True)
    for n in LENS:
        S = json.load(open(W / 'data' / f'L{n}.json'))['stories']
        tk = m.tokenizer; (O).mkdir(parents=True, exist_ok=True)
        json.dump({'q': [len(tk(P + s)['input_ids']) for s in S], 'd': [len(tk(s)['input_ids']) for s in S]}, open(O / f'tok_L{n}.json', 'w'))
        for x in MAXLEN:
            m.max_seq_length = x; (O / f'm{x}').mkdir(parents=True, exist_ok=True)
            enc = lambda **kw: m.encode(S, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, **kw).astype(np.float32)
            np.save(O / f'm{x}' / f'L{n}_q.npy', enc(prompt=P)); np.save(O / f'm{x}' / f'L{n}_d.npy', enc())
            print(f'  S{n} max {x} done', flush=True)
    del m; torch.cuda.empty_cache()
