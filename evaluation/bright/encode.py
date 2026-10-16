#!/usr/bin/env python3
"""BRIGHT, official setting: every query of a split searches that split's whole corpus; the query's excluded_ids are removed from the ranking.
Models (dataset ablation), all with query prompt 'Query:' and no document prompt, sentence-transformers, bf16, left padding, L2-normalized, cosine:
  qwen06b_qp / qwen4b_qp           vanilla Qwen3-Embedding-0.6B / 4B
  nemotron8b_qp                    vanilla nvidia/llama-embed-nemotron-8b at NEMOTRON_REV (official remote code, flash_attention_2, mean pooling)
  abl2v2_<base>_<neg>_<data>       a dataset-ablation run's final checkpoint (--ckpt PATH). Qwen: the checkpoint's saved 'query' prompt
                                   ('Query:'); abl2v2_nemotron8b_*: loaded like nemotron8b_qp, literal 'Query:'
Max length QWEN_EVAL_MAXLEN (paper ablation: 1024, every model). llama-embed-nemotron-8b needs the training/nemotron/setup_env_eval.sh
environment (transformers 4.51.0, sentence-transformers 5.1.2); Qwen the training/qwen environment.
Doc vectors: OUT/<model>/<split>/chunk_*.npy (float16, 2,000 docs each in corpus order, resumable). Output OUT/<model>/<split>/run.json
{query id: [[doc id, score], ...top 1000]}; query vectors OUT/<model>/<split>/q_emb.npy (float32) in q_ids.json order.
--shard i/n encodes only doc chunks k with k % n == i (no queries / run.json); a later call without --shard does the queries and the ranking.
  python3 bright/encode.py MODEL --data DIR --out DIR --splits biology ... [--ckpt PATH] [--device cuda:0] [--batch N] [--shard i/n]"""
import json, argparse, time, os, math, sys
from pathlib import Path
import numpy as np, pandas as pd
NEMOTRON_REV = 'aa3b43a495a9b280d1bdb716da37c54bb495d630'   # nvidia/llama-embed-nemotron-8b
QWEN_FT = {'qwen06b_qp': ('Qwen/Qwen3-Embedding-0.6B', 'Query:'),   # vanilla with the trained checkpoints' prompt 'Query:'
           'qwen4b_qp': ('Qwen/Qwen3-Embedding-4B', 'Query:'),
           'nemotron8b_qp': ('nvidia/llama-embed-nemotron-8b', 'Query:')}
# abl2v2_<base>_<neg>_<data>: weights from --ckpt (the final checkpoint of the run)
if len(sys.argv) > 1 and sys.argv[1].startswith('abl2v2_nemotron8b_'): QWEN_FT[sys.argv[1]] = (None, 'Query:')
elif len(sys.argv) > 1 and sys.argv[1].startswith('abl2v2_'): QWEN_FT[sys.argv[1]] = (None, None)
ap = argparse.ArgumentParser(); ap.add_argument('model', choices=list(QWEN_FT))
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--splits', nargs='+', required=True)
ap.add_argument('--ckpt', help='checkpoint path or HF id (required for abl2v2_* models)')
# --shard i/n: encode only doc chunks k with k % n == i and stop; a later call without --shard does the queries and the ranking (same vectors).
ap.add_argument('--shard'); ap.add_argument('--queries_only', action='store_true'); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--batch', type=int, default=32); a = ap.parse_args(); CH, TOPK = 2000, 1000
D_ = Path(a.data)
NEMO = a.model == 'nemotron8b_qp' or a.model.startswith('abl2v2_nemotron8b_')
if a.ckpt: QWEN_FT[a.model] = (a.ckpt, QWEN_FT[a.model][1] if NEMO else None)
assert QWEN_FT[a.model][0] is not None, 'abl2v2_* models need --ckpt'
def top_excl(scores, dids, excl):   # scores: 1-D numpy over the corpus; excluded ids never ranked
    s = scores.astype(np.float64).copy()
    for j in excl: s[j] = -np.inf
    idx = np.argpartition(-s, TOPK)[:TOPK]; idx = idx[np.argsort(-s[idx])]
    return [[dids[j], float(s[j])] for j in idx if np.isfinite(s[j])]
for sp in a.splits:
    OUT = Path(a.out) / a.model / sp; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    QF = OUT / 'q_emb.npy'
    if a.queries_only and (QF.exists() or not (OUT / 'run.json').exists()): print(sp, 'skip (queries_only)'); continue
    if (OUT / 'run.json').exists() and not a.queries_only: print(sp, 'already done'); continue
    docs = pd.read_parquet(D_ / f'documents_{sp}.parquet'); dids = [str(x) for x in docs['id']]; T = dict(zip(dids, docs['content'].tolist()))
    ex = pd.read_parquet(D_ / f'examples_{sp}.parquet'); qids = [str(x) for x in ex['id']]; queries = ex['query'].tolist(); pos = {d: j for j, d in enumerate(dids)}
    excl = [[pos[str(x)] for x in e if str(x) in pos] for e in ex['excluded_ids']]
    print(a.model, sp, 'docs', len(dids), 'queries', len(qids), flush=True)
    run = {}
    import torch
    from sentence_transformers import SentenceTransformer
    name, instr = QWEN_FT[a.model]
    # Qwen: processor_kwargs (sentence-transformers >= 5.2); llama-embed-nemotron-8b: official remote code with transformers 4.51.0 /
    # sentence-transformers 5.1.2 (tokenizer_kwargs), flash_attention_2
    kw = {'processor_kwargs': {'padding_side': 'left'}}
    if NEMO: kw = {'trust_remote_code': True, 'tokenizer_kwargs': {'padding_side': 'left'}, 'revision': NEMOTRON_REV if name.startswith('nvidia/') else None}
    # QWEN_EVAL_MAXLEN = max length (paper ablation: 1024). Above 512 the texts go in token-budget batches
    # (longest first, <= 16384 padded tokens, at most --batch) so a long doc does not share a batch of 32.
    ML = int(os.environ.get('QWEN_EVAL_MAXLEN', 1024))
    m = SentenceTransformer(name, device=a.device, model_kwargs={'torch_dtype': torch.bfloat16, **({'attn_implementation': 'flash_attention_2'} if NEMO else {})}, **kw); m.max_seq_length = ML
    def enc_st(xs, **k2):
        if ML <= 512: return m.encode(xs, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, **k2).astype(np.float32)
        n = [min(len(t), ML) for t in m.tokenizer(list(xs), add_special_tokens=True)['input_ids']]; order = sorted(range(len(xs)), key=lambda j: -n[j])
        out = np.zeros((len(xs), m.get_sentence_embedding_dimension()), np.float32); i = 0
        while i < len(order):
            bs = max(1, min(a.batch, 16384 // max(n[order[i]] + 16, 1))); b = order[i:i + bs]; i += bs
            out[b] = m.encode([xs[j] for j in b], batch_size=len(b), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, **k2)
        return out
    enc_d = lambda xs: enc_st(xs)
    enc_q = (lambda xs: enc_st(xs, prompt=instr)) if instr else (lambda xs: enc_st(xs, prompt_name='query'))
    nck = math.ceil(len(dids) / CH)
    for k in range(0 if a.queries_only else nck):   # chunk k = docs [k*CH, (k+1)*CH) in corpus order; resumable
        f = OUT / f'chunk_{k:04d}.npy'
        if f.exists() or (a.shard and k % int(a.shard.split('/')[1]) != int(a.shard.split('/')[0])): continue
        E = enc_d([T[d] for d in dids[k * CH:(k + 1) * CH]])
        np.save(f, E.astype(np.float16))
        print(f'  chunk {k + 1}/{nck} {time.time() - t0:.0f}s', flush=True)
    if a.shard: print(sp, 'shard', a.shard, 'done', f'{time.time() - t0:.0f}s', flush=True); continue
    Qe = enc_q(queries); json.dump(qids, open(OUT / 'q_ids.json', 'w'))
    np.save(QF, np.asarray(Qe, np.float32))
    if a.queries_only: print(sp, 'query vectors saved', flush=True); continue
    S = np.zeros((len(qids), len(dids)), np.float32)
    Qt = torch.as_tensor(Qe, device=a.device, dtype=torch.float16)
    for k in range(nck):
        E = np.load(OUT / f'chunk_{k:04d}.npy'); lo = k * CH
        with torch.no_grad(): S[:, lo:lo + len(E)] = (Qt @ torch.as_tensor(E, device=a.device).T).float().cpu().numpy()
    for qi in range(len(qids)): run[qids[qi]] = top_excl(S[qi], dids, excl[qi])
    json.dump(run, open(OUT / 'run.json', 'w')); print(sp, 'run saved', f'{time.time() - t0:.0f}s', flush=True)
