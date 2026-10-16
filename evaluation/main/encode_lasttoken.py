#!/usr/bin/env python3
"""Encoder for models.json entries of kind 'hf_lasttoken' (inf-retriever-v1-pro): plain transformers as in the model card
(AutoTokenizer / AutoModel with trust_remote_code, truncation to max_len, last-token pooling, L2 normalize). Same eval data and output layout as
encode.py: WORK/_data/{texts,pools}.json (prepare.py) -> WORK/<model>/{embeddings/, results/scores.json, config.json};
scores.json {query_id: {cid: cosine}} over the query's 151-doc pool, scored by score.py.
Docs are sorted by token length (longest first), cut into fixed batches, and the batches are dealt round-robin to --devices (one process per GPU),
so the batches do not depend on the number of GPUs. Queries are encoded on the first device. Needs transformers 4.51.0 (env/infret.txt).
  python -I encode_lasttoken.py MODEL --work WORK --devices cuda:0,...,cuda:7 [--registry models.json]"""
import argparse, json, platform, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--work', required=True); ap.add_argument('--devices', default='cuda:0')
ap.add_argument('--registry', default=str(HERE / 'models.json')); ap.add_argument('--shard_of', help=argparse.SUPPRESS); ap.add_argument('--shard_out', help=argparse.SUPPRESS)
a = ap.parse_args(); cfg = json.load(open(a.registry))[a.model]; W = Path(a.work); assert cfg['kind'] == 'hf_lasttoken', cfg['kind']
T = json.load(open(W / '_data' / 'texts.json')); pools = json.load(open(W / '_data' / 'pools.json'))
POOL = {q: [x['pos_cid']] + [c for s in ('q_side', 'pos_side', 'other') for c in x['main'][s]] for q, x in pools.items()}
ids = sorted(T); qids = sorted(pools); docs = [cfg['doc_prompt'] + T[c] for c in ids]; queries = [cfg['query_prompt'] + pools[q]['query'] for q in qids]
devs = a.devices.split(','); t0 = time.time()
import torch, torch.nn.functional as F, transformers
from transformers import AutoModel, AutoTokenizer
tk = AutoTokenizer.from_pretrained(cfg['hf'], revision=cfg['revision'], trust_remote_code=cfg['trust_remote_code'])

def last_token_pool(h, mask):   # = model card
    if (mask[:, -1].sum() == mask.shape[0]): return h[:, -1]
    n = mask.sum(dim=1) - 1; return h[torch.arange(h.shape[0], device=h.device), n]
def load(dev):
    return AutoModel.from_pretrained(cfg['hf'], revision=cfg['revision'], trust_remote_code=cfg['trust_remote_code'], torch_dtype=getattr(torch, cfg['dtype'])).to(dev).eval()
def enc(m, dev, xs):
    bt = tk(xs, max_length=cfg['max_len'], padding=True, truncation=True, return_tensors='pt').to(dev)
    with torch.no_grad(): return F.normalize(last_token_pool(m(**bt).last_hidden_state, bt['attention_mask']).float(), p=2, dim=1).cpu().numpy()

L = [len(x) for x in tk(docs, max_length=cfg['max_len'], truncation=True)['input_ids']]
order = np.argsort([-l for l in L], kind='stable'); bat = [order[i:i + cfg['batch']] for i in range(0, len(order), cfg['batch'])]
if a.shard_of:   # child: batches i, i+n, ... on its device
    i, n = map(int, a.shard_of.split('/')); m = load(devs[0]); idx = bat[i::n]; V = []
    for k, b in enumerate(idx):
        V.append(enc(m, devs[0], [docs[j] for j in b]))
        if k % 100 == 0: print(f'shard {i}/{n} batch {k}/{len(idx)}', flush=True)
    np.savez(a.shard_out, idx=np.concatenate(idx), emb=np.vstack(V)); sys.exit()
out = W / a.model; e = out / 'embeddings'; e.mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
print(a.model, 'docs', len(ids), 'queries', len(qids), 'devices', devs, 'truncated docs', sum(l >= cfg['max_len'] for l in L), flush=True)
tmp = Path(tempfile.mkdtemp(dir=out))
P = [subprocess.Popen([sys.executable, '-I', __file__, a.model, '--work', a.work, '--registry', a.registry, '--devices', d, '--shard_of', f'{i}/{len(devs)}',
                       '--shard_out', str(tmp / f'{i}.npz')]) for i, d in enumerate(devs)]
assert all(p.wait() == 0 for p in P), 'a shard failed'
D = None
for i in range(len(devs)):
    z = np.load(tmp / f'{i}.npz'); D = np.zeros((len(ids), z['emb'].shape[1]), np.float32) if D is None else D; D[z['idx']] = z['emb']; (tmp / f'{i}.npz').unlink()
tmp.rmdir(); m = load(devs[0])
Q = np.vstack([enc(m, devs[0], queries[s:s + cfg['batch']]) for s in range(0, len(queries), cfg['batch'])])
np.save(e / 'doc_emb.npy', D); np.save(e / 'q_emb.npy', Q); json.dump(ids, open(e / 'doc_ids.json', 'w')); json.dump(qids, open(e / 'q_ids.json', 'w'))
row = {c: i for i, c in enumerate(ids)}
S = {q: {c: float(Q[qi] @ D[row[c]]) for c in POOL[q]} for qi, q in enumerate(qids)}
json.dump(S, open(out / 'results' / 'scores.json', 'w'))
json.dump({'model': a.model, 'registry_file': str(Path(a.registry).resolve()), 'registry': cfg, 'eval_set': json.load(open(W / '_data' / 'source.json')),
           'embeddings': ['doc_emb.npy', 'q_emb.npy', 'doc_ids.json', 'q_ids.json'], 'truncated_docs': int(sum(l >= cfg['max_len'] for l in L)),
           'query_tokens_max': int(max(len(x) for x in tk(queries)['input_ids'])),
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__}, 'python': platform.python_version(), 'devices': devs,
           'seconds': round(time.time() - t0), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, open(out / 'config.json', 'w'), indent=1)
print('saved', a.model, round(time.time() - t0), 's', flush=True)
