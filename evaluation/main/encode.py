#!/usr/bin/env python3
"""Main experiment encoder: one models.json entry per call. Encodes every doc of the eval set (WORK/_data/texts.json, all pools)
and all queries, saves the embeddings, then scores each query against its own pool (positive + 150 negatives).
Output WORK/<model>/:
  embeddings/  doc_ids.json + q_ids.json and  dense: doc_emb.npy / q_emb.npy (float32)
                                              multi-vector: doc_emb_<k>.pt (list of fp16 [tokens, dim] tensors, 2,000 docs per file) / q_emb.pt
                                              sparse: doc_sparse.jsonl / q_sparse.jsonl ({token_id: weight} per line, id order)
               (bm25: none)
  results/scores.json  {query_id: {cid: score}} over the query's 151-doc pool (bm25: {variant: {...}}, idf depends on the pool)
  config.json  registry entry, eval-set source, library versions, device, wall time
_shared bge_m3 encodes once and writes bge_m3_dense / bge_m3_sparse / bge_m3_multivec.
  python -I encode.py MODEL --work WORK [--device cuda:0] [--devices cuda:0,cuda:1,...] [--registry models.json]
(inf-retriever-v1-pro, kind hf_lasttoken, is encoded by encode_lasttoken.py.)"""
import argparse, json, platform, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--work', required=True); ap.add_argument('--device', default='cuda:0')
ap.add_argument('--registry', default=str(HERE / 'models.json'), help='model registry (other experiments pass their own)')
# --devices cuda:0,cuda:1,... encodes the docs of an 'st' model over these GPUs (one process per GPU); the batches are exactly those of a
# single-GPU encode, so the vectors are identical (queries stay on --device)
ap.add_argument('--devices'); ap.add_argument('--shard_of', help=argparse.SUPPRESS); ap.add_argument('--shard_out', help=argparse.SUPPRESS); a = ap.parse_args()
REG = json.load(open(a.registry))
cfg = REG['_shared'][a.model] if a.model in REG['_shared'] else REG[a.model]; W = Path(a.work)
T = json.load(open(W / '_data' / 'texts.json')); pools = json.load(open(W / '_data' / 'pools.json'))
POOL = {q: [x['pos_cid']] + [c for s in ('q_side', 'pos_side', 'other') for c in x['main'][s]] for q, x in pools.items()}
VAR = {'main': ('q_side', 'pos_side', 'other'), 'cross_domain': ('pos_side', 'other')}
ids = sorted(T); qids = sorted(pools); docs = [T[c] for c in ids]; queries = [pools[q]['query'] for q in qids]; t0 = time.time()
print(a.model, cfg['kind'], 'docs', len(ids), 'queries', len(qids), flush=True)

def save(name, scores, emb_files, extra=None):
    out = W / name; (out / 'results').mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(out / 'results' / 'scores.json', 'w'))
    vers = {}
    for m in ('torch', 'transformers', 'sentence_transformers', 'pylate', 'FlagEmbedding', 'adapters', 'google.genai', 'bm25s'):
        try: vers[m] = __import__(m, fromlist=['_']).__version__
        except Exception: pass
    json.dump({'model': name, 'registry_file': str(Path(a.registry).resolve()), 'registry': REG.get(name, cfg), 'shared_encode': cfg if a.model in REG['_shared'] else None,
               'eval_set': json.load(open(W / '_data' / 'source.json')), 'embeddings': emb_files, 'versions': vers, 'python': platform.python_version(),
               'device': a.device, 'seconds': round(time.time() - t0), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **(extra or {})},
              open(out / 'config.json', 'w'), indent=1)
    print('saved', name, f'{time.time() - t0:.0f}s', flush=True)

def edir(name):
    e = W / name / 'embeddings'; e.mkdir(parents=True, exist_ok=True)
    json.dump(ids, open(e / 'doc_ids.json', 'w')); json.dump(qids, open(e / 'q_ids.json', 'w')); return e

def dense_scores(D, Q):
    row = {c: i for i, c in enumerate(ids)}
    return {q: {c: float(D[row[c]] @ Q[qi]) for c in POOL[q]} for qi, q in enumerate(qids)}

def save_dense(name, D, Q, extra=None):
    e = edir(name); np.save(e / 'doc_emb.npy', D.astype(np.float32)); np.save(e / 'q_emb.npy', Q.astype(np.float32))
    save(name, dense_scores(D.astype(np.float32), Q.astype(np.float32)), ['doc_emb.npy', 'q_emb.npy'], extra)

k = cfg['kind']
if k == 'bm25':   # bm25s, Lucene idf log(1 + (N - df + 0.5) / (df + 0.5)); bm25s leaves out the (k1 + 1) factor, multiplied back here
    import bm25s
    tok = lambda s: re.findall(r'[a-z0-9]+', s.lower()); k1, b = cfg['k1'], cfg['b']; TK = {}
    def bm25(pool, query):
        m = bm25s.BM25(k1=k1, b=b, method='lucene', dtype='float64'); m.index([TK.setdefault(i, tok(T[i])) for i in pool], show_progress=False)
        return dict(zip(pool, (m.get_scores(tok(query)) * (k1 + 1)).tolist()))
    S = {v: {q: bm25([x['pos_cid']] + [c for s in st for c in x['main'][s]], x['query']) for q, x in pools.items()} for v, st in VAR.items()}
    save(a.model, S, [])
    sys.exit()

import torch
if k == 'st':
    from sentence_transformers import SentenceTransformer
    # "remote_code": true (llama-embed-nemotron-8b, dataset ablation): the model's official remote code under transformers 4.51.0 /
    # sentence-transformers 5.1.2 (tokenizer_kwargs instead of processor_kwargs), flash_attention_2
    RC = cfg.get('remote_code', False)
    def load(dev):
        if RC: m = SentenceTransformer(cfg['hf'], revision=cfg['revision'], device=dev, trust_remote_code=True, tokenizer_kwargs={'padding_side': cfg['padding_side']},
                                       model_kwargs={'torch_dtype': getattr(torch, cfg['dtype']), 'attn_implementation': 'flash_attention_2'})
        else: m = SentenceTransformer(cfg['hf'], revision=cfg['revision'], device=dev, model_kwargs={'torch_dtype': getattr(torch, cfg['dtype'])},
                                      processor_kwargs={'padding_side': cfg['padding_side']})
        m.max_seq_length = cfg['max_len']; return m
    m = load(a.device)
    # vanilla Qwen: registry "query_prompt" = 'Query:' (same as our trained checkpoints, no instruction);
    # trained checkpoints: "query_prompt_name": "query" (their saved prompt, also 'Query:')
    prompt = cfg['query_prompt'] if 'query_prompt' in cfg else m.prompts.get(cfg['query_prompt_name'])
    enc = lambda xs, **kw: m.encode(xs, batch_size=cfg['batch'], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, **kw)
    # the exact single-GPU batches (sentence-transformers' own length sort, batch size) dealt out whole to the GPUs -> same vectors (checked bitwise);
    # one process per GPU
    if RC: order = np.argsort([-m._text_length(x) for x in docs])   # the encode order of sentence-transformers 5.1.2
    else: order = np.argsort([-m._input_length(x) for x in docs]); order = m._interleave_sorted_indices(order) if m._can_flatten_inputs() else order
    bat = [order[i:i + cfg['batch']] for i in range(0, len(order), cfg['batch'])]
    if a.shard_of:   # child: encode batches i, i+n, ... and exit
        i, n = map(int, a.shard_of.split('/')); idx = [b for b in bat[i::n]]; V = [enc([docs[j] for j in b]) for b in idx]
        np.savez(a.shard_out, idx=np.concatenate(idx), emb=np.vstack(V).astype(np.float32)); sys.exit()
    if a.devices:
        import subprocess, tempfile
        devs = a.devices.split(','); (W / a.model).mkdir(parents=True, exist_ok=True); tmp = Path(tempfile.mkdtemp(dir=W / a.model))
        P = [subprocess.Popen([sys.executable, '-I', __file__, a.model, '--work', a.work, '--registry', a.registry, '--device', d, '--shard_of', f'{i}/{len(devs)}',
                               '--shard_out', str(tmp / f'{i}.npz')]) for i, d in enumerate(devs)]
        assert all(p.wait() == 0 for p in P), 'a shard process failed'
        Z = [np.load(tmp / f'{i}.npz') for i in range(len(devs))]; D = np.zeros((len(docs), Z[0]['emb'].shape[1]), np.float32)
        for z in Z: D[z['idx']] = z['emb']
        import shutil; shutil.rmtree(tmp)
    else: D = enc(docs)
    Q = enc(queries, prompt=prompt)
    save_dense(a.model, D, Q, {'query_prompt': prompt})
elif k == 'contriever':
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(cfg['hf'], revision=cfg['revision']); mm = AutoModel.from_pretrained(cfg['hf'], revision=cfg['revision']).to(a.device).eval()
    def enc(xs):
        out = []
        for i in range(0, len(xs), cfg['batch']):
            bt = tk(xs[i:i + cfg['batch']], padding=True, truncation=True, max_length=cfg['max_len'], return_tensors='pt').to(a.device)
            with torch.no_grad(): h = mm(**bt).last_hidden_state
            msk = bt['attention_mask'].unsqueeze(-1).to(h.dtype); out.append(((h * msk).sum(1) / msk.sum(1)).float().cpu().numpy())
        return np.vstack(out)
    save_dense(a.model, enc(docs), enc(queries))
elif k == 'specter2':
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    tk = AutoTokenizer.from_pretrained(cfg['hf'], revision=cfg['revision']); mm = AutoAdapterModel.from_pretrained(cfg['hf'], revision=cfg['revision'])
    mm.load_adapter(cfg['doc_adapter'][0], source='hf', load_as='proximity', revision=cfg['doc_adapter'][1])
    mm.load_adapter(cfg['query_adapter'][0], source='hf', load_as='adhoc_query', revision=cfg['query_adapter'][1]); mm = mm.to(a.device).eval()
    def enc(xs, ad):
        mm.set_active_adapters(ad); out = []
        for i in range(0, len(xs), cfg['batch']):
            bt = tk(xs[i:i + cfg['batch']], padding=True, truncation=True, max_length=cfg['max_len'], return_tensors='pt', return_token_type_ids=False).to(a.device)
            with torch.no_grad(): out.append(mm(**bt).last_hidden_state[:, 0].float().cpu().numpy())
        return np.vstack(out)
    save_dense(a.model, enc([x.replace('. ', tk.sep_token, 1) for x in docs], 'proximity'), enc(queries, 'adhoc_query'))
elif k == 'pylate':
    from pylate import models
    m = models.ColBERT(model_name_or_path=cfg['hf'], revision=cfg['revision'], query_length=cfg['query_len'], document_length=cfg['doc_len'], device=a.device)
    e = edir(a.model); files = []; row = {}
    for j, s in enumerate(range(0, len(ids), 2000)):
        E = [torch.as_tensor(np.asarray(v)).half() for v in m.encode(docs[s:s + 2000], is_query=False, batch_size=cfg['batch'], show_progress_bar=False)]
        torch.save(E, e / f'doc_emb_{j:03d}.pt'); files.append(f'doc_emb_{j:03d}.pt'); row.update(zip(ids[s:s + 2000], E)); print('chunk', j, flush=True)
    Q = [torch.as_tensor(np.asarray(v)).half() for v in m.encode(queries, is_query=True, batch_size=cfg['batch'], show_progress_bar=False)]
    torch.save(Q, e / 'q_emb.pt')
    S = {q: {c: float((Q[qi].float().to(a.device) @ row[c].float().to(a.device).T).max(1).values.sum()) for c in POOL[q]} for qi, q in enumerate(qids)}
    save(a.model, S, files + ['q_emb.pt'])
elif k == 'bgem3':
    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download
    m = BGEM3FlagModel(snapshot_download(cfg['hf'], revision=cfg['revision']), use_fp16=cfg['fp16'], devices=[a.device])
    kw = dict(batch_size=cfg['batch'], max_length=cfg['max_len'], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    Qo = m.encode(queries, **kw); D, SP, CB = [], [], []
    for s in range(0, len(ids), 2000):
        o = m.encode(docs[s:s + 2000], **kw); D.append(o['dense_vecs']); SP += o['lexical_weights']; CB += [torch.as_tensor(v).half() for v in o['colbert_vecs']]; print('chunk', s // 2000, flush=True)
    D = np.vstack(D).astype(np.float32); row = {c: i for i, c in enumerate(ids)}
    save_dense('bge_m3_dense', D, np.asarray(Qo['dense_vecs'], np.float32), {'output': 'dense'})
    e = edir('bge_m3_sparse')
    with open(e / 'doc_sparse.jsonl', 'w') as f: [f.write(json.dumps({t: float(w) for t, w in x.items()}) + '\n') for x in SP]
    with open(e / 'q_sparse.jsonl', 'w') as f: [f.write(json.dumps({t: float(w) for t, w in x.items()}) + '\n') for x in Qo['lexical_weights']]
    save('bge_m3_sparse', {q: {c: float(m.compute_lexical_matching_score(Qo['lexical_weights'][qi], SP[row[c]])) for c in POOL[q]} for qi, q in enumerate(qids)},
         ['doc_sparse.jsonl', 'q_sparse.jsonl'], {'output': 'sparse'})
    e = edir('bge_m3_multivec'); files = []
    for j, s in enumerate(range(0, len(ids), 2000)): torch.save(CB[s:s + 2000], e / f'doc_emb_{j:03d}.pt'); files.append(f'doc_emb_{j:03d}.pt')
    QC = [torch.as_tensor(v).half() for v in Qo['colbert_vecs']]; torch.save(QC, e / 'q_emb.pt')
    save('bge_m3_multivec', {q: {c: float(m.colbert_score(QC[qi].float().numpy(), CB[row[c]].float().numpy())) for c in POOL[q]} for qi, q in enumerate(qids)},
         files + ['q_emb.pt'], {'output': 'colbert'})
elif k == 'gemini':   # gemini-embedding-001, 768 d, task types RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT
    from google import genai; from google.genai import types
    import os   # GEMINI_API_KEY from the environment, else from <repo>/.env
    key = os.environ.get('GEMINI_API_KEY', ''); env = HERE.parents[1] / '.env'
    if not key and env.exists(): key = next((l.split('=', 1)[1].strip().strip('"\'') for l in open(env) if l.startswith('GEMINI_API_KEY=')), '')
    assert key, 'set GEMINI_API_KEY (environment or <repo>/.env)'; gc = genai.Client(api_key=key)
    CF = W / a.model / 'embeddings' / 'api_cache.jsonl'; CF.parent.mkdir(parents=True, exist_ok=True); C = {}; chars = [0]
    if CF.exists():
        for l in open(CF):
            try: kk, v = json.loads(l); C[kk] = v
            except ValueError: pass   # a line cut by a kill
    def emb(xs, task):
        miss = list(dict.fromkeys(x for x in xs if f'{task}|{x}' not in C))
        for i in range(0, len(miss), cfg['batch']):
            sub = miss[i:i + cfg['batch']]
            for tr in range(6):
                try: r = gc.models.embed_content(model=cfg['api_model'], contents=sub, config=types.EmbedContentConfig(task_type=task, output_dimensionality=cfg['dim'])); break
                except Exception as ex:
                    if tr < 5 and any(s in str(ex) for s in ('429', 'RESOURCE_EXHAUSTED', '500', '503', 'UNAVAILABLE', 'INTERNAL')): time.sleep(30 * (tr + 1)); continue
                    raise
            with open(CF, 'a') as f:
                for x, ev in zip(sub, r.embeddings): C[f'{task}|{x}'] = ev.values; f.write(json.dumps([f'{task}|{x}', ev.values]) + '\n')
            chars[0] += sum(map(len, sub))
            if (i // cfg['batch']) % 20 == 0: print(f'{task} {i + len(sub)}/{len(miss)} {time.time() - t0:.0f}s', flush=True)
        V = np.asarray([C[f'{task}|{x}'] for x in xs], np.float32); return V / np.linalg.norm(V, axis=1, keepdims=True)
    save_dense(a.model, emb(docs, cfg['doc_task']), emb(queries, cfg['query_task']), {'api_chars_sent_this_run': chars[0], 'note': 'vectors L2-normalized (768-d output is not unit length)'})
else: raise SystemExit(f'unknown kind {k}')
