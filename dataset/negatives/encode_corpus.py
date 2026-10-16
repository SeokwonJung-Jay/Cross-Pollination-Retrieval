#!/usr/bin/env python3
"""Encode the paper universe for candidate pooling (paper Section 3.3, Step 1) with E5, SPLADE-v3 or SciNCL.
Input: <corpus>/export/universe_*.parquet (build_universe.py; corpusid, title, abstract). Output, per shard and block of 20,000 rows:
<vectors>/<model>/emb_<shard>_<row offset, 7 digits>.pt = {'corpusid': int64 [n], 'dense': float16 [n, d]} or {'corpusid', 'sparse':
float16 sparse CSR [n, 30522]}; pool_universe.py aligns each file with the shard rows. Resumable per file; --part i/n encodes every
n-th file (one process per GPU).
Document encoders (model weights in float16, max 512 tokens, truncation):
  e5      intfloat/e5-large-v2, 'passage: ' + title + '. ' + abstract, mean pooling, L2-normalized (queries: 'query: ' prefix)
  scincl  malteos/scincl, title + [SEP] + abstract, [CLS] vector, not normalized
  splade  naver/splade-v3, title + '. ' + abstract, max over tokens of log(1 + relu(logits)) (masked)
The vectors behind the paper were made with these settings (re-encoding sample documents reproduces them to float16 precision; the
GPU kernels differ, so they are not bitwise identical).
  python -I encode_corpus.py --corpus $DATA/universe_2025 --vectors $DATA/vectors_2025 --model e5 [--device cuda:0] [--batch 128] [--part 0/1]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); from retrievers import NAME, REVISION
ap = argparse.ArgumentParser(); ap.add_argument('--corpus', required=True); ap.add_argument('--vectors', required=True); ap.add_argument('--model', required=True, choices=list(NAME))
ap.add_argument('--device', default='cuda:0'); ap.add_argument('--batch', type=int, default=128); ap.add_argument('--part', default='0/1'); ap.add_argument('--block', type=int, default=20000)
A = ap.parse_args()
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
M = A.model; name = NAME[M]; OUT = Path(A.vectors) / M; OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(name, revision=REVISION[M])
mdl = (AutoModelForMaskedLM if M == 'splade' else AutoModel).from_pretrained(name, revision=REVISION[M], dtype=torch.float16).to(A.device).eval()


def doc_text(t, a):
    return {'e5': f'passage: {t}. {a}', 'scincl': t + tok.sep_token + a, 'splade': f'{t}. {a}'}[M]


@torch.no_grad()
def enc(texts):
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(A.device); o = mdl(**b)
    if M == 'scincl': return o.last_hidden_state[:, 0].float()
    if M == 'e5':
        m = b['attention_mask'][..., None].float(); return torch.nn.functional.normalize((o.last_hidden_state.float() * m).sum(1) / m.sum(1), dim=-1)
    return (torch.log1p(torch.relu(o.logits.float())) * b['attention_mask'][..., None]).max(1).values


jobs = []
for f in sorted((Path(A.corpus) / 'export').glob('universe_*.parquet')):
    n = pq.ParquetFile(f).metadata.num_rows
    jobs += [(f, off) for off in range(0, n, A.block)]
i, k = map(int, A.part.split('/')); jobs = jobs[i::k]; t0 = time.time()
for j, (f, off) in enumerate(jobs):
    o = OUT / f'emb_{f.stem}_{off:07d}.pt'
    if o.exists(): continue
    t = pq.read_table(f, columns=['corpusid', 'title', 'abstract']).slice(off, A.block)
    texts = [doc_text(a or '', b or '') for a, b in zip(t.column('title').to_pylist(), t.column('abstract').to_pylist())]
    V = torch.cat([enc(texts[s:s + A.batch]).cpu() for s in range(0, len(texts), A.batch)]).half()
    E = {'corpusid': torch.from_numpy(t.column('corpusid').to_numpy().astype(np.int64))}
    if M == 'splade': E['sparse'] = V.to_sparse_csr()
    else: E['dense'] = V
    tmp = o.with_suffix('.tmp'); torch.save(E, tmp); tmp.replace(o)
    print(f'{M} {j + 1}/{len(jobs)} {o.name} {time.time() - t0:.0f}s', flush=True)
