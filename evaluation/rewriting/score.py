#!/usr/bin/env python3
"""Retrieval with the rewritten queries (generate.py) over each query's 151-document pool, BM25 and Gemini Embedding, and metrics.
  BM25     evaluation/main settings: bm25s (Lucene variant), tokens [a-z0-9]+ (lowercased), k1 1.5, b 0.75, idf within the query's pool
  Gemini   document and original-query vectors of evaluation/main (WORK/gemini_emb_001/embeddings, gemini-embedding-001, 768-d,
           L2-normalized); generated texts embedded here with the same model (RETRIEVAL_DOCUMENT for HyDE passages, RETRIEVAL_QUERY
           otherwise), L2-normalized, cached in DIR/gemini_cache.jsonl; score = cosine
  hyde       BM25: query + ' ' + the 8 passages joined by ' '; Gemini: mean of the query vector and the 8 passage vectors
  query2doc  BM25: the query 5 times + ' ' + passage (joined by ' '); Gemini: one RETRIEVAL_QUERY vector of 'query [SEP] passage'
  ragfusion  rankings of the original query and the 4 generated queries fused by reciprocal rank fusion, sum of 1 / (60 + rank)
  --trunc    every generated passage (HyDE, Query2doc) is cut to its first 384 whitespace words before use (about 512 tokens), as in
             the paper run; RAG-Fusion queries are not cut
Rank of the positive = position in the pool sorted by descending score (stable sort over the pool order positive, q_side, pos_side,
other; Gemini: numpy argsort), the procedure behind the paper's rewriting numbers. A query without an output file counts as a miss
(rank 152). The 'original-*' rows are evaluation/main's per-query ranks (WORK/{bm25,gemini_emb_001}/results/per_query.json, main pool),
the paper's Original values.
Metrics as evaluation/main/score.py: R@1/5/10, MRR@10, nDCG@10, mean over all queries, 95% CI = bootstrap over queries (2,000 resamples,
random.Random(0), sorted index 50 / 1949). GEMINI_API_KEY from the environment or a .env file (--env-file, default <repo>/.env).
Writes DIR/results/{ranks.json, metrics.json} and prints the table.
  python -I score.py --work WORK [--dir WORK/rewrite_glm53] [--retrievers bm25,gemini] [--trunc 384]"""
import argparse, json, math, os, random, re, statistics, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True, help='evaluation/main WORK (prepare.py + gemini_emb_001 [+ bm25] done)')
ap.add_argument('--dir', help='generate.py output (default WORK/rewrite_glm53)'); ap.add_argument('--retrievers', default='bm25,gemini')
ap.add_argument('--trunc', type=int, default=384, help='keep the first N whitespace words of every generated passage, 0 = no cut')
ap.add_argument('--env-file', default=str(HERE.parents[1] / '.env')); a = ap.parse_args()
W = Path(a.work); RW = Path(a.dir) if a.dir else W / 'rewrite_glm53'; RET = a.retrievers.split(',')
pools = json.load(open(W / '_data/pools.json')); T = json.load(open(W / '_data/texts.json')); qids = sorted(pools)
POOL = {q: [x['pos_cid']] + [c for s in ('q_side', 'pos_side', 'other') for c in x['main'][s]] for q, x in pools.items()}; N = len(POOL[qids[0]])

tok = lambda s: re.findall(r'[a-z0-9]+', s.lower()); IX = {}
def bm25(pool, query, k1=1.5, b=0.75):   # = evaluation/main/encode.py kind 'bm25' (bm25s, Lucene idf, x (k1 + 1)); one index per pool
    key = tuple(pool)
    if key not in IX:
        import bm25s
        IX.clear(); IX[key] = bm25s.BM25(k1=k1, b=b, method='lucene', dtype='float64'); IX[key].index([tok(T[i]) for i in pool], show_progress=False)
    return dict(zip(pool, (IX[key].get_scores(tok(query)) * (k1 + 1)).tolist()))
def bm(pool, query): sc = bm25(pool, query); return sorted(pool, key=lambda d: -sc[d])
def rrf(orders, k=60):
    s = {}
    for o in orders:
        for i, d in enumerate(o): s[d] = s.get(d, 0) + 1 / (k + i + 1)
    return sorted(s, key=lambda d: -s[d])

if 'gemini' in RET:
    from google import genai; from google.genai import types
    key = os.environ.get('GEMINI_API_KEY', '')
    if not key and Path(a.env_file).exists(): key = next((l.split('=', 1)[1].strip().strip('"\'') for l in open(a.env_file) if l.startswith('GEMINI_API_KEY=')), '')
    assert key, 'set GEMINI_API_KEY (environment or .env), or --retrievers bm25'; gc = genai.Client(api_key=key)
    G = W / 'gemini_emb_001/embeddings'; D = np.load(G / 'doc_emb.npy'); DI = {c: i for i, c in enumerate(json.load(open(G / 'doc_ids.json')))}
    QE = np.load(G / 'q_emb.npy'); QI = {q: i for i, q in enumerate(json.load(open(G / 'q_ids.json')))}
    CF = RW / 'gemini_cache.jsonl'; E = {}
    if CF.exists():
        for l in open(CF):
            try: k, v = json.loads(l); E[k] = v
            except ValueError: pass   # a line cut by a kill
    def emb(texts, task):
        need = [t for t in dict.fromkeys(texts) if f'{task}|{t}' not in E]
        for i in range(0, len(need), 100):
            sub = need[i:i + 100]
            for k in range(5):
                try: r = gc.models.embed_content(model='gemini-embedding-001', contents=sub, config=types.EmbedContentConfig(task_type=task, output_dimensionality=768)); break
                except Exception:
                    if k == 4: raise
                    time.sleep(10 * (k + 1))
            with open(CF, 'a') as fh:
                for t, e in zip(sub, r.embeddings):
                    v = np.array(e.values, np.float32); v /= np.linalg.norm(v) + 1e-9; E[f'{task}|{t}'] = v.tolist(); fh.write(json.dumps([f'{task}|{t}', E[f'{task}|{t}']]) + '\n')
        return np.array([E[f'{task}|{t}'] for t in texts], np.float32)

cut = (lambda s: ' '.join(s.split()[:a.trunc])) if a.trunc else (lambda s: s)
def ld(m, q): f = RW / m / f'{q}.json'; return json.load(open(f)) if f.exists() else None
R = {f'{m}-{e}': {} for m in ('hyde', 'query2doc', 'ragfusion') for e in RET}
for q in qids:
    pool, qq = POOL[q], pools[q]['query']; pos = pool[0]; hp, q2, rf = ld('hyde_plain', q), ld('query2doc', q), ld('ragfusion', q)
    if 'gemini' in RET:
        M = D[[DI[c] for c in pool]]; qv = QE[QI[q]]; gm = lambda v: [pool[i] for i in np.argsort(-(M @ (v / (np.linalg.norm(v) + 1e-9))))]
    if hp:
        ps = [cut(x) for x in hp['passages'] if x.strip()]
        if 'bm25' in RET: R['hyde-bm25'][q] = bm(pool, qq + ' ' + ' '.join(ps)).index(pos) + 1
        if 'gemini' in RET and ps: R['hyde-gemini'][q] = gm(np.mean(np.vstack([qv[None], emb(ps, 'RETRIEVAL_DOCUMENT')]), axis=0)).index(pos) + 1
    if q2:
        if 'bm25' in RET: R['query2doc-bm25'][q] = bm(pool, ' '.join([qq] * 5) + ' ' + cut(q2['passage'])).index(pos) + 1
        if 'gemini' in RET: R['query2doc-gemini'][q] = gm(emb([qq + ' [SEP] ' + cut(q2['passage'])], 'RETRIEVAL_QUERY')[0]).index(pos) + 1
    if rf:
        if 'bm25' in RET: R['ragfusion-bm25'][q] = rrf([bm(pool, x) for x in [qq] + rf['queries']]).index(pos) + 1
        if 'gemini' in RET: R['ragfusion-gemini'][q] = rrf([gm(qv)] + [gm(v) for v in emb(rf['queries'], 'RETRIEVAL_QUERY')]).index(pos) + 1
for e, m in (('bm25', 'bm25'), ('gemini', 'gemini_emb_001')):
    f = W / m / 'results/per_query.json'
    if e in RET and f.exists(): R = {f'original-{e}': json.load(open(f))['main'], **R}

KS = ('R@1', 'R@5', 'R@10', 'MRR@10', 'nDCG@10')
def met(rs): return {'R@1': statistics.mean(r <= 1 for r in rs), 'R@5': statistics.mean(r <= 5 for r in rs), 'R@10': statistics.mean(r <= 10 for r in rs),
                     'MRR@10': statistics.mean(1 / r if r <= 10 else 0 for r in rs), 'nDCG@10': statistics.mean(1 / math.log2(r + 1) if r <= 10 else 0 for r in rs)}
rng = random.Random(0); B = [[rng.choice(qids) for _ in qids] for _ in range(2000)]; out = {'n': len(qids), 'pool_size': N, 'dir': str(RW)}
for k, d in R.items():
    r = {q: d.get(q, N + 1) for q in qids}; pt = met(list(r.values())); bs = [met([r[q] for q in b]) for b in B]
    out[k] = {'missing': len(qids) - len(d), **{m: [round(pt[m], 4), round(sorted(x[m] for x in bs)[50], 4), round(sorted(x[m] for x in bs)[1949], 4)] for m in KS}}
(RW / 'results').mkdir(parents=True, exist_ok=True); json.dump(R, open(RW / 'results/ranks.json', 'w')); json.dump(out, open(RW / 'results/metrics.json', 'w'), indent=1)
print(f'{len(qids)} queries, pool {N} (missing output = miss)')
for k in R: print(f'{k:18s} missing {out[k]["missing"]:4d} | ' + ' | '.join(f'{m} {out[k][m][0]:.4f} [{out[k][m][1]:.3f}, {out[k][m][2]:.3f}]' for m in KS))
