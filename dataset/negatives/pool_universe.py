#!/usr/bin/env python3
"""Stage 6: candidate pooling for hard negatives (paper Section 3.3, Step 1).
  spec   CONFIG [--pairs FILE]   one row per final pair (default <work>/final_pairs.json)  -> <work>/s6/queries.json
  score  CONFIG MODEL            MODEL in e5, splade, scincl: per query and group, the top POOL_K documents  -> <work>/s6/top300_<MODEL>.json
  join   CONFIG MODEL            joins the QSHARD parts of `score`  -> <work>/s6/top300_<MODEL>.json
  merge  CONFIG [--retrievers a,b]  round-robin merge of the retrievers' lists  -> <work>/s6/candidates.json (judge_negatives.py input)
Groups (strata), by OpenAlex topics rank 1-3 (<corpus>/universe_f.parquet), over the papers of the universe published before q
(year < q year):
  q_side   the document shares a topic with q
  pos_side the document shares no topic with q and a topic with the positive d
  other    the rest
Per query (spec): q_cid, pos_cid, q_year, q / pos topics, exclude_cids = the positives of q's other pairs that passed Stage 3, and
d1_cids = q's other Stage 2 pairs that Stage 3 rejected (papers q cites, on other topics, with a citing sentence). q, the positive,
exclude_cids and d1_cids never enter the retriever lists; judge_negatives.py judges d1_cids first (in its group by topics, no year rule).
Query encoding (CPU, same model settings as encode_corpus.py, max 512 tokens): E5 'query: ' + query, mean pooling, L2-normalized;
SciNCL [CLS]; SPLADE log(1 + relu) max pooling. Scores: dot product (sparse x dense for SPLADE) against <vectors>/<MODEL>/emb_*.pt.
Each group keeps a running top POOL_K (env, default 300) per query. merge: for rank r = 1..POOL_K, for each retriever in the order
scincl, splade, e5 (those with a top300 file, or --retrievers), the r-th document of its list is appended unless already present.
Env: POOL_K (300), QSHARD=i/n (score every n-th query from i, one process each; then `join`), THREADS (torch CPU threads, 16).
  python -I pool_universe.py spec CONFIG ; python -I pool_universe.py score CONFIG e5 ; python -I pool_universe.py merge CONFIG"""
import collections, heapq, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, records
sys.path.insert(0, str(Path(__file__).resolve().parent)); from retrievers import NAME, REVISION
CMD = sys.argv[1]; C = load_config(sys.argv[2]); W = Path(C['work']); LOC = W / 's6'; LOC.mkdir(parents=True, exist_ok=True)
UNI = Path(C['corpus']); UF = UNI / 'universe_f.parquet'; VEC = Path(C['vectors']); K = int(os.environ.get('POOL_K', 300))
STRATA = ('q_side', 'pos_side', 'other')


def opt(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def oa_topics(ids):
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    t = pq.read_table(UF, columns=['corpusid', 'topics'])
    t = t.filter(pc.is_in(t.column('corpusid'), value_set=pa.array(sorted(ids), pa.int64())))
    return {c: (x or [])[:3] for c, x in zip(t.column('corpusid').to_pylist(), t.column('topics').to_pylist())}


if CMD == 'spec':
    S3 = records(W / 's3'); final = sorted(json.load(open(opt('--pairs') or W / 'final_pairs.json')))
    V = {p: json.load(open(W / 's5' / 'stage5_v1a' / f'{p}.json'))['rewritten_problem'] for p in final}
    pos3 = {p for p, r in S3.items() if (r.get('parsed') or {}).get('decision') in ('observed_analogy', 'potential_analogy')}
    byq = collections.defaultdict(list)
    for p, r in S3.items(): byq[r['source']['base_id']].append(p)
    T = oa_topics({int(S3[p]['source'][k]) for p in final for k in ('base_id', 'target_id')})
    Q = []
    for p in final:
        s = S3[p]['source']; q, d = int(s['base_id']), int(s['target_id']); sib = [k for k in byq[s['base_id']] if k != p]
        Q.append({'pair_id': p, 'query': V[p], 'q_year': int(s['q']['year']), 'q_cid': q, 'pos_cid': d, 'q_topics': T.get(q, []), 'pos_topics': T.get(d, []),
                  'exclude_cids': sorted(int(S3[k]['source']['target_id']) for k in sib if k in pos3),
                  'd1_cids': sorted(int(S3[k]['source']['target_id']) for k in sib if k not in pos3)})
    json.dump(Q, open(LOC / 'queries.json', 'w'), indent=1)
    print('queries', len(Q), '| without q topics', sum(not x['q_topics'] for x in Q), '| without pos topics', sum(not x['pos_topics'] for x in Q))

elif CMD == 'score':
    import torch, pyarrow.parquet as pq
    torch.set_num_threads(int(os.environ.get('THREADS', 16)))
    M = sys.argv[3]; Q = json.load(open(LOC / 'queries.json'))
    QS = os.environ.get('QSHARD'); qi, qn = (int(x) for x in QS.split('/')) if QS else (0, 1); Q = Q[qi::qn]
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(NAME[M], revision=REVISION[M]); mdl = (AutoModelForMaskedLM if M == 'splade' else AutoModel).from_pretrained(NAME[M], revision=REVISION[M]).eval()
    with torch.no_grad():
        texts = [('query: ' + q['query']) if M == 'e5' else q['query'] for q in Q]
        b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt'); o = mdl(**b)
        if M == 'scincl': qv = o.last_hidden_state[:, 0].float()
        elif M == 'e5': m = b['attention_mask'][..., None].float(); qv = torch.nn.functional.normalize((o.last_hidden_state * m).sum(1) / m.sum(1), dim=-1)
        else: qv = (torch.log1p(torch.relu(o.logits.float())) * b['attention_mask'][..., None]).max(1).values
    qt = [set(q['q_topics']) for q in Q]; pt = [set(q['pos_topics']) for q in Q]; yr = np.array([q['q_year'] for q in Q])
    skip = [{q['q_cid'], q['pos_cid'], *q['exclude_cids'], *q['d1_cids']} for q in Q]
    heaps = [{s: [] for s in STRATA} for _ in Q]
    t0 = time.time(); files = sorted((VEC / M).glob('emb_*.pt')); cur = None
    for i, f in enumerate(files):
        shard, off = f.stem[4:].rsplit('_', 1)[0], int(f.stem.rsplit('_', 1)[1])
        if shard != cur:
            T = pq.read_table(UNI / 'export' / f'{shard}.parquet', columns=['corpusid', 'year', 'topics']); cur = shard
            Ycol = T.column('year').to_numpy(); Tcol = T.column('topics').to_pylist(); Ccol = T.column('corpusid').to_numpy()
        E = torch.load(f); cids = E['corpusid'].numpy(); n = len(cids)
        assert (cids == Ccol[off:off + n]).all(), f
        D = E['dense'].float() if 'dense' in E else E['sparse'].to(torch.float32)
        S = (qv @ D.T).numpy() if 'dense' in E else torch.sparse.mm(D, qv.T).T.numpy()
        years = Ycol[off:off + n]
        post = collections.defaultdict(list)   # topic -> row indices in this block
        for r_, x in enumerate(Tcol[off:off + n]):
            for t in (x or [])[:3]: post[t].append(r_)
        pos_of = {int(c): r_ for r_, c in enumerate(cids)}
        for j in range(len(Q)):
            valid = years < yr[j]
            for c in skip[j]:
                if c in pos_of: valid[pos_of[c]] = False
            qs = np.zeros(n, bool); ps = np.zeros(n, bool)
            for t in qt[j]: qs[post.get(t, [])] = True
            for t in pt[j]: ps[post.get(t, [])] = True
            ps &= ~qs
            for st, m in (('q_side', qs & valid), ('pos_side', ps & valid), ('other', ~qs & ~ps & valid)):
                idx = np.flatnonzero(m)
                if not len(idx): continue
                sc = S[j][idx]
                top = idx[np.argpartition(-sc, min(K, len(idx)) - 1)[:K]] if len(idx) > K else idx
                h = heaps[j][st]
                for r_ in top:
                    v = float(S[j][r_])
                    if len(h) < K: heapq.heappush(h, (v, int(cids[r_])))
                    elif v > h[0][0]: heapq.heapreplace(h, (v, int(cids[r_])))
        if (i + 1) % 20 == 0: print(f'{M} {i + 1}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    out = {q['pair_id']: {s: [c for sc, c in sorted(h, reverse=True)] for s, h in heaps[j].items()} for j, q in enumerate(Q)}
    json.dump(out, open(LOC / (f'top300_{M}.part{qi}of{qn}.json' if QS else f'top300_{M}.json'), 'w'))
    print(f'{M} done {time.time() - t0:.0f}s | per-group count min', min(len(v) for o in out.values() for v in o.values()), flush=True)

elif CMD == 'join':
    M = sys.argv[3]; parts = sorted(LOC.glob(f'top300_{M}.part*of*.json')); n = int(parts[0].stem.rsplit('of', 1)[1]) if parts else 0
    assert parts and len(parts) == n, f'{M}: {len(parts)} of {n} parts'
    out = {}
    for f in parts: out.update(json.load(open(f)))
    nq = len(json.load(open(LOC / 'queries.json'))); assert len(out) == nq, f'{M}: {len(out)} of {nq} queries'
    json.dump(out, open(LOC / f'top300_{M}.json', 'w')); print(f'{M} joined {len(parts)} parts, {len(out)} queries', flush=True)

elif CMD == 'merge':
    Q = json.load(open(LOC / 'queries.json'))
    models = [m for m in ('scincl', 'splade', 'e5') if (LOC / f'top300_{m}.json').exists() and (not opt('--retrievers') or m in opt('--retrievers').split(','))]
    R = {m: json.load(open(LOC / f'top300_{m}.json')) for m in models}; cand = {}
    for q in Q:
        p = q['pair_id']; cand[p] = {}
        for s in STRATA:   # round-robin over retrievers by rank, first occurrence kept
            seen, rows = set(), []
            for r in range(K):
                for m in models:
                    L = R[m][p][s]
                    if r < len(L) and L[r] not in seen: seen.add(L[r]); rows.append([L[r], m, r + 1])
            cand[p][s] = rows
    json.dump({'retrievers': models, 'candidates': cand}, open(LOC / 'candidates.json', 'w'))
    print('merged', len(cand), 'queries |', models, '| unique per group (median)', {s: sorted(len(cand[p][s]) for p in cand)[len(cand) // 2] for s in STRATA})
else: sys.exit(__doc__)
