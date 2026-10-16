#!/usr/bin/env python3
"""LLM reranker of the main table: MiMo-V2.6-Pro (OpenRouter) reorders each query's 151 candidates in a tournament.
  shuffle   pool = positive + q_side + pos_side + other (prepare.py order), shuffled with random.Random(f'upper-{sid}'), sid = the query id
            with both corpus ids cut to their first 8 digits (scaleup_<q cid[:8]>_<pos cid[:8]>; unique over the 1,300 queries)
  round 1   8 batches (docs[i::8], 18-19 docs) -> listwise prompt, top 5 of each        -> 40   (reasoning effort low)
  round 2   2 batches of the 40 (w1[i::2], 20 docs) -> listwise prompt, top 5 of each  -> 10   (reasoning effort low)
  final     each of the 10 finalists scored alone, 0-10 (pointwise prompt, model default reasoning); unparsed score = -1
  ties      several finalists sharing the top score -> one listwise call (k = number tied, default reasoning) orders them; tied
            finalists it does not return follow in round-2 order; the other finalists by score, then round-2 order
  ranking   finalists, then round-2 losers (interleaved over the 2 batches by position in the batch: the model's order, then the
            batch's input order), then round-1 losers (same interleaving over the 8 batches)
Listwise output = last {"top": [...]} JSON (1-based ids into the batch, invalid / duplicate ids dropped, first k kept); a round in which
no batch parsed keeps each batch's first 5 in input order. Calls: max_tokens 32000, 4 attempts (sleep 20 s, 40 s, 60 s), 'usage.include'
for the cost; the 8 (2, 10) calls of a step run at the same time, 32 queries at a time. Every call is appended to WORK/<name>/ledger.jsonl;
the run stops at --cap USD or when a call returns no choices 4 times (finished queries are kept; a rerun resumes).
Inputs: WORK/_data/{pools.json, texts.json} from evaluation/main/prepare.py (document text "title. abstract" as in the eval set).
Outputs WORK/<name>/: queries/<query_id>.json (full ranking, scores, calls, cost), results/scores.json ({query_id: {cid: 151 - position}},
the format of evaluation/main/score.py), results/ranks.json ({query_id: rank of the positive}). OPENROUTER_API_KEY from the environment
or a .env file (--env-file, default <repo>/.env).
  python -I rerank.py --work WORK [--name llm_reranker_mimo] [--model xiaomi/mimo-v2.6-pro] [--cap 120] [--limit N] [--workers 32]"""
import argparse, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--name', default='llm_reranker_mimo')
ap.add_argument('--model', default='xiaomi/mimo-v2.6-pro'); ap.add_argument('--rounds-effort', default='low', help="reasoning effort of the two selection rounds ('default' = no reasoning parameter)")
ap.add_argument('--cap', type=float, default=120.0, help='stop when this many USD are spent in this run'); ap.add_argument('--limit', type=int, default=0, help='first N queries (sorted ids), 0 = all')
ap.add_argument('--workers', type=int, default=32); ap.add_argument('--max-tokens', type=int, default=32000); ap.add_argument('--env-file', default=str(HERE.parents[1] / '.env')); A = ap.parse_args()
key = os.environ.get('OPENROUTER_API_KEY', '')
if not key and Path(A.env_file).exists(): key = next((l.split('=', 1)[1].strip().strip('"\'') for l in open(A.env_file) if l.startswith('OPENROUTER_API_KEY=')), '')
assert key, 'set OPENROUTER_API_KEY (environment or .env)'
cl = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=key)
PL = (HERE / 'prompts/listwise.txt').read_text(); PP = (HERE / 'prompts/pointwise.txt').read_text()
W = Path(A.work); T = json.load(open(W / '_data/texts.json')); pools = json.load(open(W / '_data/pools.json'))
OUT = W / A.name; (OUT / 'queries').mkdir(parents=True, exist_ok=True); (OUT / 'results').mkdir(exist_ok=True); LED = OUT / 'ledger.jsonl'
sid = lambda q: '_'.join(q.split('_')[:1] + [c[:8] for c in q.split('_')[1:]])   # shuffle seed id (docstring)
QIDS = sorted(pools)[:A.limit or None]; LOW = A.rounds_effort != 'default'; spent = [0.0]; lock = threading.Lock()

def gen(qid, text, low, what):
    if spent[0] >= A.cap: raise RuntimeError('cap reached')
    for t in range(4):
        try:
            r = cl.chat.completions.create(model=A.model, messages=[{'role': 'user', 'content': text}], max_tokens=A.max_tokens,
                                           extra_body={'usage': {'include': True}, **({'reasoning': {'effort': A.rounds_effort}} if low else {})})
            if not r.choices: raise RuntimeError('no choices')
            break
        except Exception:
            if t < 3: time.sleep(20 * (t + 1)); continue
            raise
    u = r.usage; tin, tout = u.prompt_tokens or 0, u.completion_tokens or 0; usd = float(getattr(u, 'cost', None) or (u.model_extra or {}).get('cost') or 0)
    with lock:
        spent[0] += usd
        with open(LED, 'a') as f: f.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': A.model + (f'@{A.rounds_effort}' if low else ''), 'in': tin, 'out': tout, 'usd': round(usd, 6), 'note': f'{qid} {what}'}) + '\n')
    return (r.choices[0].message.content or ''), {'in': tin, 'out': tout, 'usd': usd, 'low': low, 'what': what}

def listwise(qid, docs, k, low, what):
    txt, info = gen(qid, PL.format(query=pools[qid]['query'], k=k, candidates='\n'.join(f'[{i + 1}] {T[str(c)]}' for i, c in enumerate(docs))), low, what)
    m = re.findall(r'\{\s*"top"\s*:\s*\[[^\]]*\]\s*\}', txt); ids = []
    if m:
        for v in json.loads(m[-1])['top']:
            try: i = int(v)
            except (TypeError, ValueError): continue
            if 1 <= i <= len(docs) and docs[i - 1] not in ids: ids.append(docs[i - 1])
    info['parsed'] = bool(m); return ids[:k], info

def score(qid, c):
    txt, info = gen(qid, PP.format(query=pools[qid]['query'], candidate=T[str(c)]), False, 'point')
    m = re.findall(r'\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*\}', txt); info['parsed'] = bool(m)
    return (float(m[-1]) if m else -1.0), info

def one(qid):
    f = OUT / 'queries' / f'{qid}.json'
    if f.exists(): return json.load(open(f))
    x = pools[qid]; order = [x['pos_cid']] + [c for s in ('q_side', 'pos_side', 'other') for c in x['main'][s]]; random.Random(f'upper-{sid(qid)}').shuffle(order)
    calls = []; t0 = time.time()
    def rnd(docs, nb, k, what):
        w, losers = [], []; bs = [docs[i::nb] for i in range(nb)]
        with ThreadPoolExecutor(len(bs)) as ex: outs = list(ex.map(lambda b: listwise(qid, b, k, LOW, what), bs))
        for b, (top, info) in zip(bs, outs): calls.append(info); w += top; losers.append(top + [c for c in b if c not in top])
        return w, losers
    w1, l1 = rnd(order, 8, 5, 'r1')
    if not w1: w1 = [c for b in l1 for c in b[:5]]   # no batch parsed: each batch's first 5 in input order
    w2, l2 = rnd(w1, 2, 5, 'r2')
    if not w2: w2 = [c for b in l2 for c in b[:5]]
    sc = {}
    with ThreadPoolExecutor(len(w2)) as ex: outs = list(ex.map(lambda c: score(qid, c), w2))
    for c, (s_, info) in zip(w2, outs): sc[c] = s_; calls.append(info)
    best = max(sc.values()); tied = [c for c in w2 if sc[c] == best]; head = tied
    if len(tied) > 1:
        top, info = listwise(qid, tied, len(tied), False, 'tie'); calls.append(info); head = top + [c for c in tied if c not in top]
    final = head + sorted([c for c in w2 if c not in tied], key=lambda c: (-sc[c], w2.index(c)))
    rest2 = [c for i in range(20) for b in l2 for c in b[i:i + 1] if c not in final]
    rest1 = [c for i in range(19) for b in l1 for c in b[i:i + 1] if c not in final and c not in rest2]
    ranking = final + rest2 + rest1; assert sorted(ranking) == sorted(order), qid
    rec = {'query_id': qid, 'pos_rank': ranking.index(x['pos_cid']) + 1,
           'round_out': 'final' if x['pos_cid'] in final else ('round2' if x['pos_cid'] in rest2 else 'round1'), 'scores': {c: sc[c] for c in w2},
           'tied_top': len(tied), 'calls': len(calls), 'unparsed': sum(not c['parsed'] for c in calls), 'usd': round(sum(c['usd'] for c in calls), 4),
           'usd_rounds': round(sum(c['usd'] for c in calls if c['low']), 4), 'secs': round(time.time() - t0), 'ranking': ranking}
    json.dump(rec, open(f, 'w'), indent=1); return rec

def safe(qid):   # an API error of a query is logged and skipped (a rerun fills it in); a RuntimeError (cap reached, no choices) stops the run
    try: return one(qid)
    except RuntimeError: raise
    except Exception as e: print('query failed', qid, type(e).__name__, e, flush=True)
try:
    with ThreadPoolExecutor(min(A.workers, len(QIDS))) as ex: list(ex.map(safe, QIDS))
except RuntimeError as e: print('stopped:', e)
R = {q: json.load(open(OUT / 'queries' / f'{q}.json')) for q in QIDS if (OUT / 'queries' / f'{q}.json').exists()}
json.dump({q: {c: len(r['ranking']) - i for i, c in enumerate(r['ranking'])} for q, r in R.items()}, open(OUT / 'results/scores.json', 'w'))
json.dump({q: r['pos_rank'] for q, r in R.items()}, open(OUT / 'results/ranks.json', 'w'), indent=0)
n = len(R); rs = [r['pos_rank'] for r in R.values()]
print(f'{n} of {len(QIDS)} queries ranked | R@10 {sum(x <= 10 for x in rs) / max(1, n):.4f} | positive out at',
      {k: sum(r['round_out'] == k for r in R.values()) for k in ('final', 'round2', 'round1')}, '| top ties', sum(r['tied_top'] > 1 for r in R.values()),
      f"| cost ${sum(r['usd'] for r in R.values()):.2f} (this run ${spent[0]:.2f}) | unparsed calls {sum(r['unparsed'] for r in R.values())}")
