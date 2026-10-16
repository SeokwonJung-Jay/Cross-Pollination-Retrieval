#!/usr/bin/env python3
"""Query rewriting (HyDE, Query2doc, RAG-Fusion) with GLM-5.3 on OpenRouter, for every query of the eval set.
  hyde_plain  prompts/hyde_plain.txt, 8 calls at temperature 0.7 -> 8 passages
  query2doc   prompts/query2doc.txt (4 in-context examples), 1 call at temperature 1.0 -> 1 passage
  ragfusion   prompts/ragfusion.txt, 1 call at temperature 0.7 -> lines with list markers ('1.', '1)', '-', '*') and quotes stripped,
              lines of more than 10 characters, first 4 = the generated queries (the raw answer is kept too)
Every call: reasoning effort low, max_tokens 4000 (the model's reasoning counts against it), 4 attempts (sleep 15 s, 30 s, 45 s); an empty
answer is an error. A failed query is logged and skipped; a rerun generates only the missing files. Every call is appended to
OUT/ledger.jsonl (cost = OpenRouter usage.cost); the run stops at --cap USD.
Input: WORK/_data/pools.json from evaluation/main/prepare.py. Output OUT = WORK/rewrite_<tag>/<method>/<query_id>.json.
OPENROUTER_API_KEY from the environment or a .env file (--env-file, default <repo>/.env).
  python -I generate.py --work WORK [--model z-ai/glm-5.3] [--tag glm53] [--cap 40] [--limit N] [--workers 32]"""
import argparse, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--model', default='z-ai/glm-5.3'); ap.add_argument('--tag', default='glm53')
ap.add_argument('--cap', type=float, default=40.0); ap.add_argument('--limit', type=int, default=0, help='random sample of N queries (seed 20261010), 0 = all')
ap.add_argument('--workers', type=int, default=32); ap.add_argument('--max-tokens', type=int, default=4000); ap.add_argument('--env-file', default=str(HERE.parents[1] / '.env')); A = ap.parse_args()
key = os.environ.get('OPENROUTER_API_KEY', '')
if not key and Path(A.env_file).exists(): key = next((l.split('=', 1)[1].strip().strip('"\'') for l in open(A.env_file) if l.startswith('OPENROUTER_API_KEY=')), '')
assert key, 'set OPENROUTER_API_KEY (environment or .env)'
cl = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=key)
OUT = Path(A.work) / f'rewrite_{A.tag}'; LED = OUT / 'ledger.jsonl'
P = {k: (HERE / f'prompts/{k}.txt').read_text() for k in ('hyde_plain', 'query2doc', 'ragfusion')}
for k in P: (OUT / k).mkdir(parents=True, exist_ok=True)
pools = json.load(open(Path(A.work) / '_data/pools.json')); rows = [{'query_id': q, 'query': pools[q]['query']} for q in sorted(pools)]
if A.limit: rows = random.Random(20261010).sample(rows, A.limit)
spent = [0.0]; lock = threading.Lock()

def gen(prompt, temperature, note):
    if spent[0] >= A.cap: raise RuntimeError('cap reached')
    for t in range(4):
        try:
            r = cl.chat.completions.create(model=A.model, messages=[{'role': 'user', 'content': prompt}], temperature=temperature, max_tokens=A.max_tokens,
                                           extra_body={'usage': {'include': True}, 'reasoning': {'effort': 'low'}})
            if not r.choices: raise RuntimeError('no choices')
            break
        except Exception:
            if t == 3: raise
            time.sleep(15 * (t + 1))
    u = r.usage; usd = float(getattr(u, 'cost', None) or (u.model_extra or {}).get('cost') or 0)
    with lock:
        spent[0] += usd
        with open(LED, 'a') as f: f.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': A.model, 'in': u.prompt_tokens, 'out': u.completion_tokens, 'usd': round(usd, 6), 'note': f'{note} finish={r.choices[0].finish_reason}'}) + '\n')
    t = (r.choices[0].message.content or '').strip()
    if not t: raise RuntimeError(f'empty response, finish {r.choices[0].finish_reason}')
    return t

def one(r):
    qid, q = r['query_id'], r['query']; base = {'query_id': qid, 'query': q, 'model': A.model}
    f = OUT / 'hyde_plain' / f'{qid}.json'
    if not f.exists():
        with ThreadPoolExecutor(8) as ex: ps = list(ex.map(lambda i: gen(P['hyde_plain'].format(query=q), 0.7, f'{qid} hyde_plain'), range(8)))
        json.dump({**base, 'passages': ps}, open(f, 'w'), ensure_ascii=False, indent=1)
    f = OUT / 'query2doc' / f'{qid}.json'
    if not f.exists(): json.dump({**base, 'passage': gen(P['query2doc'].format(query=q), 1.0, f'{qid} query2doc')}, open(f, 'w'), ensure_ascii=False, indent=1)
    f = OUT / 'ragfusion' / f'{qid}.json'
    if not f.exists():
        t = gen(P['ragfusion'].format(query=q), 0.7, f'{qid} ragfusion')
        qs = [re.sub(r'^\s*(?:\d+[.)]|[-*])\s*', '', l).strip().strip('"') for l in t.splitlines() if l.strip()]
        json.dump({**base, 'queries': [x for x in qs if len(x) > 10][:4], 'raw': t}, open(f, 'w'), ensure_ascii=False, indent=1)

fails = []
def safe(r):
    try: one(r)
    except Exception as e:
        with lock: fails.append(r['query_id']); print('FAILED', r['query_id'], type(e).__name__, str(e)[:120], flush=True)
with ThreadPoolExecutor(A.workers) as ex: list(ex.map(safe, rows))
print(f'done {len(rows) - len(fails)} of {len(rows)} | failed {len(fails)} (rerun to fill in) | spent this run ${spent[0]:.2f}')
