#!/usr/bin/env python3
"""Stage 3: analogy judgment of every Stage 2 pair (paper Section 3.2, Step 2; prompt in Appendix: prompts/stage3_analogy_judgment.txt).
Model gpt-5.6-sol, reasoning effort none, Chat Completions on the flex tier with a 24 h prompt cache: the prompt is split at its
PROMPT_CACHE_BREAKPOINT marker into a developer message (shared instructions, cached) and a user message (the pair).
Pair slots: problem = q title + abstract; solution = d title + abstract; relevance = the citing sentences of q (deduplicated, blank-line
separated, 'N/A' if none); source / target domain = the rank-1 OpenAlex topic name of q / d ('Computer Science' / 'Unknown' if none).
Pass = decision observed_analogy or potential_analogy (read by stage4_transfer.py).
Input: <work>/stage2/prepared_pairs.json without pairs whose d is in config exclude_positives (a JSON list of corpus ids, e.g. the eval
positives for the training run). Output: <work>/s3/<pair_id>.json (record with source pair, parsed JSON, usage), <work>/s3/summary.json,
ledger <work>/cost_ledger.jsonl (tag s3). Resumable; stops starting calls at config caps.s3 (USD, cumulative over the saved records).
Before the bulk run, a second call must read cached prompt tokens (cache probe).
  python -I stage3_analogy.py CONFIG"""
import collections, hashlib, json, random, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, PROMPTS
import openai
from openai import OpenAI

MODEL, REASONING, PRICE = 'gpt-5.6-sol', 'none', {'input': 2.0, 'cached_input': 0.2, 'output': 10.0}
PROMPT = PROMPTS / 'stage3_analogy_judgment.txt'
MARKER = '<!-- PROMPT_CACHE_BREAKPOINT: pair-specific input begins below -->'
_local = threading.local()


def client():
    if not hasattr(_local, 'client'): _local.client = OpenAI(timeout=1800, max_retries=0)
    return _local.client


def prompt_parts():
    common, dynamic = PROMPT.read_text().split(MARKER, 1)
    return common.rstrip(), dynamic.lstrip()


def enrich(row):
    row = dict(row); q, d = row['q'], row['d']
    row['source_domain'] = ', '.join(q.get('fields') or []) or 'Computer Science'
    row['target_domain'] = ', '.join(d.get('fields') or []) or 'Unknown'
    wins = [w for w in (q.get('citing_windows') or []) if w.get('citing')]
    if wins:
        parts = ['\n'.join(f'[{k}] {w[k]}' for k in ('before', 'citing', 'after') if w.get(k)) for w in wins]
        row['relevance'] = '\n\n'.join(dict.fromkeys(parts))
    else:
        row['relevance'] = '\n\n'.join(dict.fromkeys(x for x in (q.get('citing_sentences') or []) if x)) or 'N/A'
    return row


def dynamic_prompt(template, row):
    q, d = row['q'], row['d']
    return template.format(source_domain=row['source_domain'], target_domain=row['target_domain'], problem=f'{q["title"]}\n\n{q.get("abstract") or ""}',
                           solution_title=d['title'], description=d.get('abstract') or 'N/A', relevance=row['relevance'])


def parse_json(text):
    try: return json.loads(text), None
    except Exception as first:
        s, e = text.find('{'), text.rfind('}') + 1
        if s >= 0 and e > s:
            try: return json.loads(text[s:e]), None
            except Exception as second: return None, str(second)
        return None, str(first)


def call_one(cache_key, common, dynamic, row):
    r = client().chat.completions.create(
        model=MODEL, messages=[{'role': 'developer', 'content': common}, {'role': 'user', 'content': dynamic}],
        reasoning_effort=REASONING, service_tier='flex', prompt_cache_key=cache_key, prompt_cache_retention='24h', max_completion_tokens=2000)
    text = r.choices[0].message.content or ''; parsed, error = parse_json(text); u = r.usage
    return {'pair_id': row['pair_id'], 'model': MODEL, 'endpoint': '/v1/chat/completions', 'service_tier_requested': 'flex',
            'service_tier_returned': getattr(r, 'service_tier', None), 'reasoning': REASONING, 'response_id': r.id, 'status': 'completed',
            'usage': {'input': u.prompt_tokens or 0, 'cache_read': getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0,
                      'output': u.completion_tokens or 0, 'reasoning': getattr(getattr(u, 'completion_tokens_details', None), 'reasoning_tokens', 0) or 0},
            'parsed': parsed, 'parse_error': error, 'output_text': text, 'source': row}


def cost(usage):
    return (max(0, usage['input'] - usage['cache_read']) * PRICE['input'] + usage['cache_read'] * PRICE['cached_input'] + usage['output'] * PRICE['output']) / 1e6


if __name__ == '__main__':
    C = load_config(sys.argv[1]); W = Path(C['work']); OUT = W / 's3'; OUT.mkdir(exist_ok=True); CAP = C['caps']['s3']
    excl = {str(x) for x in json.load(open(C['exclude_positives']))} if C.get('exclude_positives') else set()
    rows = [enrich(x) for x in json.load(open(W / 'stage2' / 'prepared_pairs.json')) if str(x['target_id']) not in excl]
    assert len({x['pair_id'] for x in rows}) == len(rows), 'duplicate pair_id'
    common, template = prompt_parts(); phash = hashlib.sha256(PROMPT.read_bytes()).hexdigest(); cache_key = f'cpr-stage3-{phash[:24]}-{MODEL}'
    lock = threading.Lock(); spent = [sum(cost(json.loads(f.read_text())['usage']) for f in OUT.glob('*.json') if f.name != 'summary.json')]
    NOT_SERVED = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)

    def one(row):
        with lock:
            if spent[0] >= CAP: return None
        for k in range(4):   # retry only requests that were not served
            try: rec = call_one(cache_key, common, dynamic_prompt(template, row), row); break
            except NOT_SERVED as e:
                if k == 3: raise
                print(row['pair_id'], 'not served, waiting', type(e).__name__, flush=True); time.sleep(30 * (k + 1) + random.random())
        (OUT / f"{row['pair_id']}.json").write_text(json.dumps(rec, ensure_ascii=False) + '\n'); usd = cost(rec['usage'])
        with lock:
            spent[0] += usd
            with open(W / 'cost_ledger.jsonl', 'a') as lf:
                lf.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tag': 's3', 'pair': row['pair_id'], 'model': MODEL, **rec['usage'], 'usd': round(usd, 6)}) + '\n')
        return rec
    todo = [r for r in rows if not (OUT / f"{r['pair_id']}.json").exists()]
    print(f'pairs {len(rows)} | to do {len(todo)} | spent so far ${spent[0]:.2f} | cap ${CAP}', flush=True)
    if todo:   # cache probe: first call alone, the second must read the cached prefix
        one(todo[0]); rec2 = one(todo[1]) if len(todo) > 1 else None
        if rec2 is not None and rec2['usage']['cache_read'] <= 0: sys.exit('cache probe returned zero cached tokens; bulk run not started')
        n = 2
        with ThreadPoolExecutor(C['workers']['s3']) as ex:
            for f in as_completed([ex.submit(one, r) for r in todo[2:]]):
                try: f.result()
                except Exception as e: print('FAILED (not retried):', type(e).__name__, str(e)[:200], flush=True)
                n += 1
                if n % 200 == 0: print(f'{n}/{len(todo)} spent ${spent[0]:.2f}', flush=True)
    recs = [json.loads((OUT / f"{r['pair_id']}.json").read_text()) for r in rows if (OUT / f"{r['pair_id']}.json").exists()]
    summary = {'model': MODEL, 'reasoning': REASONING, 'prompt_sha256': phash, 'requested': len(rows), 'completed': len(recs),
               'decisions': dict(collections.Counter((x.get('parsed') or {}).get('decision', 'invalid') for x in recs)),
               'tiers': dict(collections.Counter(x.get('service_tier_returned') for x in recs)),
               'usage': {k: sum(x['usage'].get(k, 0) or 0 for x in recs) for k in ('input', 'cache_read', 'output', 'reasoning')},
               'total_cost_usd': round(sum(cost(x['usage']) for x in recs), 4), 'cap_usd': CAP}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n'); print(json.dumps(summary, indent=1), flush=True)
