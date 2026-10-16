#!/usr/bin/env python3
"""Stage 4: transfer judgment of the Stage 3 positives (paper Section 3.2, Step 2; prompts/stage4_transfer_check.txt).
Targets = <work>/s3 records with decision observed_analogy or potential_analogy. The prompt gets q and d (title, abstract), the citing
sentences and the Stage 3 output (decision, object mappings, shared relations, transfer type / path, reason).
Model gpt-6.1-sol, reasoning effort low, Responses API, service tier config s4_tier (flex or default). Pass = verdict != 'reject'.
A request that was not served (429, connection, 5xx) is retried up to 8 times (waits 30, 60, ... s); other errors are logged and left.
Output <work>/s4/<pair_id>.json {pair_id, stage3_decision, parsed, raw, usage}; ledger <work>/cost_ledger.jsonl (tag s4); stops at caps.s4.
  python -I stage4_transfer.py CONFIG"""
import json, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, prompt, ledger_total
import openai
from openai import OpenAI
C = load_config(sys.argv[1]); W = Path(C['work']); S3 = W / 's3'; OUT = W / 's4'; OUT.mkdir(exist_ok=True); LED = W / 'cost_ledger.jsonl'
P = prompt('stage4_transfer_check'); TIER = C.get('s4_tier', 'flex'); PIN, POUT = (1, 5) if TIER == 'flex' else (2, 10); CAP = C['caps']['s4']
ids = sorted(f.stem for f in S3.glob('*.json') if f.name != 'summary.json' and (json.load(open(f)).get('parsed') or {}).get('decision') in ('observed_analogy', 'potential_analogy'))
cl = OpenAI(timeout=900, max_retries=0); lock = threading.Lock(); st = {'in': 0, 'out': 0, 'err': 0}; spent = [ledger_total(LED, 's4')]


def one(pid):
    f = OUT / f'{pid}.json'
    if f.exists(): return
    if spent[0] >= CAP:
        with lock: st['err'] += 1
        return
    s = json.load(open(S3 / f'{pid}.json')); src, p = s['source'], s['parsed']
    text = P.format(q_title=src['q']['title'], q_abstract=src['q'].get('abstract') or '', d_title=src['d']['title'], d_abstract=src['d'].get('abstract') or '',
        citing_sentences='\n'.join(src['q'].get('citing_sentences') or []) or 'N/A', decision=p.get('decision'),
        mappings='; '.join(f"{m.get('source')} -> {m.get('target')}" for m in p.get('object_mappings') or []) or 'none',
        relations='; '.join(f"{x.get('problem')} <-> {x.get('solution')}" for x in p.get('shared_relations') or []) or 'none',
        transfer_type=p.get('transfer_type'), transfer_path=p.get('transfer_path') or '', reason=p.get('reason') or '')
    for k in range(8):
        try: r = cl.responses.create(model='gpt-6.1-sol', input=text, max_output_tokens=4000, reasoning={'effort': 'low'}, service_tier=TIER); break
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            if k == 7:
                with lock: st['err'] += 1
                print(pid, 'error (not served after 8 tries)', type(e).__name__, flush=True); return
            time.sleep(30 * (k + 1))
        except Exception as e:
            with lock: st['err'] += 1
            print(pid, 'error', type(e).__name__, flush=True); return
    t = r.output_text or ''; j = None
    try: j = json.loads(t[t.find('{'):t.rfind('}') + 1])
    except Exception: pass
    usd = (r.usage.input_tokens * PIN + r.usage.output_tokens * POUT) / 1e6
    with lock:
        st['in'] += r.usage.input_tokens; st['out'] += r.usage.output_tokens; spent[0] += usd
        with open(LED, 'a') as lf:
            lf.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tag': 's4', 'pair': pid, 'model': 'gpt-6.1-sol', 'in': r.usage.input_tokens,
                                 'out': r.usage.output_tokens, 'usd': round(usd, 6), 'tier': TIER}) + '\n')
    json.dump({'pair_id': pid, 'stage3_decision': p.get('decision'), 'parsed': j, 'raw': t, 'usage': {'in': r.usage.input_tokens, 'out': r.usage.output_tokens}}, open(f, 'w'), indent=1)


with ThreadPoolExecutor(C['workers']['s4']) as ex: list(ex.map(one, ids))
print(f"targets {len(ids)} | errors or capped {st['err']} | new cost ${(st['in'] * PIN + st['out'] * POUT) / 1e6:.3f} | ledger total ${spent[0]:.2f} of cap {CAP}")
