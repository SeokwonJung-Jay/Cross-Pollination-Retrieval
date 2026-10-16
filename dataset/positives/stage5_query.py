#!/usr/bin/env python3
"""Stage 5: query generation with a review gate (paper Section 3.2, Step 3; Appendix: query generation).
Targets = Stage 4 pass (parsed verdict present and != 'reject'). Model gpt-6-astra, Responses API, default reasoning, standard tier,
max_output_tokens 4000. Per pair:
  1. problem extraction, prompts/local_problem_e1.txt: q title, abstract ('No abstract available' if none), the citing sentences with
     in-text citations replaced by [CITATION] (one per line, 'N/A' if none), and `contribution` = what q took from d: the first segment of
     the Stage 3 transfer_path split at '->' when the path has at least 3 segments, else the whole path;
  2. rewriting, prompts/rewrite_problem_v3.txt: problem, base_domain = d's rank-1 OpenAlex topic name ('Unknown'), target_domain = q's
     ('Computer Science');
  3. review, prompts/query_review_v3.txt (JSON problem_only / grounded / well_formed / issue) + rule flags (auto_flags below).
     Pass = no rule flag and all three review fields true. A reply that is not JSON is asked once more, then counts as a failed review.
  Up to 2 regenerations of the failing pairs: extraction with prompts/local_problem_v4_e1.txt plus the previous statement and the
  reviewer's objection (or the rule flags), rewriting with a note to keep the conditions the objection names; then a new review.
Outputs in <work>/s5/: stage5_v1a/<pair_id>.json (problem, rewritten_problem = the query, history), stage5_review.json (last review of
every pair), stage5_summary.json; ledger <work>/cost_ledger.jsonl (tag s5); stops starting calls at caps.s5.
  python -I stage5_query.py CONFIG"""
import json, re, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import load_config, prompt, ledger_total

MODEL, PRICE = 'gpt-6-astra', (10, 50)   # USD per million input / output tokens
CIT = re.compile(r"\((?:[A-Z][^()]{0,80}?\d{4}[a-z]?(?:;\s*)?)+\)|\b[A-Z][A-Za-z\-]+(?: et al\.| and [A-Z][A-Za-z\-]+)?,? \(\d{4}[a-z]?\)|\[\d{1,3}(?:[,\-–]\s*\d{1,3})*\]")
REVIEW_KEYS = ('problem_only', 'grounded', 'well_formed')


def auto_flags(q):
    f = []
    if 'CITATION' in q: f.append('citation_marker')
    if re.search(r'\b[A-Z]{6,}(?:_[A-Z]+)*\b|\b[A-Z]+_[A-Z_]+\b', q): f.append('caps_token')
    if re.search(r'\b(analog\w*|mapping\w*|instruction\w*)\b|\bgraph\b(?! neural| network| attention| convolution)', q, re.I): f.append('meta_phrase')
    if not q.strip().lower().startswith('how to'): f.append('not_how_to')
    if len(q.split()) > 45: f.append('too_long')
    if re.search(r'\b(by|through|using|via)\s+\w+ing\b|\busing (a|an|the)\b', q): f.append('method_phrase')
    return f


def fields(x):
    f = x.get('fields') or []
    return ', '.join(f) if isinstance(f, list) else str(f)


def passed(r): return not r['auto'] and all((r['review'] or {}).get(k) for k in REVIEW_KEYS)


if __name__ == '__main__':
    C = load_config(sys.argv[1]); W = Path(C['work']); OUT = W / 's5'; QOUT = OUT / 'stage5_v1a'; QOUT.mkdir(parents=True, exist_ok=True)
    LED = W / 'cost_ledger.jsonl'; CAP = C['caps']['s5']; NW = C['workers']['s5']
    union, records, contrib = [], {}, {}
    for f in sorted((W / 's4').glob('*.json')):
        j = json.load(open(f)); v = (j.get('parsed') or {}).get('verdict')
        if v is not None and v != 'reject':
            s3 = json.load(open(W / 's3' / f.name)); union.append(j['pair_id']); records[j['pair_id']] = s3['source']
            seg = [x.strip() for x in ((s3.get('parsed') or {}).get('transfer_path') or '').split('->') if x.strip()]
            path = (s3.get('parsed') or {}).get('transfer_path') or ''; contrib[j['pair_id']] = seg[0] if len(seg) >= 3 else path
    print('Stage 4 pass', len(union), flush=True)
    P_LOCAL, P_FIX, P_REW, P_REV = prompt('local_problem_e1'), prompt('local_problem_v4_e1'), prompt('rewrite_problem_v3'), prompt('query_review_v3')
    lock = threading.Lock(); spent = [ledger_total(LED, 's5')]; _oa = threading.local()

    def call(text, parse):
        """one gpt-6-astra call; not-served errors wait 60 s (up to 3 times); a non-JSON reply to a parsed call is asked once more"""
        from openai import OpenAI
        if not hasattr(_oa, 'c'): _oa.c = OpenAI(timeout=900, max_retries=0)
        for k in range(4):
            with lock:
                if spent[0] >= CAP: raise RuntimeError(f'spend cap ${CAP} reached')
            try:
                r = _oa.c.responses.create(model=MODEL, input=text, max_output_tokens=4000)
                out = (r.output_text or '').strip(); u = {'input': r.usage.input_tokens, 'output': r.usage.output_tokens, 'stop_reason': 'end_turn' if out else 'empty'}
                usd = (PRICE[0] * u['input'] + PRICE[1] * u['output']) / 1e6
                with lock:
                    spent[0] += usd
                    with open(LED, 'a') as lf:
                        lf.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tag': 's5', 'model': MODEL, 'in': u['input'], 'out': u['output'], 'usd': round(usd, 6), 'parse': parse}) + '\n')
                if not parse: return out, u
                st = out.find('{'); obj, _ = json.JSONDecoder().raw_decode(out[st:]) if st >= 0 else (json.loads(out), 0)
                return obj, u
            except json.JSONDecodeError:
                if k == 0: print('unparsable JSON reply, asking once more', flush=True); continue
                return {'problem_only': False, 'grounded': False, 'well_formed': False, 'issue': 'review reply was not valid JSON'}, {'input': 0, 'output': 0}
            except Exception as e:
                m = str(e).lower()
                if k < 3 and any(x in m for x in ('overloaded', '529', 'rate_limit', '429', 'connection', 'internal server', '500', '502', '503')):
                    print('not served, waiting 60 s:', type(e).__name__, flush=True); time.sleep(60); continue
                raise

    def generate(pid, issue=''):
        p = records[pid]; q, d = p['q'], p['d']
        ctx = '\n'.join(CIT.sub('[CITATION]', s) for s in q.get('citing_sentences') or []) or 'N/A'
        kw = dict(title=q['title'], abstract=q.get('abstract') or 'No abstract available', citation_context=ctx, contribution=contrib.get(pid, ''))
        pr = P_LOCAL.format(**kw)
        if issue:
            prev = json.loads((QOUT / f'{pid}.json').read_text())
            pr = P_FIX.format(**kw) + (f"\nA previous attempt was rejected. Previous problem statement: {prev.get('problem', '')}\nReviewer's objection: {issue}"
                                       "\nWrite a corrected problem statement that resolves the objection.")
        problem, u1 = call(pr, False)
        hide, target = fields(d) or 'Unknown', fields(q) or 'Computer Science'
        rpr = P_REW.format(problem=problem, base_domain=hide, target_domain=target)
        if issue: rpr += f'\nA previous rewrite was rejected because: {issue} Keep the conditions of the problem that this objection names, while still stating only the goal.'
        rewritten, u2 = call(rpr, False)
        prev = json.loads((QOUT / f'{pid}.json').read_text()) if (QOUT / f'{pid}.json').exists() else {}
        rec = {'pair_id': pid, 'q_id': p['base_id'], 'd_id': p['target_id'], 'variant': 'V1a', 'hidden_domain': hide, 'target_domain': target,
               'citation_context_masked': ctx, 'problem': problem, 'rewritten_problem': rewritten, 'model': MODEL, 'usage': {'extract': u1, 'rewrite': u2},
               'history': prev.get('history', [])}
        if issue: rec['history'].append({'problem': prev.get('problem'), 'query': prev.get('rewritten_problem'), 'issue': issue})
        (QOUT / f'{pid}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n')

    def review(pid):
        p, j = records[pid], json.loads((QOUT / f'{pid}.json').read_text()); q = p['q']
        out, u = call(P_REV.format(title=q['title'], abstract=q.get('abstract') or '', citation_context=j['citation_context_masked'],
                                   query=j['rewritten_problem'], local_problem=j.get('problem') or 'N/A'), True)
        return {'pair_id': pid, 'query': j['rewritten_problem'], 'auto': auto_flags(j['rewritten_problem']), 'review': out, 'usage': u}

    def pool(fn, args):
        with ThreadPoolExecutor(NW) as ex: return list(ex.map(lambda a: fn(*a) if isinstance(a, tuple) else fn(a), args))
    t0 = time.time(); rf = OUT / 'stage5_review.json'
    try:
        missing = [p for p in union if not (QOUT / f'{p}.json').exists()]
        pool(generate, missing)
        reviews = {x['pair_id']: x for x in (json.loads(rf.read_text()) if rf.exists() else [])}
        for rnd in range(3):   # review; rounds 2 and 3 regenerate the failing pairs first
            todo = [p for p in union if p not in reviews or not passed(reviews[p])]
            if not todo: break
            if rnd: pool(generate, [(p, (reviews[p]['review'] or {}).get('issue') or ', '.join(reviews[p]['auto'])) for p in todo])
            for r in pool(review, todo): reviews[r['pair_id']] = r
            rf.write_text(json.dumps([reviews[p] for p in union if p in reviews], ensure_ascii=False, indent=2) + '\n')
    except RuntimeError as e: print('stage 5 stopped:', e, flush=True)
    reviews = {x['pair_id']: x for x in (json.loads(rf.read_text()) if rf.exists() else [])}
    ok = [p for p in union if p in reviews and passed(reviews[p])]
    (OUT / 'stage5_summary.json').write_text(json.dumps({'targets': len(union), 'passed': len(ok), 'failed': sorted(set(union) - set(ok))}, indent=1))
    print(f'targets {len(union)} | passed {len(ok)} | {time.time() - t0:.0f}s | ledger ${spent[0]:.2f} of cap {CAP}', flush=True)
