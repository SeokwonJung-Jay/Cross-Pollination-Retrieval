#!/usr/bin/env python3
"""Stage 5b (same-problem dedup) and Stage 7b (positive usefulness u(q, d)) (paper Section 3.2, Step 3). Model gpt-6-astra, Responses API,
default reasoning, standard tier, max_output_tokens 4000; not-served requests are retried (30/60/90 s waits), served ones never.
  s5b  per base paper q, over its Stage 5 queries that passed the review gate: pairs sharing a citing sentence form one group (union-find);
       every two groups are compared with prompts/dedup_same_problem.txt (query_a / query_b) and merged when same_problem is true; a
       group of several pairs keeps the strongest one, chosen with prompts/dedup_strongest_analogy.txt (q title, abstract and per pair
       the cited title, abstract, citing sentences, Stage 3 object mappings, shared relations, transfer type); otherwise its first pair
       (sorted pair ids).  -> <work>/s5b/q_<q>.json {q, kept, merged, calls}
  s7b  every kept pair: prompts/stage7b_umbrela_mod_mapping.txt with query, passage = d title + '. ' + whitespace-normalized abstract[:2000],
       Stage 3 object mappings and shared relations; u = last 'final score: n'.  -> <work>/s7b/<pair_id>.json
       Final pairs = u >= 2  -> <work>/final_pairs.json
Ledger <work>/cost_ledger.jsonl (tags s5b, s7b); caps.s5b / caps.s7b are cumulative over the ledger.
  python -I dedup_usefulness.py CONFIG s5b|s7b"""
import collections, itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, prompt, records, Spend, ledger_total, ask_json, openai_call, final_score, run_items
C = load_config(sys.argv[1]); STAGE = sys.argv[2]; W = Path(C['work']); LED = W / 'cost_ledger.jsonl'; S5 = W / 's5'
spend = Spend(C['caps'][STAGE] - ledger_total(LED, STAGE), LED, STAGE)
REVIEW_KEYS = ('problem_only', 'grounded', 'well_formed')


def s5b():
    S3 = records(W / 's3')
    rev = {x['pair_id']: x for x in json.load(open(S5 / 'stage5_review.json'))}
    ok5 = [p for p, x in rev.items() if not x['auto'] and all((x['review'] or {}).get(k) for k in REVIEW_KEYS)]
    V = {p: json.load(open(S5 / 'stage5_v1a' / f'{p}.json')) for p in ok5}
    P1, P2 = prompt('dedup_same_problem'), prompt('dedup_strongest_analogy')
    byq = collections.defaultdict(list)
    for p in ok5: byq[S3[p]['source']['base_id']].append(p)
    items = [{'id': f'q_{q}', 'q': q, 'pairs': sorted(ps)} for q, ps in sorted(byq.items())]
    ctx = lambda k: set(S3[k]['source']['q'].get('citing_sentences') or [])

    def one(it):
        q, ks = it['q'], it['pairs']; par = {k: k for k in ks}; calls = []

        def f(x):
            while par[x] != x: x = par[x]
            return x
        for a, b in itertools.combinations(ks, 2):
            if ctx(a) & ctx(b): par[f(a)] = f(b)
        comp = lambda: {r: sorted(k for k in ks if f(k) == r) for r in {f(k) for k in ks}}
        for a, b in itertools.combinations(sorted(comp()), 2):
            ans, _, _ = ask_json('gpt-6-astra', P1.format(query_a=V[a]['rewritten_problem'], query_b=V[b]['rewritten_problem']), spend)
            calls.append({'type': 'same_problem', 'a': a, 'b': b, **(ans or {'parse_failed': True})})
            if (ans or {}).get('same_problem') is True: par[f(a)] = f(b)
        kept, merged = [], {}
        for r, members in comp().items():
            keep = members[0]
            if len(members) > 1:
                def cand(i, k):
                    d = S3[k]['source']['d']; p = S3[k]['parsed']
                    return (f"[{i + 1}] Cited work: {d['title']}\nAbstract: {d.get('abstract') or ''}\nCiting sentences: {' | '.join(sorted(ctx(k))) or 'N/A'}\n"
                            f"Object mappings (problem -> cited work): {'; '.join(m.get('source', '') + ' -> ' + m.get('target', '') for m in p.get('object_mappings') or [])}\n"
                            f"Shared relations: {'; '.join(x.get('problem', '') + ' <-> ' + x.get('solution', '') for x in p.get('shared_relations') or [])}\nTransfer type: {p.get('transfer_type')}")
                qq = S3[members[0]]['source']['q']
                ans, _, _ = ask_json('gpt-6-astra', P2.format(q_title=qq['title'], q_abstract=qq.get('abstract') or '', candidates='\n'.join(cand(i, k) for i, k in enumerate(members))), spend)
                calls.append({'type': 'strongest', 'members': members, **(ans or {'parse_failed': True})})
                if ans and str(ans.get('strongest', '')).isdigit() and 1 <= int(ans['strongest']) <= len(members): keep = members[int(ans['strongest']) - 1]
                for k in members:
                    if k != keep: merged[k] = keep
            kept.append(keep)
        return {'q': q, 'kept': sorted(kept), 'merged': merged, 'calls': calls}
    run_items('s5b', items, one, C['workers']['s5b'], W / 's5b')
    print('review pass', len(ok5), '-> kept', sum(len(r['kept']) for r in records(W / 's5b', 'q_').values()), flush=True)


def s7b():
    S3 = records(W / 's3'); P7 = prompt('stage7b_umbrela_mod_mapping')
    kept = sorted(k for r in records(W / 's5b', 'q_').values() for k in r['kept'])
    V = {p: json.load(open(S5 / 'stage5_v1a' / f'{p}.json')) for p in kept}

    def one(it):
        k = it['id']; p = S3[k]['parsed']; d = S3[k]['source']['d']
        maps = '; '.join(f"{m.get('source', '')} -> {m.get('target', '')}" for m in p.get('object_mappings') or []) or 'none'
        rels = '; '.join(f"{x.get('problem', '')} <-> {x.get('solution', '')}" for x in p.get('shared_relations') or []) or 'none'
        t, u = openai_call('gpt-6-astra', P7.format(query=V[k]['rewritten_problem'], passage=f"{d['title']}. {' '.join((d.get('abstract') or '').split())[:2000]}",
                                                    mappings=maps, relations=rels), spend)
        return {'pair_id': k, 'score': final_score(t), 'query': V[k]['rewritten_problem'], 'raw_tail': t[-300:], 'usage': u}
    run_items('s7b', [{'id': p} for p in kept], one, C['workers']['s7b'], W / 's7b')
    fin = sorted(p for p, r in records(W / 's7b').items() if (r.get('score') or 0) >= 2)
    json.dump(fin, open(W / 'final_pairs.json', 'w'), indent=1); print(f'kept {len(kept)} | u >= 2: {len(fin)} -> final_pairs.json', flush=True)


{'s5b': s5b, 's7b': s7b}[STAGE]()
