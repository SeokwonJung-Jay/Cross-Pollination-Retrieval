#!/usr/bin/env python3
"""Stage 7: hard-negative judgment (paper Section 3.3, Step 2; prompts/stage7_umbrela_mod.txt).
Per query and group (q_side, pos_side, other) the judge order is: q's d1 candidates first (pool_universe.py spec; grouped by their
OpenAlex topics, kept only if in the universe), then the merged Stage 6 candidates in round-robin order. Candidates are judged in
rounds until the group has config s7_per_stratum usable negatives (u <= 1) or runs out of candidates. A round asks, per short group,
ceil(short / y * 1.1) more judgments in judge order, y = 0.68, or with s7_adaptive the group's own usable rate once 20 are judged (at
least 0.03); without s7_adaptive there are at most 7 rounds. With s7_adaptive a round is trimmed to what the remaining cap pays
(s7_per_j_r1 / s7_per_j_later USD per judgment), round-robin over (query, group).
Judge: gpt-6.1-sol, default reasoning, query = Stage 5 query, passage = title + '. ' + whitespace-normalized abstract[:2000] from
<corpus>/export; u = last 'final score: n'; an unparsed reply is not usable. s7_mode: batch (OpenAI Batch API, /v1/responses, batches of
at most 20,000 requests; requests a batch did not serve go to the next round), batch_first (round 1 batch, then standard), standard, flex.
Negatives = the first s7_per_stratum usable candidates per group in judge order (CPR-eval 50, CPR-train 3).
Outputs: <work>/s7/<pair_id>__<cid>.json (one per judgment; existing ones are reused), <work>/s7/<s7_pools_name, default
_pools_main.json> {pair_id: {query, pos_cid, q_cid, main: {group: [cid]}, usable: {group: [every usable cid]}}}, <work>/s7/<s7_short_name,
default _short.json> [[pair_id, group, n]] for groups with fewer negatives; text cache <work>/s7_texts.json; ledger tag s7.
Optional s7_only: JSON list of pair ids to judge (others in s6 are left out).
  python -I judge_negatives.py CONFIG [--smoke N]"""
import collections, json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, prompt, Spend, ledger_total, openai_call, final_score, run_items
C = load_config(sys.argv[1]); SMOKE = int(sys.argv[sys.argv.index('--smoke') + 1]) if '--smoke' in sys.argv else 0
W = Path(C['work']); LED = W / 'cost_ledger.jsonl'; UNI = Path(C['corpus']); STRATA = ('q_side', 'pos_side', 'other')


def batch_round(rnd, todo, T, P7, Q, od, spend):
    """One round through the OpenAI Batch API: batches of at most 20,000 requests, submitted together and polled every 5 minutes.
    Batch ids are kept in od/_batch_<round>.json so a restarted run resumes polling the same round."""
    from openai import OpenAI
    cl = OpenAI(max_retries=2); t0 = time.time(); meta = od / f'_batch_{rnd}.json'; CH = 20000
    if meta.exists() and json.load(open(meta)).get('n') == len(todo):
        bids = json.load(open(meta))['ids']
    elif meta.exists():
        meta.rename(meta.with_name(f'{meta.stem}.done_{int(time.time())}.json')); return batch_round(rnd, todo, T, P7, Q, od, spend)
    else:
        bids = []
        for k in range(0, len(todo), CH):
            req = W / f's7_batch_{rnd}_{k // CH}.jsonl'
            with open(req, 'w') as fo:
                for p, c in todo[k:k + CH]:
                    fo.write(json.dumps({'custom_id': f'{p}__{c}', 'method': 'POST', 'url': '/v1/responses',
                                         'body': {'model': 'gpt-6.1-sol', 'input': P7.format(query=Q[p]['query'], passage=T[c]), 'max_output_tokens': 4000}}) + '\n')
            fid = cl.files.create(file=open(req, 'rb'), purpose='batch').id
            bids.append(cl.batches.create(input_file_id=fid, endpoint='/v1/responses', completion_window='24h').id)
            meta.write_text(json.dumps({'ids': bids, 'n': len(todo), 'submitted': time.strftime('%Y-%m-%dT%H:%M:%S')}))
        print(f'[s7] {len(bids)} batches submitted ({len(todo)} requests): {bids}', flush=True)
    while True:
        bs = [cl.batches.retrieve(i) for i in bids]
        print(time.strftime('%H:%M:%S'), ' | '.join(f'{b.status} {b.request_counts.completed}/{b.request_counts.total}' for b in bs), flush=True)
        if all(b.status in ('completed', 'failed', 'expired', 'cancelled') for b in bs): break
        time.sleep(300)
    saved = failed = 0
    for b in bs:
        if b.output_file_id:
            for l in cl.files.content(b.output_file_id).text.splitlines():
                r = json.loads(l); p, c = r['custom_id'].split('__'); body = (r.get('response') or {}).get('body') or {}
                if (r.get('response') or {}).get('status_code') != 200: failed += 1; continue
                if (od / f'{p}__{c}.json').exists(): continue   # saved (and billed) by an earlier read of this batch
                txt = ''.join(x.get('text', '') for o in body.get('output', []) for x in (o.get('content') or []) if x.get('type') == 'output_text')
                u = body.get('usage') or {}; tin, tout = u.get('input_tokens', 0), u.get('output_tokens', 0)
                spend.add('gpt-6.1-sol', tin, (u.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0, tout, note=f'batch {b.id}')
                rec = {'pair_id': p, 'cid': int(c), 'score': final_score(txt), 'output': txt, 'usage': {'in': tin, 'out': tout}, 'round': rnd, 'tier': 'batch'}
                (od / f'{p}__{c}.json').write_text(json.dumps(rec, ensure_ascii=False)); saved += 1
        if b.error_file_id: failed += len(cl.files.content(b.error_file_id).text.splitlines())
    print(f'[s7] {len(bids)} batches done: saved {saved}, not served {failed}', flush=True)
    return len(todo), saved, failed, time.time() - t0


def main():
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    S6 = W / 's6'; Q = {q['pair_id']: q for q in json.load(open(S6 / 'queries.json'))}
    cand = json.load(open(S6 / 'candidates.json'))['candidates']; P7 = prompt('stage7_umbrela_mod')
    pids = sorted(Q)[:1] if SMOKE else sorted(Q)   # smoke: one query, SMOKE judgments, one round
    if C.get('s7_only'): only = set(json.load(open(C['s7_only']))); pids = [p for p in pids if p in only]
    U = pq.read_table(UNI / 'universe_f.parquet', columns=['corpusid', 'topics'])
    d1 = {c for p in pids for c in Q[p]['d1_cids']}
    Ut = U.filter(pc.is_in(U.column('corpusid'), value_set=pa.array(sorted(d1), pa.int64())))
    d1t = {c: set((x or [])[:3]) for c, x in zip(Ut.column('corpusid').to_pylist(), Ut.column('topics').to_pylist())}
    order = {}
    for p in pids:
        q = Q[p]; qt, pt = set(q['q_topics']), set(q['pos_topics']); order[p] = {s: [] for s in STRATA}; seen = set()
        for c in q['d1_cids']:
            if c not in d1t: continue
            ts = d1t[c]; st = 'q_side' if ts & qt else ('pos_side' if ts & pt else 'other'); order[p][st].append(c); seen.add(c)
        for st in order[p]: order[p][st] += [c for c, m, r in cand[p][st] if c not in seen]
    od = W / ('s7_smoke' if SMOKE else 's7'); od.mkdir(parents=True, exist_ok=True)
    spend = Spend(C['caps']['s7'] - ledger_total(LED, 's7'), LED, 's7')

    def score_of(p, c):
        f = od / f'{p}__{c}.json'
        return json.load(open(f))['score'] if f.exists() else 'todo'
    # passage texts of every candidate, read once from the universe shards and cached
    tc_f = W / 's7_texts.json'; allc = sorted({c for p in pids for lst in order[p].values() for c in lst})
    T = {int(k): v for k, v in json.load(open(tc_f)).items()} if tc_f.exists() else {}
    need = np.array(sorted(set(allc) - set(T)), dtype=np.int64)
    if len(need):
        for f in sorted((UNI / 'export').glob('universe_*.parquet')):
            ids = pq.read_table(f, columns=['corpusid']).column(0).to_numpy(); rows = np.flatnonzero(np.isin(ids, need))
            if not len(rows): continue
            t = pq.read_table(f, columns=['corpusid', 'title', 'abstract']).take(pa.array(rows))
            for c, ti, ab in zip(t.column('corpusid').to_pylist(), t.column('title').to_pylist(), t.column('abstract').to_pylist()):
                T[c] = f"{ti}. {' '.join((ab or '').split())[:2000]}"
        tc_f.write_text(json.dumps(T))
    print(f'[s7] queries {len(pids)} | candidates {len(allc)} | texts {len(T)}', flush=True)
    total = [0, 0, 0, 0.0]; adaptive = C.get('s7_adaptive', False); per_j = 0.0019; NEG = int(C.get('s7_per_stratum', 50)); mode = C.get('s7_mode', 'flex')
    for rnd in range(1, 1000 if adaptive else 8):
        todo = []; stratum_of = {}
        for p in pids:
            for st, lst in order[p].items():
                sc = [score_of(p, c) for c in lst]
                usable = sum(1 for x in sc if x not in ('todo', None) and x <= 1); short = NEG - usable
                if short <= 0: continue
                judged = sum(1 for x in sc if x != 'todo'); y = usable / judged if adaptive and judged >= 20 else 0.68
                add = [(p, c) for c, x in zip(lst, sc) if x == 'todo'][: math.ceil(short / max(y, 0.03) * 1.1)]
                todo += add; stratum_of.update({x: st for x in add})
        if adaptive and todo:
            room = int((spend.cap - spend.usd) / (C.get('s7_per_j_later', per_j) if rnd > 1 else C.get('s7_per_j_r1', per_j)))
            if room < len(todo):   # trim round-robin over (query, group)
                by = collections.defaultdict(list)
                for p, c in todo: by[(p, stratum_of[(p, c)])].append((p, c))
                rr = [x for i in range(max(map(len, by.values()))) for v in by.values() if i < len(v) for x in [v[i]]]
                print(f'[s7] round {rnd} trimmed to {room} of {len(todo)} judgments (remaining cap)', flush=True); todo = rr[:max(room, 0)]
        if not todo: break
        if SMOKE: todo = todo[:SMOKE]
        todo = [(p, c) for p, c in todo if c in T]

        def one(it):
            p, c = it['p'], it['c']
            t, u = openai_call('gpt-6.1-sol', P7.format(query=Q[p]['query'], passage=T[c]), spend, flex=mode not in ('standard', 'batch_first'))
            return {'pair_id': p, 'cid': c, 'score': final_score(t), 'output': t, 'usage': u, 'round': rnd}
        print(f'[s7] round {rnd}: {len(todo)} judgments ({mode})', flush=True)
        if mode == 'batch' or (mode == 'batch_first' and rnd == 1): res = batch_round(rnd, todo, T, P7, Q, od, spend)
        else: res = run_items('s7', [{'id': f'{p}__{c}', 'p': p, 'c': c} for p, c in todo], one, C['workers']['s7'], od)
        total = [a + b for a, b in zip(total, res)]
        if not spend.ok(): print('[s7] spend cap reached', flush=True); break
        if SMOKE: break
    pools, short = {}, []
    for p in pids:
        pools[p] = {'query': Q[p]['query'], 'pos_cid': Q[p]['pos_cid'], 'q_cid': Q[p]['q_cid'], 'main': {}, 'usable': {}}
        for st, lst in order[p].items():
            allok = [c for c in lst if score_of(p, c) not in ('todo', None) and score_of(p, c) <= 1]
            pools[p]['main'][st] = allok[:NEG]; pools[p]['usable'][st] = allok
            if len(allok) < NEG: short.append([p, st, len(allok)])
    (od / C.get('s7_pools_name', '_pools_main.json')).write_text(json.dumps(pools)); (od / C.get('s7_short_name', '_short.json')).write_text(json.dumps(short))
    print(f'[s7] judged {total[1]} (not served {total[2]}) | queries {len(pids)} | short groups {len(short)} | ledger ${spend.usd:.2f} this run', flush=True)


if __name__ == '__main__':
    main()
