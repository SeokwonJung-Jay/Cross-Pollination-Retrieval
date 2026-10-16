#!/usr/bin/env python3
"""Held-out domain folds: for each OpenAlex domain D (Life / Social / Physical / Health Sciences) a training set without D and the eval
queries of D.
  - Domains (build_domains.py): query = field of the pair, positive = OpenAlex primary topic (top-1). Hard negatives are not looked at.
  - Training pairs without a positive domain are dropped from every fold (listed in OUT/dropped_no_domain.jsonl).
  - Fold D train = pairs whose query domain != D and positive domain != D, then every fold is sampled down to the smallest fold size
    with random.Random(seed).sample (rows written in their original order).
  - Fold D eval = CPR-eval queries whose query domain is D (selected by query domain only, so queries whose positive has no
    OpenAlex domain are kept); every query keeps its full pool of 151 candidates.
  - Checks (assert): no training query / positive domain == D; no query / positive corpus id of a fold's eval queries among the fold's
    training queries / positives; N distinct rows per fold.
--data is the training file whose rows are written (rows matched by query_id): train_texts.jsonl of the release, or the copy cut to 1023 tokens + EOS
for Qwen3-Embedding (ablation/dataset: DATA_ml1024/hardneg/ours/train.jsonl). Both give the same rows per fold.
Writes OUT/fold_<slug>/{train.jsonl, train_ids.txt, eval_ids.txt}, OUT/folds.json (sizes, removal counts, domain counts, checks),
OUT/dropped_no_domain.jsonl.
  python -I build_folds.py --data train.jsonl --pairs PAIRS/pairs_train.jsonl --pairs-test PAIRS/pairs_test.jsonl \
    --pools WORK/_data/pools.json --out FOLDS [--seed 42]"""
import argparse, collections, json, os, random
ap = argparse.ArgumentParser()
for x in ('--data', '--pairs', '--pairs-test', '--pools', '--out'): ap.add_argument(x, required=True)
ap.add_argument('--seed', type=int, default=42); A = ap.parse_args()
SLUG = {'Life Sciences': 'life', 'Social Sciences': 'social', 'Physical Sciences': 'physical', 'Health Sciences': 'health'}
rows = [json.loads(l) for l in open(A.data)]
dom = {x['query_id']: x for x in map(json.loads, open(A.pairs))}
tdom = {x['query_id']: x for x in map(json.loads, open(A.pairs_test))}
pools = json.load(open(A.pools))
assert len(rows) == len(dom) and all(r['query_id'] in dom for r in rows)
assert all(q in tdom for q in pools) and all(tdom[q]['q_domain'] for q in pools)
ok = lambda d: bool(d['q_domain']) and bool(d['pos_domain'])
kept = [r for r in rows if ok(dom[r['query_id']])]
elig = {D: [r for r in kept if D not in (dom[r['query_id']]['q_domain'], dom[r['query_id']]['pos_domain'])] for D in SLUG}
N = min(len(v) for v in elig.values())
os.makedirs(A.out, exist_ok=True)
info = {'seed': A.seed, 'train_total': len(rows), 'train_no_domain_dropped': len(rows) - len(kept), 'train_size_per_fold': N, 'folds': {}}
with open(f'{A.out}/dropped_no_domain.jsonl', 'w') as f:
    for r in rows:
        d = dom[r['query_id']]
        if not ok(d): f.write(json.dumps({'split': 'train', 'query_id': r['query_id'], 'q_cid': d['q_cid'], 'pos_cid': d['pos_cid'], 'q_domain': d['q_domain'],
                                          'pos_openalex': d['pos_openalex'], 'action': 'dropped from all folds'}) + '\n')
    for q in pools:
        d = tdom[q]
        if not d['pos_domain']: f.write(json.dumps({'split': 'test', 'query_id': q, 'q_cid': d['q_cid'], 'pos_cid': d['pos_cid'], 'q_domain': d['q_domain'],
                                                    'pos_openalex': d['pos_openalex'], 'action': 'kept in the fold eval (queries are selected by query domain)'}) + '\n')
for D, s in SLUG.items():
    idx = sorted(random.Random(A.seed).sample(range(len(elig[D])), N)); tr = [elig[D][i] for i in idx]
    ev = [q for q in pools if tdom[q]['q_domain'] == D]
    # leakage checks
    assert all(dom[r['query_id']]['q_domain'] != D and dom[r['query_id']]['pos_domain'] != D for r in tr), D
    trc = {dom[r['query_id']]['q_cid'] for r in tr} | {dom[r['query_id']]['pos_cid'] for r in tr}
    evc = {tdom[q]['q_cid'] for q in ev} | {tdom[q]['pos_cid'] for q in ev}
    assert not (trc & evc), (D, len(trc & evc))
    assert len(tr) == N == len({r['query_id'] for r in tr})
    o = f'{A.out}/fold_{s}'; os.makedirs(o, exist_ok=True)
    with open(f'{o}/train.jsonl', 'w') as f: f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in tr)
    open(f'{o}/train_ids.txt', 'w').write('\n'.join(r['query_id'] for r in tr) + '\n')
    open(f'{o}/eval_ids.txt', 'w').write('\n'.join(ev) + '\n')
    info['folds'][s] = {'eval_domain': D, 'train_before_leak_removal': len(kept), 'removed_q_in_D': sum(dom[r['query_id']]['q_domain'] == D for r in kept),
        'removed_pos_only_in_D': sum(dom[r['query_id']]['q_domain'] != D and dom[r['query_id']]['pos_domain'] == D for r in kept),
        'train_after_leak_removal': len(elig[D]), 'train_final': N, 'train_q_domains': dict(collections.Counter(dom[r['query_id']]['q_domain'] for r in tr)),
        'train_pos_domains': dict(collections.Counter(dom[r['query_id']]['pos_domain'] for r in tr)), 'eval_pairs': len(ev),
        'eval_no_pos_domain': [q for q in ev if not tdom[q]['pos_domain']],
        'leak_check': 'pass: no train q/pos domain == eval domain; no shared q/pos corpus id with eval pairs'}
    print(s, json.dumps(info['folds'][s], ensure_ascii=False))
json.dump(info, open(f'{A.out}/folds.json', 'w'), indent=1, ensure_ascii=False); print('train per fold', N)
