#!/usr/bin/env python3
"""Domain of the query and of the positive of every pair (train + eval), one OpenAlex domain per paper (top-1).
  query domain    = the pair's `field` (an OpenAlex field) -> domain (evaluation/main/field_domain.json)
  positive domain = domain of the OpenAlex primary topic of pos_cid (fetch_openalex.py), null if the paper has none
  pos_domains_top3 = domains of the primary + top-3 topics (recorded, not used by build_folds.py)
Hard negatives are not looked at. Writes OUT/pairs_{train,test}.jsonl (no texts) and prints the counts per domain.
  python -I build_domains.py --pairs PAIRS --papers META/paper_domains.jsonl --out PAIRS"""
import argparse, collections, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--pairs', required=True); ap.add_argument('--papers', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--field-domain', default=os.path.join(HERE, '..', '..', 'evaluation', 'main', 'field_domain.json')); A = ap.parse_args()
FD = json.load(open(A.field_domain)); F2D = {f: d for d, fs in FD.items() if not d.startswith('_') for f in fs}
M = {x['cid']: x for x in map(json.loads, open(A.papers))}
os.makedirs(A.out, exist_ok=True)
for sp in ('train', 'test'):
    rows = []
    for l in open(f'{A.pairs}/{sp}.jsonl'):
        r = json.loads(l); p = M[str(r['pos_cid'])]; pr = p['primary'] or {}
        rows.append({'query_id': r['query_id'], 'q_cid': str(r['q_cid']), 'pos_cid': str(r['pos_cid']), 'q_field': r['field'], 'q_domain': F2D[r['field']],
                     'pos_field': pr.get('field'), 'pos_domain': pr.get('domain'), 'pos_subfield': pr.get('subfield'),
                     'pos_domains_top3': sorted({pr['domain']} | {t['domain'] for t in p['topics']}) if pr else [], 'pos_openalex': p['openalex']})
    with open(f'{A.out}/pairs_{sp}.jsonl', 'w') as f: f.writelines(json.dumps(x) + '\n' for x in rows)
    print(sp, len(rows), '| q_domain', dict(collections.Counter(x['q_domain'] for x in rows)), '| pos_domain', dict(collections.Counter(x['pos_domain'] for x in rows)))
