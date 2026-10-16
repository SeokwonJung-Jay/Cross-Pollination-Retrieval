#!/usr/bin/env python3
"""OpenAlex inputs (public data, no API key).
  works  : column subset of every works parquet file of the OpenAlex snapshot on S3 (s3://openalex, read over HTTPS with DuckDB httpfs):
           id, doi, mag, publication_year, language, type, primary source id, topic ids (OpenAlex order, rank 1 first),
           referenced_works, cited_by_count -> $OPENALEX/works_cols/<partition>_<part>.parquet (resumable; .done markers)
  topics : the OpenAlex topic catalog (about 4,500 topics, each with its subfield, field and domain) from the OpenAlex API
           -> $OPENALEX/topics_catalog.json  [{id, display_name, subfield{id, display_name}, field{...}, domain{...}}]
The paper used the snapshot of 2026-09-23. Remote column reads move a small share of the file bytes.
  python -I openalex_extract.py works [THREADS]
  python -I openalex_extract.py topics"""
import json, os, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env
B = 'https://openalex.s3.amazonaws.com/'; OUT = env('OPENALEX')


def keys(prefix):
    tok = None; out = []
    while True:
        u = f'{B}?list-type=2&prefix={prefix}&max-keys=1000' + (f'&continuation-token={urllib.parse.quote(tok)}' if tok else '')
        s = urllib.request.urlopen(u).read().decode(); out += [k for k in re.findall(r'<Key>([^<]*)</Key>', s) if k.endswith('.parquet')]
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', s)
        if not m: return out
        tok = m.group(1)


COLS = ("id, doi, ids.mag AS mag, publication_year, language, type, primary_location.source.id AS src, "
        "list_transform(topics, x -> x.id) AS topics, referenced_works, cited_by_count")


def one(k):
    import duckdb
    o = OUT / 'works_cols' / (k.split('works/')[1].replace('/', '_').replace('updated_date=', '')); done = Path(str(o) + '.done')
    if done.exists(): return k, 'skip'
    for at in range(5):
        try:
            c = duckdb.connect(); c.execute("INSTALL httpfs; LOAD httpfs;")
            c.execute(f"COPY (SELECT {COLS} FROM read_parquet('{B}{k}')) TO '{o}.tmp' (FORMAT parquet)"); os.rename(f'{o}.tmp', o); done.touch(); return k, 'ok'
        except Exception as e: err = e; time.sleep(10 * (at + 1))
    return k, f'FAIL {err}'


def topics():
    out, cursor = [], '*'
    while cursor:
        u = 'https://api.openalex.org/topics?' + urllib.parse.urlencode({'per-page': 200, 'cursor': cursor, 'select': 'id,display_name,subfield,field,domain'})
        r = json.loads(urllib.request.urlopen(u).read().decode()); out += r['results']; cursor = r['meta'].get('next_cursor')
        time.sleep(0.2)
    json.dump(out, open(OUT / 'topics_catalog.json', 'w')); print('topics', len(out), flush=True)


if __name__ == '__main__':
    (OUT / 'works_cols').mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == 'topics': topics()
    elif sys.argv[1] == 'works':
        ks = keys('data/parquet/works/'); print('works files', len(ks), flush=True); n = 0
        with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as ex:
            for k, st in ex.map(one, ks):
                n += 1
                if st != 'skip' and (n % 50 == 0 or st.startswith('FAIL')): print(n, k, st, flush=True)
        print('works done', flush=True)
    else: sys.exit(__doc__)
