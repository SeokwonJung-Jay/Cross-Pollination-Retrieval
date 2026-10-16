#!/usr/bin/env python3
"""Paper universe for pooling and base-paper selection: the papers of top-20% venues and every paper they cite, with an abstract.
  meta        per S2 papers file: (corpusid, year, venueid, doi, mag, pubtypes) for year <= ymax  -> meta/<f>.parquet
  seed        papers with year <= ymax whose publicationvenueid is a top-20% venue of its field (venue_field.py)  -> seed_ids.npy
  refs        every paper cited by a seed (first order, from the compact edges of s2_index.py)  -> refs/<f>.npy
  universe_s2 (seed or refs) with year <= ymax (null year dropped) and an abstract (s2_index.py absids)  -> universe_s2.parquet
  lang        langid.py on the first 1,500 characters of each abstract  -> lang/<f>.npz
  universe    OpenAlex join by DOI, else MAG: language, type, topics (OpenAlex order), field = field of the rank-1 topic  -> universe.parquet
  filter      langid 'en' and (publicationtypes null or containing JournalArticle / Conference / Review)  -> universe_f.parquet
              (corpusid, year, venueid, doi, is_seed, pubtypes, field, topics [short ids 'T123'], alen)
  export      universe_f + title (S2 papers) + whitespace-normalized abstract (S2 abstracts); rows without either are dropped;
              500k-row zstd shards ordered by corpusid  -> export/universe_NNN.parquet (+ manifest.json)
Everything is written under --out (per-input-file parts, atomic tmp -> rename, skipped when present = resumable).
The paper used two universes: --ymax 2025 (all training queries and the 2020-25 eval queries) and --ymax 2019 (the 2010-19 eval
queries). Python with duckdb, langid, pyarrow.
  python -I build_universe.py STEP [STEP ...] --ymax 2025 --out $DATA/universe_2025      (STEP = meta ... export, or all)"""
import argparse, gzip, json, os, sys, time, numpy as np
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env, topic_field_rows
SNAP = env('S2AG'); IDX = env('DATA') / 's2index'; WORKS = str(env('OPENALEX') / 'works_cols' / '*.parquet')
VENUE = env('DATA') / 'venue_if' / 'venue_top20.parquet'
TYPE_RULE = "(pubtypes IS NULL OR pubtypes LIKE '%JournalArticle%' OR pubtypes LIKE '%Conference%' OR pubtypes LIKE '%Review%')"
ap = argparse.ArgumentParser()
ap.add_argument('steps', nargs='+'); ap.add_argument('--ymax', type=int, default=2025, help='latest year kept (seeds and references)')
ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--memory', default='120GB', help='duckdb memory_limit')
ap.add_argument('--force', action='store_true', help='rebuild single-file outputs of the given steps')
A = ap.parse_args()
L = Path(A.out); L.mkdir(parents=True, exist_ok=True)


def log(*x): print(time.strftime('%H:%M:%S'), *x, flush=True)
def files(p, pat='*.gz'): return sorted(Path(p).glob(pat))
def has(arr, x):
    i = np.searchsorted(arr, x); return (i < len(arr)) & (arr[np.minimum(i, len(arr) - 1)] == x)
def done(p): return Path(p).exists() and not A.force
def atomic_np(o, arr):
    t = Path(str(o) + '.tmp.npy'); np.save(t, arr); t.replace(o)
def duck():
    import duckdb
    c = duckdb.connect(); c.execute(f"SET memory_limit='{A.memory}'"); c.execute(f"SET temp_directory='{L / '_duck_tmp'}'"); return c
def meta_src(): return sorted(str(f) for f in (L / 'meta').glob('*.parquet'))


def meta_one(f):
    import pyarrow as pa, pyarrow.parquet as pq
    o = L / 'meta' / (f.name + '.parquet')
    if o.exists(): return f.name, 'skip'
    c, y, v, d, m, t = [], [], [], [], [], []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            r = json.loads(l); yr = r.get('year')
            if not yr or yr > A.ymax: continue
            e = r.get('externalids') or {}
            c.append(r['corpusid']); y.append(yr); v.append(r.get('publicationvenueid')); d.append((e.get('DOI') or '').lower() or None); m.append(e.get('MAG'))
            t.append('|'.join(r.get('publicationtypes') or []) or None)
    tmp = o.with_suffix('.tmp')
    pq.write_table(pa.table({'corpusid': pa.array(c, pa.int64()), 'year': pa.array(y, pa.int16()), 'venueid': pa.array(v, pa.string()), 'doi': pa.array(d, pa.string()),
                             'mag': pa.array(m, pa.string()), 'pubtypes': pa.array(t, pa.string())}), tmp); tmp.replace(o); return f.name, len(c)


def step_meta():
    (L / 'meta').mkdir(exist_ok=True)
    with Pool(A.workers) as p:
        for x in p.imap_unordered(meta_one, files(SNAP / 'papers')): log('meta', *x)


def step_seed():
    o = L / 'seed_ids.npy'
    if done(o): log('seed: exists', o); return
    ids = duck().execute(f"""SELECT DISTINCT m.corpusid FROM read_parquet({meta_src()}) m JOIN read_parquet('{VENUE}') t ON m.venueid = t.venueid
                             WHERE t.top20 AND m.year <= {A.ymax} ORDER BY 1""").fetchnumpy()['corpusid'].astype(np.int64)
    atomic_np(o, ids); log('seed papers', len(ids))


_R = {}
def refs_one(f):
    o = L / 'refs' / f.name
    if o.exists(): return f.name, 'skip'
    if not _R: _R['s'] = np.load(L / 'seed_ids.npy')
    e = np.load(f); m = has(_R['s'], e[:, 0].astype(np.int64)); atomic_np(o, np.unique(e[m, 1].astype(np.int64))); return f.name, int(m.sum())


def step_refs():
    (L / 'refs').mkdir(exist_ok=True)
    with Pool(A.workers) as p:
        for x in p.imap_unordered(refs_one, files(IDX / 'edges', '*.npy')): log('refs', *x)


def step_universe_s2():
    o = L / 'universe_s2.parquet'
    if done(o): log('universe_s2: exists'); return
    import pyarrow as pa
    seed = np.load(L / 'seed_ids.npy'); refs = np.unique(np.concatenate([np.load(f) for f in (L / 'refs').glob('*.npy') if not f.name.endswith('.tmp.npy')]))
    ab = np.unique(np.concatenate([np.load(f) for f in (IDX / 'abs_ids').glob('*.npy')]))
    keep = np.union1d(seed, refs); keep = keep[has(ab, keep)]
    c = duck(); c.register('keep_a', pa.table({'corpusid': keep, 'is_seed': has(seed, keep)})); c.execute("CREATE TABLE keep_t AS SELECT * FROM keep_a")
    c.execute(f"""COPY (SELECT m.*, k.is_seed FROM read_parquet({meta_src()}) m JOIN keep_t k USING (corpusid)) TO '{o}.tmp' (FORMAT parquet)"""); os.replace(f'{o}.tmp', o)
    log('universe_s2', c.execute(f"SELECT count(*), sum(is_seed::INT) FROM read_parquet('{o}')").fetchall(), '| refs', len(refs), '| seeds', len(seed))


_U = {}
def lang_one(f):
    import langid
    o = L / 'lang' / (f.name + '.npz')
    if o.exists(): return f.name, 'skip'
    if not _U: _U['ids'] = set(np.load(L / 'lang_todo.npy').tolist())
    ids = _U['ids']; idn = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True); c, lg, pr, ln = [], [], [], []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"corpusid":'); j = l.find(',', k)
            try: cid = int(l[k + 11:j])
            except ValueError: continue
            if cid not in ids: continue
            a = json.loads(l).get('abstract') or ''
            g, q = idn.classify(a[:1500]); c.append(cid); lg.append(g); pr.append(q); ln.append(len(a))
    t = Path(str(o) + '.tmp.npz'); np.savez(t, cid=np.array(c, np.int64), lang=np.array(lg), prob=np.array(pr, np.float32), alen=np.array(ln, np.int32)); t.replace(o)
    return f.name, len(c)


def step_lang():
    (L / 'lang').mkdir(exist_ok=True); tp = L / 'lang_todo.npy'
    if not tp.exists():
        import duckdb
        atomic_np(tp, np.sort(duckdb.connect().execute(f"SELECT corpusid FROM read_parquet('{L}/universe_s2.parquet')").fetchnumpy()['corpusid'].astype(np.int64)))
    with Pool(A.workers) as p:
        for x in p.imap_unordered(lang_one, files(SNAP / 'abstracts')): log('lang', *x)


def step_universe():
    o = L / 'universe.parquet'
    if done(o): log('universe: exists'); return
    import pyarrow as pa
    c = duck(); t2f = topic_field_rows(); c.register('t2f_a', pa.table({'tid': [a for a, _ in t2f], 'field': [b for _, b in t2f]})); c.execute("CREATE TABLE t2f AS SELECT * FROM t2f_a")
    c.execute(f"CREATE TABLE u AS SELECT * FROM read_parquet('{L}/universe_s2.parquet')")
    log('universe before OpenAlex join', c.execute("SELECT count(*), sum(is_seed::INT) FROM u").fetchall())
    # only the OpenAlex rows whose DOI or MAG id occurs in the universe (semi-join while scanning)
    c.execute(f"""CREATE TABLE oa_doi AS SELECT lower(replace(w.doi, 'https://doi.org/', '')) AS doi, w.language AS oa_language, w.type AS oa_type, w.topics AS topics
                  FROM read_parquet('{WORKS}') w WHERE w.doi IS NOT NULL AND lower(replace(w.doi, 'https://doi.org/', '')) IN (SELECT doi FROM u WHERE doi IS NOT NULL)""")
    c.execute(f"""CREATE TABLE oa_mag AS SELECT CAST(w.mag AS VARCHAR) AS mag, w.language AS oa_language, w.type AS oa_type, w.topics AS topics
                  FROM read_parquet('{WORKS}') w WHERE w.mag IS NOT NULL AND CAST(w.mag AS VARCHAR) IN (SELECT mag FROM u WHERE mag IS NOT NULL)""")
    log('OpenAlex rows matched: doi', c.execute("SELECT count(*) FROM oa_doi").fetchone(), 'mag', c.execute("SELECT count(*) FROM oa_mag").fetchone())
    c.execute(f"""COPY (
        SELECT u.*, coalesce(a.oa_language, b.oa_language) AS oa_language, coalesce(a.oa_type, b.oa_type) AS oa_type, coalesce(a.topics, b.topics) AS topics, t2f.field AS field
          FROM u LEFT JOIN (SELECT DISTINCT ON (doi) * FROM oa_doi ORDER BY doi, topics) a ON u.doi = a.doi
          LEFT JOIN (SELECT DISTINCT ON (mag) * FROM oa_mag ORDER BY mag, topics) b ON a.doi IS NULL AND u.mag = b.mag
          LEFT JOIN t2f ON t2f.tid = coalesce(a.topics, b.topics)[1]) TO '{o}.tmp' (FORMAT parquet)"""); os.replace(f'{o}.tmp', o)
    log('universe', c.execute(f"SELECT count(*), sum(is_seed::INT), count(topics), count(field), count(*) FILTER (WHERE oa_language='en') FROM read_parquet('{o}')").fetchall())


def step_filter():
    o = L / 'universe_f.parquet'
    if done(o): log('universe_f: exists'); return
    import pyarrow as pa
    z = [np.load(f) for f in sorted((L / 'lang').glob('*.npz')) if not f.name.endswith('.tmp.npz')]
    c = duck(); c.register('lt0', pa.table({'corpusid': np.concatenate([x['cid'] for x in z]), 'lang': np.concatenate([x['lang'].astype(str) for x in z]),
                                           'alen': np.concatenate([x['alen'] for x in z])}))
    c.execute("CREATE TABLE lt AS SELECT corpusid, any_value(lang) lang, any_value(alen) alen FROM lt0 GROUP BY 1")
    c.execute(f"""COPY (SELECT u.corpusid, u.year, u.venueid, u.doi, u.is_seed, u.pubtypes, u.field,
                  list_transform(u.topics, x -> replace(x, 'https://openalex.org/', '')) AS topics, l.alen
                  FROM read_parquet('{L}/universe.parquet') u JOIN lt l USING (corpusid)
                  WHERE l.lang = 'en' AND {TYPE_RULE.replace('pubtypes', 'u.pubtypes')}) TO '{o}.tmp' (FORMAT parquet)"""); os.replace(f'{o}.tmp', o)
    log('filtered universe', c.execute(f"SELECT count(*), sum(is_seed::INT), count(field) FROM read_parquet('{o}')").fetchall())


_X = {}
def text_one(a):
    import pyarrow as pa, pyarrow.parquet as pq
    f, kind = a; o = L / 'export' / '_parts' / f'{kind}_{f.name}.parquet'
    if o.exists(): return o.name, 'skip'
    if not _X: _X['ids'] = set(np.load(L / 'export_ids.npy').tolist())
    ids = _X['ids']; cs, ts = [], []
    with gzip.open(f, 'rt') as fi:
        for l in fi:
            k = l.find('"corpusid":'); j = l.find(',', k)
            try: c = int(l[k + 11:j])
            except ValueError: continue
            if c in ids:
                r = json.loads(l); cs.append(c); ts.append((r.get('title') or '') if kind == 'title' else ' '.join((r.get('abstract') or '').split()))
    t = o.with_suffix('.tmp'); pq.write_table(pa.table({'corpusid': pa.array(cs, pa.int64()), 'text': pa.array(ts, pa.string())}), t, compression='zstd'); t.replace(o)
    return o.name, len(cs)


def step_export():
    import duckdb, pyarrow as pa, pyarrow.parquet as pq
    E = L / 'export'; (E / '_parts').mkdir(parents=True, exist_ok=True); U = L / 'universe_f.parquet'
    if (E / 'manifest.json').exists() and not A.force: log('export: exists'); return
    if not (L / 'export_ids.npy').exists():
        atomic_np(L / 'export_ids.npy', np.sort(duckdb.connect().execute(f"SELECT corpusid FROM read_parquet('{U}')").fetchnumpy()['corpusid'].astype(np.int64)))
    with Pool(A.workers) as p:
        for x in p.imap_unordered(text_one, [(f, 'title') for f in files(SNAP / 'papers')] + [(f, 'abstract') for f in files(SNAP / 'abstracts')]): log('export', *x)
    c = duck(); P = E / '_parts'
    q = f"""SELECT u.corpusid, u.year, u.field, u.is_seed, u.topics, t.tx AS title, a.tx AS abstract FROM read_parquet('{U}') u
            JOIN (SELECT corpusid, any_value("text") tx FROM read_parquet('{P}/title_*.parquet') GROUP BY 1) t USING (corpusid)
            JOIN (SELECT corpusid, any_value("text") tx FROM read_parquet('{P}/abstract_*.parquet') GROUP BY 1) a USING (corpusid)
            WHERE t.tx <> '' AND a.tx <> '' ORDER BY u.corpusid"""
    for f in E.glob('universe_*.parquet'): f.unlink()
    rd = c.execute(q).fetch_record_batch(100_000); buf, nbuf, n, rows, B = [], 0, 0, 0, 500_000

    def write(t):
        nonlocal n, rows
        pq.write_table(t, E / f'universe_{n:03d}.parquet.tmp', compression='zstd'); os.replace(E / f'universe_{n:03d}.parquet.tmp', E / f'universe_{n:03d}.parquet'); n += 1; rows += t.num_rows
    for b in rd:
        buf.append(b); nbuf += b.num_rows
        while nbuf >= B:
            t = pa.Table.from_batches(buf).combine_chunks(); write(t.slice(0, B)); rest = t.slice(B); buf = rest.to_batches(); nbuf = rest.num_rows
    if nbuf: write(pa.Table.from_batches(buf))
    total = c.execute(f"SELECT count(*) FROM read_parquet('{U}')").fetchone()[0]
    json.dump({'rows': rows, 'shards': n, 'dropped_no_text': total - rows, 'ymax': A.ymax}, open(E / 'manifest.json', 'w'), indent=1)
    log(f'export rows {rows:,} shards {n} dropped {total - rows:,}')


STEPS = ['meta', 'seed', 'refs', 'universe_s2', 'lang', 'universe', 'filter', 'export']
if __name__ == '__main__':
    kp = L / 'params.json'
    if kp.exists() and json.load(open(kp))['ymax'] != A.ymax: sys.exit(f'{L} was built with ymax {json.load(open(kp))["ymax"]}; use another --out')
    json.dump({'ymax': A.ymax}, open(kp, 'w'))
    for s in (STEPS if A.steps == ['all'] else A.steps):
        t0 = time.time(); log('== step', s); globals()['step_' + s](); log(f'== step {s} done {time.time() - t0:.0f}s')
