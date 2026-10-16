#!/usr/bin/env python3
"""Venue field + top-20% venues per field.
Field of a Semantic Scholar venue = majority OpenAlex field of the rank-1 topic over up to 200 of its 2015-2025 papers (random sample,
seed 0) joined to OpenAlex works by DOI. Eligible venue: type journal or conference and at least MIN_N (default 50) papers in
2023-2024. Top 20% = percent rank of the 2025 2-year IF (venue_if.py) >= 0.8 within the field.
Writes $DATA/venue_if/venue_top20.parquet (venue_if columns + field, field_share, n_matched, pr, top20) and prints per-field thresholds.
  python -I venue_field.py [MIN_N]"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import env, topic_field_rows
import duckdb, pyarrow as pa
OUT = env('DATA') / 'venue_if'; WORKS = env('OPENALEX') / 'works_cols' / '*.parquet'; MIN_N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
t2f = dict(topic_field_rows())
by = {}
for f in sorted((OUT / 'venue_papers').glob('*.tsv')):
    for l in open(f):
        x = l.rstrip('\n').split('\t')
        if len(x) != 3: continue   # DOI strings with embedded newlines/tabs
        by.setdefault(x[0], []).append(x[2].lower())
rng = random.Random(0); rows = [(v, 'https://doi.org/' + d) for v, ds in by.items() for d in (rng.sample(ds, 200) if len(ds) > 200 else ds)]
c = duckdb.connect(); c.register("s_arrow", pa.table({"venueid": [r[0] for r in rows], "doi": [r[1] for r in rows]})); c.execute("CREATE TABLE s AS SELECT * FROM s_arrow")
print("sample rows", len(rows), flush=True)
c.register("t2f_arrow", pa.table({"tid": list(t2f), "field": list(t2f.values())})); c.execute("CREATE TABLE t2f AS SELECT * FROM t2f_arrow")
c.execute(f"""CREATE TABLE vf AS SELECT venueid, arg_max(field, n) AS field, max(n) * 1.0 / sum(n) AS field_share, sum(n) AS n_matched FROM (
  SELECT s.venueid, t2f.field, count(*) n FROM s JOIN (SELECT lower(doi) doi, topics[1] tid FROM read_parquet('{WORKS}') WHERE doi IS NOT NULL) w USING (doi)
  JOIN t2f USING (tid) GROUP BY 1, 2) GROUP BY 1""")
c.execute(f"""CREATE TABLE el AS SELECT v.*, vf.field, vf.field_share, vf.n_matched, percent_rank() OVER (PARTITION BY vf.field ORDER BY if_2025) pr
  FROM read_parquet('{OUT}/venue_if.parquet') v JOIN vf USING (venueid) WHERE v.type IN ('journal', 'conference') AND n_2023_24 >= {MIN_N}""")
c.execute(f"COPY (SELECT *, pr >= 0.8 AS top20 FROM el) TO '{OUT}/venue_top20.parquet' (FORMAT parquet)")
print(f"{'field':45} {'venues':>6} {'top20':>5} {'IF threshold':>12} {'median IF':>9} {'conf in top20':>13}")
for r in c.execute("SELECT field, count(*), count(*) FILTER (WHERE pr>=0.8), min(if_2025) FILTER (WHERE pr>=0.8), median(if_2025), count(*) FILTER (WHERE pr>=0.8 AND type='conference') FROM el GROUP BY 1 ORDER BY 1").fetchall():
    print(f"{r[0]:45} {r[1]:6} {r[2]:5} {r[3]:12.2f} {r[4]:9.2f} {r[5]:13}")
