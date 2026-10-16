# CPR-Dataset construction

Code that builds CPR-eval and CPR-train, in the order of the paper's Section 3. Corpora, vectors and intermediate files are not
shipped; every step rebuilds them from the public sources.

## Setup

```bash
pip install -r requirements.txt          # Python 3.11
```

Environment (or a `.env` file in the repo root / working directory):

| variable | content |
|---|---|
| `S2AG` | Semantic Scholar Academic Graph release (`papers/`, `abstracts/`, `citations/`, `publication-venues/`, `*.gz`); paper: 2026-09-09 |
| `S2ORC` | Semantic Scholar `s2orc_v2` release (`*.gz`); paper: 2026-09-22 |
| `OPENALEX` | output dir of `corpus/openalex_extract.py` (OpenAlex snapshot on S3; paper: 2026-09-23) |
| `DATA` | derived data (venue table, indexes, universes, vectors, Stage 2 caches) |
| `WORK` | run outputs (`$WORK/eval`, `$WORK/train`, exported JSONL files) |
| `OPENAI_API_KEY` | Stages 3-7 |
| `S2_API_KEY` | optional, Semantic Scholar Graph API (openAccessPdf, external ids) |

Network sources used without a key: OpenAlex S3 snapshot and API (topic catalog), Europe PMC REST, PMC Open Access on AWS
(`pmc-oa-opendata`), open-access PDFs.

## Layout

```
common.py          config loading (${ENV} in config strings), OpenAI call / spend cap / ledger helpers
config/            eval.json, train.json: all run settings
prompts/           the LLM prompts (verbatim; the appendix prompts)
corpus/            openalex_extract, venue_if, venue_field, s2_index, build_universe, sample_base_papers
positives/         stage2_input, openalex_tags, pmcid_lookup, jats_prefetch, s2orc_extract, stage2_parallel, stage2/ (Stage 2 code),
                   stage3_analogy, stage4_transfer, stage5_query, dedup_usefulness (5b, 7b)
negatives/         encode_corpus, pool_universe, judge_negatives, retrievers.py (pooling models, pinned revisions)
export/            select_eval, export_eval, export_train
release/           license_table, collect_ids, fetch_externalids, build_release, fetch_abstracts
run.sh             one run: prep -> stage2 -> s3 -> s4 -> s5 -> s5b -> s7b -> select -> pool -> s7 -> export
```

All scripts run as `python -I <script> ...`; each docstring gives its inputs, outputs and options.

## 1. Corpus (once)

```bash
python -I corpus/openalex_extract.py works 8          # $OPENALEX/works_cols/*.parquet
python -I corpus/openalex_extract.py topics           # $OPENALEX/topics_catalog.json
python -I corpus/venue_if.py papers && python -I corpus/venue_if.py lookup && python -I corpus/venue_if.py citations && python -I corpus/venue_if.py combine
python -I corpus/venue_field.py 50                    # $DATA/venue_if/venue_top20.parquet (top 20% per field, venues with >= 50 papers in 2023-24)
for c in absids edges refcount s2orc; do python -I corpus/s2_index.py $c; done      # $DATA/s2index/
python -I corpus/build_universe.py all --ymax 2025 --out $DATA/universe_2025        # universe_f.parquet, export/universe_*.parquet
python -I corpus/build_universe.py all --ymax 2019 --out $DATA/universe_2019        # for 2010-19 base papers (see Notes)
python -I corpus/sample_base_papers.py --universe $DATA/universe_2019 --ymin 2010 --ymax 2019 --per-field 800 --seed 20261005 \
    --out $DATA/base_papers/pool_2010_2019.json
python -I corpus/sample_base_papers.py --universe $DATA/universe_2025 --ymin 2010 --ymax 2025 --per-field 1600 --seed 20261006 \
    --exclude $DATA/base_papers/pool_2010_2019.json --out $DATA/base_papers/pool_2010_2025.json
# pooling vectors, one process per GPU (--part i/n); then the same for splade and scincl
python -I negatives/encode_corpus.py --corpus $DATA/universe_2025 --vectors $DATA/vectors_2025 --model e5 --device cuda:0 --part 0/1
```

## 2. Runs

```bash
bash run.sh config/eval.json             # -> $WORK/eval.jsonl   (run first: the training run reads its exclusion lists)
bash run.sh config/train.json            # -> $WORK/train.jsonl + train.meta.jsonl
bash run.sh config/eval.json s7 export   # any subset of steps
```

| step | script | output in `<work>` | model / settings (config) |
|---|---|---|---|
| prep | `positives/stage2_input.py`, `openalex_tags.py`, `pmcid_lookup.py`, `jats_prefetch.py`, `s2orc_extract.py` | `stage2_input/`; caches in `stage2_cache` | base papers: `q_pool`, `q_per_field`, `exclude_base_papers` |
| stage2 | `positives/stage2_parallel.py` plan / run / merge (`stage2/stage2_candidate_preparation.py` per shard) | `stage2/prepared_pairs.json` | `workers.stage2_shards`; pair id `<pair_prefix>_<q>_<d>`; full text JATS -> s2orc_v2 -> PDF, at most 6 citing sentences |
| s3 | `positives/stage3_analogy.py` | `s3/<pair>.json` | gpt-5.6-sol, reasoning none, Chat Completions, flex, prompt cache; pass = observed_analogy / potential_analogy; `exclude_positives` |
| s4 | `positives/stage4_transfer.py` | `s4/<pair>.json` | gpt-6.1-sol, reasoning low, `s4_tier`; pass = verdict != reject |
| s5 | `positives/stage5_query.py` | `s5/stage5_v1a/`, `s5/stage5_review.json` | gpt-6-astra, default reasoning, standard tier; review gate, up to 2 regenerations |
| s5b | `positives/dedup_usefulness.py s5b` | `s5b/q_<q>.json` | gpt-6-astra |
| s7b | `positives/dedup_usefulness.py s7b` | `s7b/<pair>.json`, `final_pairs.json` | gpt-6-astra; keep u >= 2 |
| select | `export/select_eval.py` (eval only) | `eval_selection.json`, `eval_pairs.json` | `queries_per_field` (50) + spares |
| pool | `negatives/pool_universe.py` spec / score x3 / join / merge | `s6/queries.json`, `s6/top300_<m>.json`, `s6/candidates.json` | E5 / SPLADE-v3 / SciNCL, top `POOL_K`=300 per retriever and group; `corpus`, `vectors` |
| s7 | `negatives/judge_negatives.py` | `s7/<pair>__<cid>.json`, `s7/_pools_main.json` | gpt-6.1-sol, `s7_mode` batch; keep u <= 1; `s7_per_stratum` 50 (eval) / 3 (train) |
| export | `export/export_eval.py`, `export/export_train.py --neg 9 --exclude-eval` | `$WORK/eval.jsonl`, `$WORK/train.jsonl` | eval text = title + '. ' + abstract[:2000]; train text = full abstract |

Every LLM step resumes from its saved records and stops starting calls at its `caps.<stage>` (USD, cumulative over
`<work>/cost_ledger.jsonl`).

Output formats:
- eval: `{query_id, query, field, q_year, source, pos: {cid, text}, negs: [{cid, stratum, text}]}` (the fields of `eval_texts.jsonl`,
  the `evaluation/main` input, plus `q_year` and `source`)
- train: `{query_id, query, pos: [text], neg: [text x 9]}` (the `training/` input format), plus `.meta.jsonl` with corpus ids

## 3. Release (id-only)

```bash
python -I release/license_table.py                                    # $DATA/license/abstracts/
python -I release/collect_ids.py $WORK/eval.jsonl $WORK/train.meta.jsonl OUT
python -I release/fetch_externalids.py OUT/release_ids.json OUT/release_externalids.jsonl
python -I release/build_release.py --eval $WORK/eval.jsonl --train $WORK/train.jsonl --judgments $WORK/eval/s7 $WORK/train/s7 \
    --corpus $DATA/universe_2025 $DATA/universe_2019 --out OUT
```

`OUT/` holds `papers.jsonl`, `eval.jsonl`, `train.jsonl` and `fetch_abstracts.py`. The release carries ids, queries, labels and the
abstracts whose license allows redistribution (CC-BY, CC0 / public domain); every other abstract is identified by its Semantic Scholar
corpus id and the SHA-256 of its text. It is published as one dataset, CPR-Dataset, with the splits `eval/` (CPR-eval: `eval.jsonl`)
and `train/` (CPR-train: `train.jsonl`), each with the `papers.jsonl` rows of its own papers, plus `fetch_abstracts.py`.

### Rebuilding the texts

`release/fetch_abstracts.py` fills in the missing abstracts and writes the text files every other part of this repository reads:

```bash
huggingface-cli download CPR-Dataset/CPR-Dataset --repo-type dataset --local-dir $RELEASE_DIR
EVAL_DIR=$RELEASE_DIR/eval; TRAIN_DIR=$RELEASE_DIR/train
python3 release/fetch_abstracts.py --release $EVAL_DIR  --snapshot $S2AG/abstracts    # or --api (Semantic Scholar Graph API)
python3 release/fetch_abstracts.py --release $TRAIN_DIR --snapshot $S2AG/abstracts
```

- `--snapshot`: the Semantic Scholar Academic Graph `abstracts` dataset files; release 2026-09-09 gives every missing abstract with a
  matching SHA-256, and `train_texts.jsonl` is then byte-identical to the training file of the paper (sha256 `d008ed77...`). `--api`: the Graph API (`S2_API_KEY` optional); abstracts changed since then are listed in
  `hash_mismatch.json` and keep the fetched text.
- Outputs next to the release files (or in `--out`): `eval_texts.jsonl` (`{query_id, query, field, pos: {cid, text}, negs: [{cid,
  stratum, text}]}`, text = `title. abstract` with the abstract cut at 2,000 characters), `train_texts.jsonl` (`{query_id, query, pos:
  [text], neg: [9 texts]}`, uncut abstracts), `papers_full.jsonl`, `hash_mismatch.json`.
- The code reads `EVAL_DIR` (CPR-eval: `eval.jsonl` + `eval_texts.jsonl`) and `TRAIN_DIR` (CPR-train: `train.jsonl` + `train_texts.jsonl`);
  `RELEASE_DIR`, the downloaded CPR-Dataset directory, gives `$RELEASE_DIR/eval` and `$RELEASE_DIR/train` when they are not set.

## Notes on the released splits

- The released CPR-eval combines two collection runs: 190 queries with 2010-19 base papers (base-paper pool `pool_2010_2019.json`,
  pooling on `universe_2019`, Stage 7 without `s7_adaptive`; a few of these pools
  used `POOL_K=1000`) and 1,110 queries with 2020-25 base papers from the 2010-2025 pool. Queries whose groups could not be filled with
  50 negatives were replaced by other final pairs of the same field. The assembly lists of these steps are not part of this code.
- The final splits leave out training pairs that share a base paper or a positive with the eval set (`export_train.py --exclude-eval`).
- Training pairs come from several collection waves over the 2010-2025 base-paper pool (each wave adds the same number of base papers
  per field, in pool order); `run.sh` runs one pass over `q_pool` / `q_per_field`.
- `encode_corpus.py` reproduces the stored pooling vectors on sample documents to float16 precision (same inputs, pooling and
  max length); the paper's vectors were encoded with another script on other hardware.
- Pinned model revisions of the three pooling retrievers: `negatives/retrievers.py`.
