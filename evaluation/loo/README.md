# Leave-one-out pooling (paper Section 4.3, Figure robust (b))

Rebuilds the candidate pools of the 1,300 eval queries with one pooling retriever (E5 `intfloat/e5-large-v2`, SPLADE-v3
`naver/splade-v3`, SciNCL `malteos/scincl`) left out, judges the new candidates with the dataset code, evaluates the main-table
retrievers on each pool, and reports Recall@10 (mean over all queries, main pool of 151 documents) per retriever and pool, its change
against the original pools, and Kendall's tau-b between the retriever orders.

## Steps

| step | code | what it does |
|---|---|---|
| 1. pools | `build_pools.sh` (+ `pools.py`) | per left-out retriever: merge the other two retrievers' top-300 lists (`dataset/negatives/pool_universe.py merge`), judge with `dataset/negatives/judge_negatives.py` (same prompt, GPT-6.1 Sol, Batch API, 50 negatives per stratum), reusing every judgment of the original eval run; queries still short are re-scored at deeper `POOL_K` (1000, then 3000) and judged again |
| 2. export | `export.py` | three files `eval_loo_{e5,splade,scincl}.jsonl` in the `eval_texts.jsonl` format (same queries, positives and order; new negatives) |
| 3. evaluate | `run_eval.sh` (`prepare.py`, `split.py`) | `prepare.py` writes the three pools as one `evaluation/main` work dir (3 x 1,300 pseudo-queries `<query_id>@loo_<r>`, every document encoded once per model), `evaluation/main/encode.py` / `encode_lasttoken.py` encode and score each model with `evaluation/main/models.json` unchanged (max length 512), `split.py` splits the scores per pool and runs `evaluation/main/score.py` on each |
| 4. compare | `compare.py` | Recall@10 per retriever and pool, change against the original pools, largest absolute change, Kendall's tau-b vs the original order |

## Run

```bash
# 1. pools, once per pooling run of the eval set (2010-19 queries: <=2019 universe; 2020-25 queries: <=2025 universe)
SRC=/path/eval_run_2025 TAG=new CORPUS=/path/corpus_2025 VECTORS=/path/vectors_2025 BASE_CONFIG=/path/eval_config.json \
  EVAL=$EVAL_DIR/eval_texts.jsonl LOO_WORK=/path/loo bash build_pools.sh
SRC=/path/eval_run_2019 TAG=old CORPUS=/path/corpus_2019 VECTORS=/path/vectors_2019 BASE_CONFIG=/path/eval_config.json \
  EVAL=$EVAL_DIR/eval_texts.jsonl LOO_WORK=/path/loo bash build_pools.sh
# 2. export
python -I export.py --eval $EVAL_DIR/eval_texts.jsonl --loo-work /path/loo --out /path/loo_pools \
  --corpus '/path/corpus_2025/export/universe_*.parquet' '/path/corpus_2019/export/universe_*.parquet'
# 3 + 4. evaluate the main-table retrievers and compare with the main evaluation (evaluation/main/run.sh output)
EVAL_DIR=/path/eval_release WORK=/path/loo_eval LOO_DIR=/path/loo_pools ORIG=/path/main_results/per_query.json bash run_eval.sh
```

Inputs
- `SRC`: the work dir of the dataset pipeline's eval run (`dataset/`): `s6/queries.json`, `s6/top300_{e5,splade,scincl}.json`, the Stage 7
  judgments `s7/<pair>__<cid>.json` and the text cache `s7_texts.json`. `CORPUS` / `VECTORS`: that run's corpus and document-vector dirs
  (config keys `corpus` / `vectors`). `BASE_CONFIG`: the eval config of `dataset/` (Stage 7 settings); `build_pools.sh` writes one config
  per run with its own work dir, pool file names and `CAP` (USD, default 200).
- `EVAL_DIR`: the CPR-eval release dir after `dataset/release/fetch_abstracts.py` (`eval_texts.jsonl`; `RELEASE_DIR/eval` with `RELEASE_DIR` = the CPR-Dataset directory also works). Pools
  are matched to eval queries by (query text, positive corpus id), so the exported files carry the released `query_id`s.
- `ORIG`: `per_query.json` of `evaluation/main/score.py` on the original pools (same query ids).

Outputs: `LOO_WORK/<TAG>/pools_loo_<r>.json` (pools), new judgments / pool files inside `SRC/s7/` (the judgment directory is shared on purpose,
so a candidate is never judged twice), `LOO_DIR/eval_loo_<r>.jsonl`, `WORK/<model>/` (embeddings, combined scores), `WORK/_split/<pool>/`
(`score.py` output per pool: `results/{eval.json, eval.md, per_query.json}`), `WORK/_split/compare.json`.
Secrets: `OPENAI_API_KEY` (Stage 7), `GEMINI_API_KEY`, optional `HF_TOKEN`, from the environment or `ENV_FILE`. Python environments as in
`evaluation/main` (`QWEN_PY`, `LATEON_PY`, `SPECTER2_PY`, `M3_PY`, `INFRET_PY`); `PY` for the `dataset/` code.

## Details

- Per query and stratum (q_side / pos_side / other): the base paper's other cited papers first (`d1_cids`, not from a retriever), then
  the round-robin merge of the two remaining retrievers' lists (order scincl, splade, e5); the first 50 with usefulness <= 1 are kept.
- Queries with a stratum short of 50 in any of the three pools are re-scored at `POOL_K` 1000, then 3000 (`DEPTHS`), merged without
  the left-out retriever at the same depth and judged again; the deepest pool is used.
- The combined evaluation keeps one text per corpus id (the original eval set's text when the id is there).

## Reproducibility

- `prepare.py` on the three leave-one-out files of the paper run gives exactly the combined pools and texts that were evaluated
  (3,900 pseudo-queries); BM25 run through `evaluation/main/encode.py` + `split.py` gives the same rank of the positive for every query in
  all three pools (main and cross_domain) as the paper run.
- `compare.py` on the paper run's per-query ranks gives the values reported in the paper (largest change of Recall@10 below 0.04,
  Kendall's tau-b >= 0.83 against the original order).
- In the paper run the 1,110 queries of 2020-25 were filled at depths 300 and 3000 (the intermediate pass merged its 1,000-deep lists
  at depth 300) and one 2010-19 query at depth 1000; `DEPTHS="300 3000"` follows the 2020-25 part.
