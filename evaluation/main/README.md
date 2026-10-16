# Main evaluation (CPR-eval)

11 retrievers on our eval set with the same pools, the same metric code and **max length 512**.

## Eval set and metrics

- CPR-eval from the id-only release (`eval_texts.jsonl` written by `dataset/release/fetch_abstracts.py`; `prepare.py` reads it from
  `EVAL_DIR`, else `RELEASE_DIR/eval`): 1,300 queries (26 OpenAlex fields x 50), 187,411 candidate documents. Query ids are the release ids.
- Each query is ranked against its own pool of 151 documents: the positive + 150 negatives (50 from the query's field, 50 from the positive's
  field, 50 other). `cross_domain` drops the query-field negatives (101 documents).
- Metrics: R@1, R@5, R@10, MRR@10, nDCG@10 of the positive; ties rank the positive after every equal-score document.
- Domains: query field -> OpenAlex domain (`field_domain.json`): Health 250 / Life 250 / Physical 500 / Social 300. Avg = mean over all
  queries. 95% CI = bootstrap over queries within each domain (2,000 resamples, seed 0).
- Paper main table = R@1, R@10 per domain + Avg (main pool); appendix = R@5, nDCG@10.

## Models (exact settings in `models.json`, revisions pinned)

| family | model | settings |
|---|---|---|
| Sparse | BM25 | `bm25s` (Lucene variant), tokens `[a-z0-9]+`, k1 1.5, b 0.75, idf within the query's pool |
| Sparse | BGE-M3 (sparse) | `BAAI/bge-m3`, FlagEmbedding, max 512, fp16, lexical matching score |
| Dense (general) | Qwen3-Embedding-0.6B / 4B | sentence-transformers, bf16, max 512, query prompt `Query:`, no document prompt, cosine |
| Dense (general) | BGE-M3 (dense) | normalized CLS, dot product |
| Dense (general) | Gemini Embedding | `gemini-embedding-001`, 768 d, RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT, L2-normalized, cosine |
| Dense (scientific) | SPECTER2 | `allenai/specter2_base` + `specter2` (proximity) for documents / `specter2_adhoc_query` for queries, CLS, documents `title [SEP] abstract` |
| Dense (scientific) | OpenScholar Retriever | Contriever-style mean pooling, no normalization, dot product, max 512 |
| Dense (reasoning) | inf-retriever-v1-pro | `infly/inf-retriever-v1-pro` (Qwen2.5-7B, remote code, transformers 4.51.0), fp16, max 512, official query prompt, last-token pooling (`encode_lasttoken.py`) |
| Late interaction | LateOn | PyLate, query 512 / document 512 tokens, MaxSim (sum) |
| Late interaction | BGE-M3 (multi-vector) | FlagEmbedding colbert_score |

## Run

```bash
RELEASE_DIR=/path/CPR-Dataset WORK=/path/main_work ENV_FILE=/path/.env bash run.sh     # prepare -> encode (GPU queue + CPU/API models + inf-retriever) -> score
```
Environments (`env/*.txt` = exact package lists): `QWEN_PY` (the Qwen training environment), `LATEON_PY`, `SPECTER2_PY`, `M3_PY`
(BM25 via `bm25s`, BGE-M3, OpenScholar, Gemini), `INFRET_PY` (inf-retriever, transformers 4.51.0). `GEMINI_API_KEY` for Gemini.

Outputs per model in `WORK/<model>/`: `embeddings/` (document / query vectors), `results/scores.json` (score of every pool document per query),
`results/{metrics.json, per_query.json}` (rank of the positive per query), `config.json`. Tables: `results/{eval.md, eval.json, per_query.json}`.

Other experiments pass their own registry (`--registry`, e.g. `ablation/dataset/models.json`). A `kind: st` entry with `"remote_code": true`
(llama-embed-nemotron-8b in the dataset ablation) loads the model's official remote code (`trust_remote_code`, flash_attention_2) and needs
the `training/nemotron/setup_env_eval.sh` environment (transformers 4.51.0, sentence-transformers 5.1.2).

## Reproducibility

- `prepare.py` on the release (texts rebuilt with `fetch_abstracts.py --snapshot`, Semantic Scholar release 2026-09-09) gives the
  pools of the paper run for all 1,300 queries (same positives, negatives, strata and order).
- The release query ids (`scaleup_<query corpus id>_<positive corpus id>`) sort in the same order as the ids of the paper run (the same
  ids with both corpus ids cut to 8 digits), so the bootstrap resamples (drawn over sorted ids) and the confidence intervals are the same.
- `encode.py --devices` (multi-GPU) reproduces the single-GPU batches, so document vectors are bitwise identical.
- Qwen3-Embedding-0.6B re-encoded with this code on another machine gives the same rank of the positive for all 1,300 queries as the run behind the paper.
