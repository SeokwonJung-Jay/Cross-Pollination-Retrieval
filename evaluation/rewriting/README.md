# Query rewriting (HyDE, Query2doc, RAG-Fusion)

Query rewriting baselines on CPR-eval (paper Section 4.3 and Appendix F): GLM-5.3 rewrites each query, BM25 and Gemini Embedding
rank the query's 151 candidates (the main pool) with the rewritten query.

## Settings

Generation (`generate.py`), `z-ai/glm-5.3` through OpenRouter, reasoning effort `low`, `max_tokens` 4000 (reasoning included):

| method | prompt (`prompts/`, verbatim) | calls per query | temperature | output |
|---|---|---|---|---|
| HyDE | `hyde_plain.txt` | 8 | 0.7 | 8 passages |
| Query2doc | `query2doc.txt` (4 in-context query / abstract pairs from pilot data, none in the eval set) | 1 | 1.0 | 1 passage |
| RAG-Fusion | `ragfusion.txt` | 1 | 0.7 | first 4 lines longer than 10 characters, list markers and quotes stripped |

Retrieval (`score.py`), same BM25 and Gemini Embedding as `evaluation/main` (BM25: `bm25s` Lucene variant, tokens `[a-z0-9]+`, k1 1.5, b 0.75, idf within the
pool; Gemini: `gemini-embedding-001`, 768-d, L2-normalized, the document and original-query vectors of the main run):

| method | BM25 query | Gemini Embedding query vector |
|---|---|---|
| HyDE | query + the 8 passages | mean of the query vector and the 8 passage vectors (RETRIEVAL_DOCUMENT) |
| Query2doc | query x 5 + passage | RETRIEVAL_QUERY vector of `query [SEP] passage` |
| RAG-Fusion | reciprocal rank fusion (k = 60) of the rankings of the original query and the 4 generated queries | same, with RETRIEVAL_QUERY vectors |

Every generated passage is cut to its first 384 whitespace words (`--trunc 384`, about 512 tokens) before retrieval, as in the paper run.
Rank of the positive = its position after a stable sort by descending score over the pool order (positive first). Metrics: R@1/5/10,
MRR@10, nDCG@10 over all 1,300 queries (a query without an output counts as a miss), bootstrap 95% CI as in `evaluation/main/score.py`.

## Run

```bash
WORK=/path/main_work bash ../main/run.sh                     # or at least prepare.py, bm25 and gemini_emb_001 of the main evaluation
OPENROUTER_API_KEY=... python -I generate.py --work $WORK      # -> WORK/rewrite_glm53/{hyde_plain,query2doc,ragfusion}/<query_id>.json
GEMINI_API_KEY=... python -I score.py --work $WORK             # -> WORK/rewrite_glm53/results/{ranks.json, metrics.json}
```

`generate.py` resumes (existing files are kept) and logs a failed query instead of stopping; rerun until no query fails.
`score.py --retrievers bm25` needs no API key. The `original-bm25` / `original-gemini` rows of `score.py` are the main evaluation's
per-query ranks (`WORK/{bm25,gemini_emb_001}/results/per_query.json`), the paper's Original values. Packages: `openai`, `numpy`, `bm25s`,
`google-genai` (the M3 environment of `evaluation/main/env/m3.txt` has numpy, bm25s and google-genai).

## Reproducibility

- The paper run generated all 1,300 queries with 13,051 calls ($28.47); embedding the generated texts cost about $2.
- `score.py --retrievers bm25` on the paper run's generated texts reproduces the per-query rank of the positive for all 1,300 queries
  and all three methods (R@10: HyDE 0.4408, Query2doc 0.2954, RAG-Fusion 0.2354; Original 0.1646 from the main evaluation).
  The Gemini rows (paper: HyDE 0.2815, Query2doc 0.3046, RAG-Fusion 0.2285; Original 0.2569) use the same code path, but need the
  embeddings of the generated texts, which were not re-computed.
- LLM generations are not deterministic; a new generation gives close but not identical numbers.
