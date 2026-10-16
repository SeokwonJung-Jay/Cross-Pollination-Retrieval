# LLM reranker (main table)

The LLM reranker row of the main table: MiMo-V2.6-Pro reorders the same 151 candidates of each CPR-eval query in a tournament
(paper Appendix D).

## Settings

- Model `xiaomi/mimo-v2.6-pro` through OpenRouter (OpenAI-compatible client), `max_tokens` 32000, cost from OpenRouter `usage.cost`.
- Candidates = the query's main pool (positive + 150 negatives, `evaluation/main/prepare.py`), shuffled with
  `random.Random(f'upper-{sid}')` where `sid` = the query id with both corpus ids cut to their first 8 digits
  (`scaleup_<query cid[:8]>_<positive cid[:8]>`), shown as `[i] title. abstract`.
- Round 1: 8 batches `docs[i::8]` (18-19 candidates), listwise prompt, top 5 of each -> 40. Round 2: 2 batches of 20, top 5 of each -> 10
  finalists. Both rounds: reasoning effort `low`.
- Final: each finalist scored alone from 0 to 10 with the pointwise prompt (model default reasoning; an unparsed score is -1).
  When several finalists share the top score, one listwise call (k = number tied, default reasoning) orders them.
- Ranking: finalists (score descending; top tie by that call; other ties by round-2 order), then round-2 losers, then round-1 losers.
- Parsing: the last `{"top": [...]}` (listwise) / `{"score": n}` (pointwise) in the answer; listwise ids are 1-based within the batch,
  invalid and duplicate ids are dropped. A round with no parsed batch keeps each batch's first 5 in input order; a round where only some
  batches parse passes on fewer candidates (in the paper run 17 queries had fewer than 10 finalists).
- Calls: 4 attempts per call (sleep 20 s, 40 s, 60 s); the calls of one step run concurrently, 32 queries at a time; a ledger of every
  call in `WORK/<name>/ledger.jsonl`; `--cap` stops the run at a USD budget.
- Prompts (`prompts/`, verbatim): `listwise.txt` (selection rounds and tie break; `{k}` = 5 or the number tied), `pointwise.txt`.

## Run

```bash
EVAL_DIR=/path/eval_release python -I ../main/prepare.py --work $WORK   # WORK/_data (skip if the main evaluation already ran in WORK)
OPENROUTER_API_KEY=... python -I rerank.py --work $WORK                 # -> WORK/llm_reranker_mimo/
python -I score.py --work $WORK --ranks $WORK/llm_reranker_mimo/results/ranks.json --out $WORK/llm_reranker_mimo/results
```

`rerank.py` resumes: a query whose `queries/<query_id>.json` exists is not called again. `--limit N` runs the first N queries.
Needs `openai` (any recent version); `score.py` needs only the standard library.

Outputs in `WORK/llm_reranker_mimo/`: `queries/<query_id>.json` (full ranking, finalist scores, round where the positive dropped out,
calls, cost), `results/ranks.json` (rank of the positive), `results/scores.json` (`{query_id: {cid: 151 - position}}`, the input format
of `evaluation/main/score.py`), `results/metrics.json` (from `score.py`).

## Metrics

`score.py` follows `evaluation/main/score.py`: R@1/5/10, MRR@10, nDCG@10 over all 1,300 queries (a query without a ranking counts as a
miss), per OpenAlex domain (Health / Life / Physical / Social) and Avg = mean over all queries, 95% CI = bootstrap over queries (2,000
resamples, seed 0; stratified by domain for the domain table). It accepts `ranks.json` or `scores.json`.

## Reproducibility

- The code is the script behind the paper run (1,300 queries, OpenRouter, 2026-10): `score.py` on that run's per-query ranks gives the
  paper's values (R@1 0.579, R@10 0.8146, per domain Health 0.60 / 0.86, Life 0.60 / 0.80, Physical 0.56 / 0.77, Social 0.58 / 0.86).
  That run made 20.2 calls per query (top-score ties in 362 queries) and cost $91.78.
- The shuffle seed id `sid` is the query id the paper run seeded with (identical for all 1,300 queries), so the batches are those of the
  paper run (checked for all 1,300 queries: the 40 round-1 winners of every query contain exactly 5 candidates of each of its 8 batches).
- LLM outputs are not deterministic; a rerun gives a close but not identical ranking.
