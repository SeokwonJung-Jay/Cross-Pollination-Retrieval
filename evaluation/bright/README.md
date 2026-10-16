# BRIGHT evaluation (dataset ablation)

BRIGHT (`xlangai/BRIGHT`), the 7 StackExchange splits, official setting: every query searches its split's whole corpus and the query's
`excluded_ids` are removed from the ranking. Used for the dataset ablation with **max length 1024**.

## Metrics

- nDCG@10 per split (binary gains on `gold_ids`, as BRIGHT's pytrec_eval) + macro mean over the 7 splits (paper metric).
  `score.py` also reports R@1/5/10 and MRR@10, a bootstrap 95% CI over queries (2,000 resamples, seed 0) and the exact random baseline
  (`../common/random_baseline.py`).
- All queries of each split (749 over the 7 splits). Optional `--sample N --sample_splits ...` scores `random.Random(20261007).sample(sorted
  query ids, N)` of each listed split instead (ids written to `OUT/sample_<split>.json`); retrieval still searches the whole corpus.

## Models

| model | name | settings |
|---|---|---|
| Qwen3-Embedding-0.6B / 4B (vanilla) | `qwen06b_qp` / `qwen4b_qp` | sentence-transformers, bf16, left padding, last-token pooling, L2-normalized, query prompt `Query:`, no document prompt, cosine |
| llama-embed-nemotron-8b (vanilla) | `nemotron8b_qp` | `nvidia/llama-embed-nemotron-8b` @ `aa3b43a4`, official remote code (transformers 4.51.0), bf16, flash_attention_2, left padding, mean pooling, L2-normalized, query prompt `Query:` (not the model's default instruction), no document prompt, cosine |
| fine-tuned Qwen3-Embedding | `abl2v2_qwen3_<size>_<neg>_<data>` + `--ckpt PATH` | as the vanilla model, query prompt saved in the checkpoint (`Query:`) |
| fine-tuned llama-embed-nemotron-8b | `abl2v2_nemotron8b_<neg>_<data>` + `--ckpt PATH` | as `nemotron8b_qp` (`--ckpt` = the folder written by `training/nemotron/export.sh`) |

Max length `QWEN_EVAL_MAXLEN` (default 1024, every model); above 512 the texts are encoded in token-budget batches (longest first, at most
16,384 padded tokens and `--batch` texts). Batch: 32 (Qwen 0.6B), 16 (Qwen 4B, llama-embed-nemotron-8b). Environments: Qwen = the
`training/qwen` environment; llama-embed-nemotron-8b = `training/nemotron/setup_env_eval.sh`.

## Run

```bash
SE="biology earth_science economics psychology robotics stackoverflow sustainable_living"
python -I prepare.py --out DATA --splits $SE
# per model (environment: see Models)
python -I encode.py qwen06b_qp --data DATA --out OUT --splits $SE --batch 32
python -I encode.py abl2v2_qwen3_4b_hardneg_ours --ckpt CKPT_DIR --data DATA --out OUT --splits $SE --batch 16
python -I encode.py nemotron8b_qp --data DATA --out OUT --splits $SE --batch 16
# optional, multi-GPU: document chunks of a split in n parts (one GPU each), then one call without --shard for queries + ranking
python -I encode.py MODEL --data DATA --out OUT --splits earth_science --shard 0/5 ...
# all models found in OUT
python -I score.py --data DATA --out OUT --splits $SE
```

Outputs per model and split in `OUT/<model>/<split>/`: document vectors (`chunk_*.npy`, 2,000 documents each in corpus order, resumable),
query vectors (`q_emb.npy` in `q_ids.json` order), `run.json` (top 1,000 documents and scores per query). Tables: `OUT/eval_full.{json,md}`.

## Reproducibility

- `score.py` from this folder, run on the paper's runs, gives an `eval_full.json` identical to the paper's results (Qwen3-Embedding and
  llama-embed-nemotron-8b: all 9 models, every split, metric and CI).
- With a stubbed sentence-transformers, `encode.py` makes the same model loads and encode calls and writes byte-identical outputs as the
  encoder of the paper's evaluation, for the Qwen and the llama-embed-nemotron-8b models.
