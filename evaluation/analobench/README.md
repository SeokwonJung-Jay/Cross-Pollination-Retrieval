# AnaloBench evaluation (dataset ablation)

AnaloBench (`jhu-clsp/AnaloBench`, pinned revision in `prepare.py`) used as a retrieval benchmark: story-level analogies at three story
lengths (S1 = 1 sentence, S10 ~ 10 sentences, S30 ~ 30 sentences; the same 340 stories). Used for the dataset ablation with **max length 1024**.

## Tasks and metrics

- **T2** (paper metric): each query story ranks its own bank of 200 stories (7.24 gold on average). nDCG@10 (binary gains), plus the
  AnaloBench T2 metrics P@3, P@5, R@3, R@5, MAP, MRR (over the top 10).
- **T1** (paper metric): each query story picks 1 of 4 options; accuracy on T1-Full (2,460 rows). T1-Subset (340 rows, AnaloBench's subset)
  is reported for reference.
- Paper table: T2 nDCG@10 per length and the mean over S1/S10/S30, T1 accuracy averaged over S1/S10/S30.
- 95% CI = bootstrap over query stories (2,000 resamples, seed 0). `random` row = exact random baseline.

## Models

Every story is encoded once as a query (`Query:` + story) and once as a document (story, no prompt), at max length 512 and 1024 (paper: 1024).

| model | name | settings |
|---|---|---|
| Qwen3-Embedding-0.6B / 4B (vanilla) | `qwen06b` / `qwen4b` | sentence-transformers, bf16, left padding, last-token pooling, L2-normalized, cosine |
| llama-embed-nemotron-8b (vanilla) | `nemotron8b_qp` | `nvidia/llama-embed-nemotron-8b` @ `aa3b43a4`, official remote code (transformers 4.51.0), bf16, flash_attention_2, left padding, mean pooling, L2-normalized, cosine |
| fine-tuned Qwen3-Embedding | `abl2v2_qwen3_<size>_<neg>_<data>` + `--ckpt PATH` | as the vanilla model |
| fine-tuned llama-embed-nemotron-8b | `abl2v2_nemotron8b_<neg>_<data>` + `--ckpt PATH` | as `nemotron8b_qp` (`--ckpt` = the folder written by `training/nemotron/export.sh`) |

Batch: 32 (Qwen 0.6B), 16 (Qwen 4B, llama-embed-nemotron-8b). Environments: Qwen = the `training/qwen` environment;
llama-embed-nemotron-8b = `training/nemotron/setup_env_eval.sh`.

## Run

```bash
python -I prepare.py --work WORK
python -I encode.py qwen06b --work WORK --batch 32
python -I encode.py abl2v2_qwen3_4b_hardneg_ours --ckpt CKPT_DIR --work WORK --batch 16
python -I encode.py nemotron8b_qp --work WORK --batch 16
python -I score.py --work WORK --models qwen06b abl2v2_qwen3_4b_hardneg_ours ... --out RESULTS
```

Outputs: `WORK/data/L<n>.json` (story pools and tasks), `WORK/emb/<model>/m<max_len>/L<n>_q.npy`, `L<n>_d.npy` (query / document vectors).
Tables: `RESULTS/{eval.json, eval.md, per_query.json}` (per query: gold ranks in the bank for T2, 0/1 per row for T1).

## Reproducibility

- `score.py` from this folder, run on the paper's embeddings, gives `eval.json` and `per_query.json` identical to the paper's results;
  the llama-embed-nemotron-8b T2 nDCG@10 / T1 means recomputed from the paper's per-query results equal its `eval.json`.
- With a stubbed sentence-transformers, `encode.py` makes the same model loads and encode calls and writes byte-identical embeddings as the
  encoder of the paper's evaluation, for the Qwen and the llama-embed-nemotron-8b models.
