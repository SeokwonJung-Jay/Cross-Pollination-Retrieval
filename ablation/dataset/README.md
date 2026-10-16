# Dataset ablation

Four training sets of **10,573 rows** each (the size of our training set), each in two negative settings:

| set | source | hard negatives per row |
|---|---|---|
| ours | CPR-train (`train_texts.jsonl` of the release, `dataset/release/fetch_abstracts.py`) | 9 |
| msmarco | `microsoft/ms_marco` v1.1 train | 9 |
| reasonir | `reasonir/reasonir-data` hq + vl (hq 3,058 / vl 7,515 sampled) | 1 |
| specter | `allenai/scirepeval` cite_prediction | 1 |

- `hardneg`: hard negatives above + in-batch negatives.
- `inbatch`: the same rows with the hard negatives removed (in-batch negatives only).

## Build

```bash
# HF dataset snapshots under RAW_DIR/<org>__<name>/: allenai__scirepeval (781d35d1), reasonir__reasonir-data (0275f825),
# microsoft__ms_marco (a47ee7aa); BRIGHT documents (xlangai/BRIGHT 3066d29c) for the ReasonIR hq positives.
# TRAIN_DIR = the CPR-train release dir after fetch_abstracts.py (holds train_texts.jsonl)
python -I build_data.py --raw RAW_DIR --bright RAW_DIR/xlangai__BRIGHT --out DATA/hardneg --n-from $TRAIN_DIR/train_texts.jsonl --seed 42
cp $TRAIN_DIR/train_texts.jsonl DATA/hardneg/ours/train.jsonl
python -I make_inbatch.py DATA
# max length 1024 for Qwen3-Embedding (texts cut to 1023 tokens + <|endoftext|>); llama-embed-nemotron-8b trains on DATA itself
python -I ../../training/common/make_maxlen.py DATA DATA_ml1024 1024
```

Eligible rows before sampling: specter 638,309 / reasonir 345,476 / msmarco 12,291.

## Checksums (sha256 of `train.jsonl`, before `make_maxlen.py`)

`hardneg/ours` = `train_texts.jsonl` of the release with the abstracts from the Semantic Scholar snapshot (release 2026-09-09,
`fetch_abstracts.py --snapshot`).

```
d008ed7753ec992533339dd4d08db6017b45ba08b10b801c455f362d37967155  hardneg/ours
25ef7fb07ba675b1a756ed2efe1fee2d44654de8b6171496ce7c55b6ec4db086  hardneg/msmarco
f24a7c5bf9f191b20a16115d74db4adb3a19548890f4c17f111d0391aed3d8e5  hardneg/reasonir
523fc35307c3ff7a48238cf6af1ae287b8c9436570ac82e1d59672f597fd7b74  hardneg/specter
9f29997148ec07365c620d6fab443dcafd11d545f4b8fb628f2075ed3c759f46  inbatch/ours
23a7aa5c33cc4a2f4793b640953a46d02bcbe4b1810ac16e381cb8599c1ca18c  inbatch/msmarco
0cb62f08d685b2c7a6dae0f0052451489bb4e373cdf15c5c2f658400a2efe726  inbatch/reasonir
64a164f12cd9edd7349ff92c149d4373c560c9859daecc1204c1088396305fed  inbatch/specter
```

## Train

```bash
DATA=/path/DATA_ml1024 DATA_FULL=/path/DATA OUT=/path/runs bash run.sh   # 24 runs: {qwen3_0.6b, qwen3_4b, nemotron8b} x {hardneg, inbatch} x 4 sets
DATA=/path/DATA_ml1024 OUT=/path/runs bash run.sh qwen3_4b                # one base model
```

All runs: 1 epoch (83 steps), final checkpoint evaluated, 8 GPUs, flash attention, seed 42, max length 1024. Qwen3-Embedding trains on the
`make_maxlen.py` output (`DATA`); llama-embed-nemotron-8b on the uncut sets (`DATA_FULL`; the NeMo collator truncates to 1024 tokens).

| | Qwen3-Embedding-0.6B / 4B | llama-embed-nemotron-8b |
|---|---|---|
| recipe | official SWIFT.md, ms-swift (`training/qwen/train.sh`) | official NeMo AutoModel bi-encoder recipe (`training/nemotron/train.sh`) |
| batch | 128 = 8 GPUs x 4 x grad-acc 4 | 128 = 8 GPUs x 1 x grad-acc 16 |
| lr / schedule | 6e-6 cosine, no warmup | 2e-6, 2 warmup steps, cosine to 1% |
| optimizer | AdamW (0.9, 0.95), wd 0.1, clip 1.0 | AdamW (0.9, 0.999), wd 0.01, clip 1.0 |
| loss | InfoNCE, temperature 0.1 | InfoNCE, temperature 0.02 |
| hard negatives | all in the row (9 / 1 / 0) | all in the row (9 / 1 / 0; `n_passages` 10 / 2 / 1) |
| in-batch negatives | per micro-batch of 32 queries (8 GPUs x 4) | gathered across the 8 GPUs (8 queries per micro-step) |
| max length | 1024 (data cut by `make_maxlen.py`) | query 1024 / document 1024 (collator) |
| pooling | last token, L2-normalized | mean, L2-normalized |
| precision | bf16 | fp32 weights, bf16 computation |
| prompts | query `Query:`, documents none | query `Query:`, documents none |
| memory | DeepSpeed ZeRO-2, gradient checkpointing | FSDP2, activation checkpointing |
| checkpoint for evaluation | finalized `checkpoint-83` | `eval_model/` (`export.sh`: consolidated bf16 + the base repo's remote code) |

## Evaluate

```bash
RUNS=/path/runs EVAL_DIR=/path/eval_release WORK=/path/eval bash eval.sh     # every model in models.json (3 vanilla + 24 runs)
RUNS=/path/runs EVAL_DIR=/path/eval_release WORK=/path/eval bash eval.sh qwen3_4b abl2v2_qwen3_4b_hardneg_ours
```

Max length 1024 everywhere, query prompt `Query:`, no document prompt. Environments: `QWEN_PY` (Qwen, the `training/qwen` environment),
`NEMO_PY` (llama-embed-nemotron-8b, `training/nemotron/setup_env_eval.sh`: transformers 4.51.0 for the official remote code).
Benchmarks (code in `../../evaluation/`):

| benchmark | metric (paper) | code |
|---|---|---|
| BRIGHT, 7 StackExchange splits, all 749 queries, whole split corpus | nDCG@10 per split + macro | `evaluation/bright` |
| AnaloBench | T2 nDCG@10 (S1 / S10 / S30, mean), T1 accuracy (mean) | `evaluation/analobench` |
| CPR-eval | R@1 / R@10 per domain + Avg over all queries (pool 151) | `evaluation/main` (`encode.py`, `score.py`) with `models.json` |

`models.json` = registry for CPR-eval and the list of models `eval.sh` evaluates: vanilla `qwen3_0.6b`, `qwen3_4b`, `nemotron8b` (HF
repos, pinned revisions) and the runs `abl2v2_<base>_<neg>_<set>`, each pointing to its `run.sh` output dir (`"run": "$RUNS/<base>/<neg>/<set>"`).
No fine-tuned checkpoints are distributed: `eval.sh` evaluates your own runs. `resolve_runs.py` (run by `eval.sh`) resolves each run to
its final checkpoint (Qwen: the last finalized `checkpoint-<N>`; llama-embed-nemotron-8b: `eval_model/` from `export.sh`), writes the
resolved registry to `WORK/_logs/models.json` and reports runs that are not found (or not finished: no `DONE`); those are skipped.
`EVAL_DIR` (else `RELEASE_DIR/eval`) = the CPR-eval release dir after `dataset/release/fetch_abstracts.py` (`evaluation/main/prepare.py`).

Outputs: encoder outputs in `WORK/{bright,analobench,ours}/`; tables in `results/` (`bright/eval_full.*`, `analobench/eval.*`,
`ours/eval.*`) and `results/RES.md` (summary + one table per benchmark, written by `tables.py`).

Reproducibility: with stubbed encoders, `eval.sh` issues the same encoder and scoring calls (model, checkpoint, split, shard, batch,
max length) as the evaluation run for the paper; `tables.py` on the paper's results gives the paper's tables. For llama-embed-nemotron-8b,
the scorers here on the paper's per-query results give its values (e.g. CPR-eval R@10 0.249 before training and 0.700 after
hardneg / ours; BRIGHT macro nDCG@10 0.194 / 0.328), and `training/nemotron/train.sh` writes the configuration of each of the 8 runs.
