# Held-out domain experiment

Does training on our data help on a domain that is not in the training set? For each of the four OpenAlex domains (Health, Life,
Physical, Social Sciences) we train Qwen3-Embedding-0.6B and 4B without that domain and evaluate on the CPR-eval queries of that
domain, next to the same model before training and after training on all of the training set.

## Folds

- Domain of a paper = one OpenAlex domain (top-1). Query: the pair's `field` (one of the 26 OpenAlex fields) -> domain
  (`evaluation/main/field_domain.json`). Positive: the domain of the paper's OpenAlex primary topic, looked up from its Semantic Scholar
  corpus id (S2 externalIds -> OpenAlex work by DOI, else PMID, else MAG). Hard negatives are not looked at.
- Training pairs whose positive has no OpenAlex domain are dropped: 10,573 -> 10,503 (70 pairs).
- Fold D = the pairs whose query **and** positive are both outside D (no leakage of D through either side), sampled down to the smallest
  fold, **5,896 pairs** (Physical), with `random.Random(42).sample`, rows kept in the original order.

| held-out | query in domain | only positive in domain | remaining | used | eval queries |
|---|---|---|---|---|---|
| Health | 2,137 | 1,126 | 7,240 | 5,896 | 250 |
| Life | 2,963 | 1,217 | 6,323 | 5,896 | 250 |
| Physical | 2,957 | 1,650 | 5,896 | 5,896 | 500 |
| Social | 2,446 | 692 | 7,365 | 5,896 | 300 |

- Eval for fold D = the CPR-eval queries whose query domain is D (selected by query domain only; 16 queries whose positive has no
  OpenAlex domain stay in), each with its full pool of 151 candidates.
- `build_folds.py` asserts per fold: no training query / positive in D, and no query / positive corpus id of D's eval queries among the
  fold's training queries / positives.

## Build

```bash
# TRAIN_DIR / EVAL_DIR: the CPR-train / CPR-eval release dirs after dataset/release/fetch_abstracts.py (or RELEASE_DIR = the downloaded CPR-Dataset directory)
# eval set pools: evaluation/main/prepare.py (eval.sh runs it too)
python -I ../../evaluation/main/prepare.py --work WORK/ours
python -I prep_pairs.py --train-dir $TRAIN_DIR --eval-dir $EVAL_DIR --out PAIRS        # ids, corpus ids and field of every pair (release files)
python -I fetch_openalex.py --pairs PAIRS --out META                    # ~20.5k papers; S2 + OpenAlex APIs, resumable caches in META/
python -I build_domains.py --pairs PAIRS --papers META/paper_domains.jsonl --out PAIRS
# Qwen3-Embedding rows are cut to 1023 tokens + <|endoftext|>: the max-length-1024 copy of the training set from ablation/dataset
# (DATA_ml1024/hardneg/ours/train.jsonl = make_maxlen.py on train_texts.jsonl)
python -I build_folds.py --data DATA_ml1024/hardneg/ours/train.jsonl --pairs PAIRS/pairs_train.jsonl --pairs-test PAIRS/pairs_test.jsonl \
  --pools WORK/ours/_data/pools.json --out FOLDS --seed 42
```

Outputs: `PAIRS/pairs_{train,test}.jsonl` (query / positive domain per pair), `FOLDS/fold_<domain>/{train.jsonl, train_ids.txt,
eval_ids.txt}`, `FOLDS/folds.json` (sizes, removal counts, domain counts, checks), `FOLDS/dropped_no_domain.jsonl`.
`fetch_openalex.py` takes optional `SEMANTIC_SCHOLAR_API_KEY` and `OPENALEX_MAILTO` from the environment.

Checksums (sha256 of `FOLDS/fold_<domain>/train.jsonl`; `--data $TRAIN_DIR/train_texts.jsonl` without the 1024 cut gives the same rows):

```
a9fc5c3b4e1f52c3d8dadfc273d23af00c211087e18104a277cdaa297b517ad4  fold_health   (1024 cut)
6b1b41f453ae289f6878d0cc41bdca71dc2781555b4f0b587738426a7a4e4722  fold_life     (1024 cut)
9c2b4aae62e819dc90c5709d02e988782de90db67f4dff6afe44dda1d25686ba  fold_physical (1024 cut)
796f3ece2592e0645782f83a5c59fb4c95fb004e6b7b0349b58d1687dcc759c2  fold_social   (1024 cut)
e9a1872bfa8db551f9048204491d9a874dd4bee5c92aa93ab88de9f9579cd0ad  fold_health   (uncut)
8e8bcac259853f3157b4f67d662262aa854d07a6f25893e7710562496f7092f5  fold_life     (uncut)
3de788fc0810abc954e712790b8c68d8a339aaa99ad4f4c630e33232a3e4ba6a  fold_physical (uncut)
11e29d36af4a7cb1c9d2e11e6817efb8ceb05396930e6ac331fc479aaaf89cec  fold_social   (uncut)
```

OpenAlex topic assignments change over time, so a new lookup can move a few positives to another domain and change the folds; the
checksums above hold for the domains used in the paper (`pairs_{train,test}.jsonl` of the paper run).

## Train

```bash
DATA=/path/FOLDS OUT=/path/runs bash run.sh                  # 8 runs: {qwen3_0.6b, qwen3_4b} x 4 held-out domains
DATA=/path/FOLDS OUT=/path/runs FOLDS="life" bash run.sh qwen3_4b
```

The recipe of the dataset ablation (`training/qwen/train.sh` defaults, official SWIFT.md); only the training data differ:

| setting | value |
|---|---|
| batch | 128 = 8 GPUs x 4 x grad-acc 4 (in-batch negatives per micro-batch of 32) |
| lr / schedule | 6e-6 cosine, no warmup; AdamW (0.9, 0.95), weight decay 0.1, clip 1.0 |
| loss | InfoNCE, temperature 0.1; all 9 hard negatives of the row + in-batch negatives |
| length | max length 1024 (data cut beforehand) |
| epochs | 1 (5,896 pairs -> 46 steps), final checkpoint evaluated, no validation split |
| other | bf16, flash attention, DeepSpeed ZeRO-2, gradient checkpointing, seed 42 / data seed 42, query prompt `Query:` |

## Evaluate

```bash
RUNS=/path/runs DATASET_RUNS=/path/dataset_runs EVAL_DIR=/path/eval_release WORK=/path/eval FOLDS_JSON=/path/FOLDS/folds.json bash eval.sh
# 12 models: 2 vanilla + 2 all-domain runs (ablation/dataset run.sh output, DATASET_RUNS) + 8 fold runs (run.sh output, RUNS)
```

- Every model encodes the whole eval set (187,411 documents, 1,300 queries; `evaluation/main/encode.py`, max length 1024, bf16, query
  prompt `Query:`, cosine) and `evaluation/main/score.py` ranks the positive in each query's pool of 151 (ties: positive last).
- `tables.py` builds, per base model, the rows *before training* (vanilla), *without domain* (each domain column scored with the fold
  model trained without it) and *all domains* (`abl2v2_<base>_hardneg_ours`, the dataset-ablation run on all 10,573 pairs), on the queries
  of each domain: R@1 / R@5 / R@10 / nDCG@10 with 95% CIs (bootstrap within the domain, 2,000 resamples, seed 0), and the Recall@10 gain
  of the fold models over the vanilla model (the paper's figure).
- `models.json`: the vanilla models with pinned HF revisions; every fine-tuned entry names its run dir (`"run"`: `$RUNS/<base>/<domain>`
  for the fold runs, `$DATASET_RUNS/<base>/hardneg/ours` for the all-domain runs). No fine-tuned checkpoints are distributed: `eval.sh`
  resolves each run to its last finalized `checkpoint-<N>` with `ablation/dataset/resolve_runs.py` and skips runs that are not found.
  The vanilla and all-domain entries are the same as in `ablation/dataset/models.json`, so with the same `WORK` their encodings are reused.

Outputs: encoder outputs in `WORK/ours/<model>/`; `results/ours/` (score.py over all queries and domains), `results/eval.json`
and `results/RES.md` (fold sizes with `FOLDS_JSON`, Recall@10 with CIs, R@1 / R@10, R@5 / nDCG@10, gains).

## Reproducibility

- `prep_pairs.py` on the release gives the corpus ids and fields of the paper run for every pair, and `build_folds.py` on the paper's
  domain assignments gives the paper's folds: identical training id lists, the same eval queries (the release ids; the paper run cut both
  corpus ids in eval ids to 8 digits), the same `folds.json` counts and the checksums above (both the cut and the uncut rows).
- `fetch_openalex.py` + `build_domains.py` re-run on a sample of pairs gave the same query and positive domains as the paper run.
- `run.sh` issues the same ms-swift arguments and training file as the paper runs (checked with a stubbed launcher).
- `score.py` + `tables.py` on the paper's per-query ranks give the paper's values and CIs (Recall@10 gain without the domain over
  before training: 0.21 to 0.36 for 0.6B, 0.38 to 0.54 for 4B).
