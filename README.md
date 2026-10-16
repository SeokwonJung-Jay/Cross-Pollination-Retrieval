<div align="center">

# Cross-Pollination Retrieval

**Finding papers on other topics whose ideas can be transferred to a research problem**

Data: [CPR-Dataset](https://huggingface.co/datasets/CPR-Dataset/CPR-Dataset) (splits `eval` = CPR-eval, `train` = CPR-train)

</div>

---

[Quick Start](#quick-start) · [Overview](#overview) · [Task](#task) · [Dataset at a Glance](#dataset-at-a-glance) · [Schema](#schema) · [Repository](#repository) · [Reproducing the Paper](#reproducing-the-paper) · [License](#license) · [Citation](#citation)

---

## Quick Start

### Getting the data

CPR-Dataset is released on Hugging Face in an id-only form: queries, labels, and Semantic Scholar corpus ids, with abstracts
included only where their license allows (CC-BY, CC0, public domain). `fetch_abstracts.py`, shipped with the dataset, fills in the
remaining abstracts and writes the text files that the code reads.

```bash
huggingface-cli download CPR-Dataset/CPR-Dataset --repo-type dataset --local-dir $RELEASE_DIR
EVAL_DIR=$RELEASE_DIR/eval; TRAIN_DIR=$RELEASE_DIR/train

# abstracts from the Semantic Scholar "abstracts" dataset (release 2026-09-09; exact match), or --api for the Graph API
python3 dataset/release/fetch_abstracts.py --release $EVAL_DIR  --snapshot /path/s2ag/abstracts   # -> $EVAL_DIR/eval_texts.jsonl
python3 dataset/release/fetch_abstracts.py --release $TRAIN_DIR --snapshot /path/s2ag/abstracts   # -> $TRAIN_DIR/train_texts.jsonl
```

```python
import json

for line in open(f"{EVAL_DIR}/eval_texts.jsonl"):
    q = json.loads(line)
    candidates = [q["pos"]] + q["negs"]      # 151 papers: rank them for q["query"]
    print(q["field"], "|", q["query"])
    print(len(candidates), candidates[0]["text"][:100])
    break
```

## Overview

![CPR-eval spans all four OpenAlex domains](https://huggingface.co/datasets/CPR-Dataset/CPR-Dataset/resolve/main/assets/fig_landscape.png)

Many scientific advances come from *cross-pollination*: carrying an idea from one field over to a problem in another. Yet existing
retrievers return papers that are semantically similar to the query, and benchmarks for finding papers that inspire research keep
their query-positive pairs within a narrow region of science (above: 100 random pairs per benchmark on a map of paper embeddings,
colored by OpenAlex domain; a filled dot marks the query and an open circle its positive).

We introduce **cross-pollination retrieval**: given a research problem, find papers on *other topics* whose ideas can be transferred
to it. We build **CPR-Dataset** with a scalable pipeline that mines citations through which a paper transferred an idea from another
topic. **CPR-eval** has **1,300 queries across 26 fields**, and **CPR-train** has **10,573 training pairs**.

![An example of cross-pollination retrieval](https://huggingface.co/datasets/CPR-Dataset/CPR-Dataset/resolve/main/assets/fig_example.png)

For the query above, a paper on the same topic (Generative Adversarial Nets) does not offer what the query needs. A paper on another
topic, equilibrium free-energy differences in statistical physics, does: the data distribution maps to an initial equilibrium state
and the tractable distribution to a final equilibrium state, so its idea can be transferred to the query's problem.

## Task

Given a query that describes the research problem of a base paper, rank candidate papers so that a paper meeting all three conditions
comes first:

| | Condition | Meaning |
|---|---|---|
| **1** | **Cross-topic** | The paper shares no OpenAlex topic with the base paper |
| **2** | **Analogous** | The two problems share at least two objects playing the same role and one relation among them |
| **3** | **Transferable** | The paper offers a method, finding, or formulation that can be carried over to the problem of the query |

Each CPR-eval query comes with a fixed pool of **151 candidates**: its positive and 150 hard negatives judged not relevant. Report
Recall@1/5/10 and MRR@10 over all 1,300 queries (random ranking: Recall@10 = 10/151).

## Dataset at a Glance

| | CPR-eval | CPR-train |
|---|---|---|
| Queries | 1,300 (26 OpenAlex fields x 50) | 10,573 |
| OpenAlex fields / domains | 26 / 4 | 26 / 4 |
| Base papers published | 2010-2025 | 2010-2025 |
| Candidates per query | 1 positive + 150 negatives | 1 positive + 9 negatives |
| Negative strata | 50 each: `q_side`, `pos_side`, `other` | 3 each: `q_side`, `pos_side`, `other` |
| Papers in `papers.jsonl` | 188,475 | 103,338 |
| Paper text | title + abstract (abstract cut at 2,000 characters) | title + full abstract |

No base paper or positive of CPR-eval appears in CPR-train. Negative strata: `q_side` shares an OpenAlex topic with the base paper,
`pos_side` with the positive, `other` with neither.

## Schema

**`eval/eval.jsonl` / `train/train.jsonl`** (Hugging Face, one query per line)

| Field | Type | Notes |
|---|---|---|
| `query_id` | string | `scaleup_<base paper corpus id>_<positive corpus id>` |
| `query` | string | The research problem of the base paper |
| `field` | string | OpenAlex field of the base paper |
| `q_corpusid` | int | Semantic Scholar corpus id of the base paper |
| `q_year` | int | Publication year of the base paper |
| `positive` | object | `{corpusid}` |
| `negatives` | list&lt;object&gt; | `[{corpusid, stratum, judge_score}]` |

**`eval/papers.jsonl` / `train/papers.jsonl`** (Hugging Face, one paper per line): `corpusid`, `title`, `year`, `externalIds` (DOI, PubMed, ...), `roles`,
`abstract_license`, `abstract_redistributable`, `abstract` (null unless redistributable), `abstract_sha256`.

**`eval_texts.jsonl`** (written by `fetch_abstracts.py`): `{query_id, query, field, pos: {cid, text}, negs: [{cid, stratum, text}]}`.

**`train_texts.jsonl`** (written by `fetch_abstracts.py`): `{query_id, query, pos: [text], neg: [9 texts]}`, the training format
of `training/`.

Croissant metadata (with Responsible AI fields) is in `croissant.json` of the dataset.

## Repository

| Paper | Code |
|---|---|
| Dataset construction (Section 3) | [`dataset/`](dataset/) |
| Main results: 11 retrievers on CPR-eval | [`evaluation/main/`](evaluation/main/) |
| LLM reranker | [`evaluation/reranker/`](evaluation/reranker/) |
| Removing same-topic candidates | [`evaluation/main/`](evaluation/main/) (`score.py`, 101-candidate pools) |
| Leave-one-out pooling | [`evaluation/loo/`](evaluation/loo/) |
| Query rewriting (HyDE, Query2doc, RAG-Fusion) | [`evaluation/rewriting/`](evaluation/rewriting/) |
| Training Qwen3-Embedding 0.6B / 4B | [`training/qwen/`](training/qwen/) |
| Training llama-embed-nemotron-8b | [`training/nemotron/`](training/nemotron/) |
| Training on CPR-train vs. MS MARCO, ReasonIR, SPECTER data (hard / in-batch negatives) | [`ablation/dataset/`](ablation/dataset/) |
| Unseen domains | [`ablation/domain/`](ablation/domain/) |
| AnaloBench, BRIGHT | [`evaluation/analobench/`](evaluation/analobench/), [`evaluation/bright/`](evaluation/bright/) |

Each folder has a README with its exact settings, commands, and outputs.

<details>
<summary>Folder layout</summary>

```
dataset/        construction pipeline: corpus, positives, hard negatives, export, id-only release (fetch_abstracts.py)
training/
  qwen/         Qwen3-Embedding full fine-tuning with ms-swift
  nemotron/     llama-embed-nemotron-8b full fine-tuning with NeMo AutoModel
  common/       data check (validate_data.py), token-budget cut (make_maxlen.py)
evaluation/
  main/         11 retrievers on CPR-eval (151- and 101-candidate pools)
  reranker/     LLM reranker (tournament, OpenRouter)
  rewriting/    query rewriting with BM25 / Gemini Embedding
  loo/          leave-one-out pooling
  bright/       BRIGHT, 7 StackExchange splits, all 749 queries
  analobench/   AnaloBench
  common/       exact random baseline
ablation/
  dataset/      training-data comparison (24 runs)
  domain/       held-out domains (8 runs)
assets/         figures of this README
```

</details>

## Reproducing the Paper

**Setup.** Linux, CUDA 12.8, Python 3.12 ([uv](https://github.com/astral-sh/uv)); our runs used 8 x A100 80GB.

```bash
bash training/qwen/setup_env.sh            # training/.venv-qwen     (also the Qwen evaluation environment)
bash training/nemotron/setup_env.sh        # training/.venv-nemotron
bash training/nemotron/setup_env_eval.sh   # training/.venv-nemotron-eval
cp .env.example .env                       # API keys: OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, ...
```

Each setup script installs the pinned versions in its lock file. The other evaluation environments of the main table are exact package
lists in `evaluation/main/env/*.txt`, and the dataset pipeline lists its packages in `dataset/requirements.txt`. Secrets are read only
from the environment or the git-ignored `.env` at the repository root.

**Data paths.** Scripts that need the eval set read `EVAL_DIR` (`eval.jsonl`, `eval_texts.jsonl`); scripts that need the training set
read `TRAIN_DIR` (`train.jsonl`, `train_texts.jsonl`). `RELEASE_DIR`, the downloaded CPR-Dataset directory, gives `$RELEASE_DIR/eval` and `$RELEASE_DIR/train` when either is unset.

**Training data format.** One JSONL row per query: `{"query": str, "pos": [str], "neg": [str, ...]}` with the same number of hard
negatives in every row (none for in-batch training). Both `train.sh` scripts check the file with `training/common/validate_data.py`.

```bash
# evaluate the 11 retrievers on CPR-eval (max length 512)
EVAL_DIR=$EVAL_DIR WORK=/path/main_work bash evaluation/main/run.sh

# train on CPR-train (paper settings are the defaults: 1 epoch, max length 1024, query prompt 'Query:')
TRAIN_DATA=$TRAIN_DIR/train_texts.jsonl OUTPUT_DIR=runs/qwen bash training/qwen/train.sh
TRAIN_DATA=$TRAIN_DIR/train_texts.jsonl OUTPUT_DIR=runs/nemotron bash training/nemotron/train.sh && bash training/nemotron/export.sh runs/nemotron

# training-data comparison: 24 runs, then BRIGHT / AnaloBench / CPR-eval
DATA=/path/data_ml1024 DATA_FULL=/path/data OUT=/path/runs bash ablation/dataset/run.sh
RUNS=/path/runs EVAL_DIR=$EVAL_DIR WORK=/path/eval bash ablation/dataset/eval.sh

# held-out domains: 8 runs, then CPR-eval per domain
DATA=/path/folds OUT=/path/domain_runs bash ablation/domain/run.sh
RUNS=/path/domain_runs DATASET_RUNS=/path/runs EVAL_DIR=$EVAL_DIR WORK=/path/eval bash ablation/domain/eval.sh
```

The LLM reranker, query rewriting, leave-one-out pooling, and the dataset construction pipeline are run from their folders (see their
READMEs). No fine-tuned checkpoints are distributed: the ablation evaluations evaluate the runs you train with `ablation/*/run.sh`.

All models are loaded at pinned Hugging Face revisions (`evaluation/main/models.json`, `ablation/*/models.json`). The code was checked
against the runs behind the paper: re-scoring the paper's outputs gives the paper's values, and the training launchers and data
builders give the same arguments and data checksums.

## License

The data are released under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/), the license of the Semantic Scholar Academic
Graph from which they are built. Included abstracts keep their original licenses (CC-BY, CC0, or public domain); the source paper of
each abstract is identified by its corpus id and DOI.

The code is released under the [Apache License 2.0](LICENSE). `training/nemotron/finetune.py` and `training/nemotron/train.yaml` are
adapted from NeMo AutoModel (Apache License 2.0); the model weights they start from (`nvidia/llama-embed-nemotron-8b`) are not
included and keep their own license.

## Citation

Citation information will be added after the anonymous review period.
