# llama-embed-nemotron-8b fine-tuning (NeMo AutoModel)

Full fine-tuning of `nvidia/llama-embed-nemotron-8b` (revision `aa3b43a4`; Llama-3.1-8B with bidirectional attention, mean pooling,
4,096 dimensions) with NeMo AutoModel, following its official bi-encoder recipe
(`examples/retrieval/bi_encoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml` at commit `8e562fce`, NeMo AutoModel 0.8.0).
Used for the dataset ablation (`ablation/dataset`, base name `nemotron8b`).

## Settings (`train.yaml`)

| | value | official recipe |
|---|---|---|
| start model | `nvidia/llama-embed-nemotron-8b` @ `aa3b43a4` | `meta-llama/Llama-3.1-8B` |
| batch | 128 queries per step = 8 GPUs x 1 x gradient accumulation 16 | same |
| steps | 1 epoch; 10,573 rows on 8 GPUs = 83 steps (each GPU gets 1,322 rows, the last accumulation group is a step) | 1 epoch |
| learning rate | 2e-6, 2 warmup steps, cosine to 1% of the learning rate | same |
| optimizer | AdamW (torch, fused), betas 0.9 / 0.999, eps 1e-8, weight decay 0.01, fp32 states; gradient clipping 1.0 | TransformerEngine FusedAdam with fp32 master weights |
| loss | InfoNCE, temperature 0.02 | same |
| negatives | every hard negative of the row (`n_passages` = negatives + 1: 10 / 2 / 1) + in-batch negatives gathered across the 8 GPUs | 4 hard negatives, no cross-GPU in-batch negatives |
| max length | query 1,024 / document 1,024 tokens (collator truncation) | 512 / 512 |
| prompts | query `Query:` + text (no space, written into the data by `prepare_data.py`), documents none | dataset instructions |
| precision | fp32 weights, bf16 computation (FSDP2 mixed precision) | bf16 weights + fp32 master weights |
| other | FSDP2, activation checkpointing, flash_attention_2; seed 42 (dataset shuffle seed 42); no validation | seed 125, validation |
| checkpoints | every 50 steps and the final step (`checkpoints/epoch_0_step_82`), sharded; `export.sh` consolidates the final one | every 3,000 steps, consolidated |

## Setup

```bash
bash training/nemotron/setup_env.sh        # -> training/.venv-nemotron      (NeMo AutoModel @ 8e562fce, torch 2.11.0 cu128, transformers 5.18.0, flash-attn 2.8.3)
bash training/nemotron/setup_env_eval.sh   # -> training/.venv-nemotron-eval (transformers 4.51.0, sentence-transformers 5.1.2, flash-attn 2.8.3)
```

Both install the pinned versions in `requirements.lock` / `requirements-eval.lock` (`--no-deps`); flash-attn is built from source
(`SKIP_FLASH_ATTN=1` skips it). The training environment keeps a NeMo AutoModel clone at the pinned commit in `<venv>/src/Automodel`
(`export.sh` uses its `tools/offline_hf_consolidation.py`). The evaluation environment is the one the model's official remote code needs;
the encoders in `evaluation/` use it for every llama-embed-nemotron-8b model. `HF_TOKEN` / optional `WANDB_API_KEY` are read from the
environment or `training/.env`.

## Run

```bash
. training/.venv-nemotron/bin/activate
TRAIN_DATA=/abs/train.jsonl OUTPUT_DIR=/local/runs/exp NPROC=8 bash training/nemotron/train.sh     # MIN_NEG=0 for in-batch-only data
bash training/nemotron/export.sh /local/runs/exp            # -> /local/runs/exp/eval_model (sentence-transformers folder, bf16)
```

- `TRAIN_DATA`: our format (`{"query", "pos": [1], "neg": [...]}`, e.g. `train_texts.jsonl` of the CPR-train release), texts **not**
  cut (no `make_maxlen.py`); every row has the same number of hard negatives. `train.sh` checks it (`../common/validate_data.py`), converts it (`prepare_data.py` -> `OUTPUT_DIR/nemo_train.jsonl`),
  links the base model without its `auto_map` (`prepare_base.py` -> `OUTPUT_DIR/base_model`, so NeMo uses its own `llama_bidirec` class),
  writes the filled config `OUTPUT_DIR/train.yaml` and launches `torchrun finetune.py` (the official entry point plus the `llama_bidirec`
  registration). Extra arguments are NeMo `--dotted.key value` overrides.
- `export.sh` consolidates the last checkpoint to bf16 safetensors on CPU and assembles `eval_model/`: the base repository's config (with
  `auto_map`), remote code, sentence-transformers files and tokenizer + the trained weights (tensor names checked against the base model).
  The encoders load it with `trust_remote_code` and pass the query prompt `Query:` themselves (the folder keeps the base model's default
  instruction in `config_sentence_transformers.json`, which is not used).
- Hardware: 8 x A100 80GB (at most about 53 GiB per GPU with every text at 1,024 tokens). A run took 50-90 minutes including two
  checkpoint saves of about 105 GB each (model + optimizer); put `OUTPUT_DIR` on a local disk.

## Reproducibility

- `train.yaml` filled in by `train.sh` gives the same configuration (all keys and values except the local paths) as each of the 8
  dataset-ablation runs behind the paper; W&B was off in those runs.
- `prepare_data.py` on CPR-train (`train_texts.jsonl` of the release after `dataset/release/fetch_abstracts.py`, sha256 `d008ed77...`)
  gives the converted file of the paper run
  (sha256 `aed00206f6a6a19a29c69746c06cd0d25ddecd7378ae9578fe4cc2cccc16911d`).
- Versions known from the paper runs: NeMo AutoModel `8e562fce`, torch 2.11.0+cu128, transformers 5.18.0, flash-attn 2.8.3, einops 0.8.2
  (training); transformers 4.51.0, sentence-transformers 5.1.2, accelerate 1.6.0, torch 2.11.0+cu128, flash-attn 2.8.3 (evaluation). The other
  packages in the lock files are resolved as of 2026-10-12 (`--exclude-newer`), the date of the runs.
- Liger kernels were not installed in the paper runs (NeMo falls back to the standard kernels); the lock does not include them.
