#!/usr/bin/env python3
# Adapted from NeMo AutoModel examples/retrieval/bi_encoder/finetune.py (Apache License 2.0); modified: one added import.
"""Entry point launched by train.sh (torchrun): NeMo AutoModel's examples/retrieval/bi_encoder/finetune.py (@8e562fce) plus one import.
The import registers NeMo's native llama_bidirec model with the HF Auto classes before the recipe reads the checkpoint config
(nvidia/llama-embed-nemotron-8b has model_type llama_bidirec; the official recipe starts from meta-llama/Llama-3.1-8B and does not need it).
  torchrun --nproc-per-node 8 finetune.py --config OUTPUT_DIR/train.yaml [--dotted.key value ...]"""
import nemo_automodel.components.models.llama_bidirectional.model  # noqa: F401  registers llama_bidirec with the HF Auto classes
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.retrieval import TrainBiEncoderRecipe

if __name__ == '__main__':
    cfg = parse_args_and_load_config()
    recipe = TrainBiEncoderRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()
