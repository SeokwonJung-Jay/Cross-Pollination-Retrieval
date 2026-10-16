#!/usr/bin/env python3
"""Base model folder for NeMo training: nvidia/llama-embed-nemotron-8b at the pinned revision (downloaded to the HF cache), every file
linked from the snapshot except config.json, which is written without "auto_map" so that NeMo loads its own llama_bidirec class
(same weights; the remote code in auto_map needs transformers 4.51, the training environment has 5.18). Prints the snapshot directory
(the tokenizer path of train.yaml and the file source of export.sh) as the last line.
  python -I prepare_base.py BASE_DIR       (BASE_DIR is created / refreshed)
  python -I prepare_base.py --snapshot     (only download and print the snapshot directory)"""
import json, os, sys
from huggingface_hub import snapshot_download
REPO, REV = 'nvidia/llama-embed-nemotron-8b', 'aa3b43a495a9b280d1bdb716da37c54bb495d630'
snap = snapshot_download(REPO, revision=REV)
if sys.argv[1] != '--snapshot':
    base = os.path.abspath(sys.argv[1])
    for root, _, files in os.walk(snap):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), snap); dst = os.path.join(base, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.lexists(dst): os.remove(dst)
            if rel == 'config.json':
                cfg = json.load(open(os.path.join(snap, rel))); cfg.pop('auto_map', None); json.dump(cfg, open(dst, 'w'), indent=2)
            else: os.symlink(os.path.realpath(os.path.join(snap, rel)), dst)
    print('base model', base, file=sys.stderr)
print(snap)
