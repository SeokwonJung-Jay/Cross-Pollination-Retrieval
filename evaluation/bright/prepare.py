#!/usr/bin/env python3
"""Download BRIGHT (xlangai/BRIGHT at the pinned REV, configs examples + documents) as one parquet per split: DIR/examples_<split>.parquet, DIR/documents_<split>.parquet.
  python bright/prepare.py --out DIR [--splits biology economics ...]   (default: the 7 StackExchange splits + theoremqa_theorems + pony)"""
import argparse
from pathlib import Path
from datasets import load_dataset
REV = '3066d29c9651a576c8aba4832d249807b181ecae'   # xlangai/BRIGHT main (the data of the paper's evaluation)
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
ap.add_argument('--splits', nargs='+', default=['biology', 'earth_science', 'economics', 'psychology', 'robotics', 'stackoverflow', 'sustainable_living', 'theoremqa_theorems', 'pony'])
a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
for cfg in ('examples', 'documents'):
    for s in a.splits:
        f = O / f'{cfg}_{s}.parquet'
        if f.exists(): print(cfg, s, 'exists'); continue
        ds = load_dataset('xlangai/BRIGHT', cfg, split=s, revision=REV); ds.to_parquet(str(f)); print(cfg, s, len(ds), flush=True)
