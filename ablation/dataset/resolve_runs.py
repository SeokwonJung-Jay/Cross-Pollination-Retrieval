#!/usr/bin/env python3
"""Registry with the fine-tuned models resolved to your own training runs (no fine-tuned checkpoints are distributed).
A registry entry with "run": "$RUNS/<base>/<neg>/<set>" (environment variables expanded; the output layout of run.sh) is resolved to
  - llama-embed-nemotron-8b ("remote_code": true): <run>/eval_model (training/nemotron/export.sh)
  - Qwen3-Embedding: the last <run>/checkpoint-<N>, finalized for sentence-transformers by training/qwen/finalize.py
and gets "hf" = that directory, "revision" = null. An entry whose run variable is unset, whose run has no DONE marker (run.sh writes
it after the checkpoint is finalized / exported) or whose checkpoint is not found is left out (reported on stderr), so the evaluation
skips it; entries without "run" (the base models, pinned HF revisions) are kept as they are.
Writes the resolved registry to OUT (same keys as REGISTRY minus the skipped ones) and prints one line per kept model: name hf.
  python -I resolve_runs.py REGISTRY OUT"""
import json, os, re, sys
from pathlib import Path
reg, out = sys.argv[1:3]; REG = json.load(open(reg))
def resolve(e):
    run = e['run']; var = re.match(r'\$\{?(\w+)', run)
    if var and not os.environ.get(var.group(1)): return None, f'{var.group(1)} not set'
    d = Path(os.path.expandvars(run))
    if not (d / 'DONE').exists(): return None, f'run not found or not finished ({d}/DONE missing)'
    if e.get('remote_code'): ck = d / 'eval_model'
    else:
        cks = sorted(d.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1)
        ck = cks[-1] if cks else d / 'checkpoint-<N>'
    if not (ck / 'config_sentence_transformers.json').exists(): return None, f'no finalized checkpoint at {ck}'
    return str(ck.resolve()), None
R = {k: v for k, v in REG.items() if k.startswith('_')}
for n, e in REG.items():
    if n.startswith('_'): continue
    if 'run' in e:
        ck, why = resolve(e)
        if ck is None: print(f'skip {n}: {why}', file=sys.stderr); continue
        e = {**e, 'hf': ck, 'revision': None}
    R[n] = e; print(n, e['hf'])
Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(R, open(out, 'w'), indent=1)
