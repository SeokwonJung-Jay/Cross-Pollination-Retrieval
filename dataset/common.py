"""Shared helpers for the CPR-Dataset pipeline: run configs, data locations, OpenAI calls with spend caps and cost ledgers.

Config files (config/eval.json, config/train.json) are JSON; every string may use ${ENV} variables (expanded here, an unset variable
is an error). Data locations come from the environment (or a .env file next to the repo root / in the working directory):
  S2AG      Semantic Scholar Academic Graph release dir with papers/, abstracts/, citations/, publication-venues/ (*.gz)
  S2ORC     Semantic Scholar s2orc_v2 release dir (*.gz)
  OPENALEX  OpenAlex column extract written by corpus/openalex_extract.py (works_cols/, topics_catalog.json)
  DATA      derived data (venue table, S2 indexes, universes, vectors, Stage 2 caches)
  WORK      per-run outputs (one dir per config)
Secrets: OPENAI_API_KEY (Stages 3-7), S2_API_KEY (optional, Semantic Scholar Graph API), HF_TOKEN (optional)."""
import json, os, re, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / 'prompts'


def load_env():
    """Read KEY=VALUE lines from .env (repo root, then the working directory) into os.environ without overriding set values."""
    for f in (HERE.parent / '.env', Path.cwd() / '.env'):
        if f.is_file():
            for line in f.read_text().splitlines():
                m = re.match(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$', line)
                if m and not line.lstrip().startswith('#'):
                    os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


load_env()


def env(name):
    v = os.environ.get(name)
    if not v: sys.exit(f'set {name} (see dataset/README.md)')
    return Path(v)


def _expand(x):
    if isinstance(x, str):
        for k in re.findall(r'\$\{(\w+)\}', x):
            if not os.environ.get(k): sys.exit(f'config uses ${{{k}}}; set {k}')
        return os.path.expandvars(x)
    if isinstance(x, dict): return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list): return [_expand(v) for v in x]
    return x


def load_config(path):
    """Config dict with ${ENV} expanded; `work` is created."""
    C = _expand(json.load(open(path)))
    Path(C['work']).mkdir(parents=True, exist_ok=True)
    return C


def prompt(name):
    return (PROMPTS / f'{name}.txt').read_text()


# ------------------------------------------------------------------ OpenAI calls (Stages 5b, 7b, 7)
# USD per million tokens (input, cached input, output); used for the spend caps and the ledgers only
PRICE = {'gpt-5.6-sol': (2.0, 0.2, 10.0), 'gpt-6.1-sol': (1.0, 0.1, 5.0), 'gpt-6.1-sol@standard': (2.0, 0.2, 10.0),
         'gpt-6-astra': (10.0, 1.0, 50.0)}
NOT_SERVED = ('429', 'rate limit', 'rate_limit', '529', 'overloaded', '500 ', '502', '503', '504', 'internal server error',
              'service unavailable', 'bad gateway', 'connection', 'timed out', 'timeout')


class Spend:
    """Running USD total with a cap; every billed call is appended to the ledger (jsonl) with a stage tag."""
    def __init__(self, cap, ledger=None, tag=''):
        self.cap, self.usd, self.lock, self.ledger, self.tag = cap, 0.0, threading.Lock(), ledger, tag

    def add(self, model, tin, tcache, tout, note=''):
        p = PRICE[model]; usd = ((tin - tcache) * p[0] + tcache * p[1] + tout * p[2]) / 1e6
        with self.lock:
            self.usd += usd
            if self.ledger:
                with open(self.ledger, 'a') as f:
                    f.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tag': self.tag, 'model': model, 'in': tin, 'cache': tcache,
                                        'out': tout, 'usd': round(usd, 6), 'note': note}) + '\n')

    def ok(self):
        with self.lock: return self.usd < self.cap


def ledger_total(ledger, tag=None):
    """USD already spent according to a ledger (optionally one stage tag), so caps are cumulative over reruns."""
    if not Path(ledger).exists(): return 0.0
    return sum(j.get('usd', 0) for j in map(json.loads, open(ledger)) if tag is None or j.get('tag') == tag)


def not_served(e):
    m = f'{type(e).__name__} {e}'.lower(); return any(x in m for x in NOT_SERVED)


_oa = threading.local()


def openai_call(model, text, spend, reasoning=None, flex=False, max_tokens=4000):
    """One Responses API call; retries only requests that were not served (waits 30/60/90 s). Returns (output text, usage)."""
    from openai import OpenAI
    if not hasattr(_oa, 'c'): _oa.c = OpenAI(timeout=900, max_retries=0)
    kw = {'model': model, 'input': text, 'max_output_tokens': max_tokens}
    if reasoning: kw['reasoning'] = {'effort': reasoning}
    if flex: kw['service_tier'] = 'flex'
    for k in range(4):
        if not spend.ok(): raise RuntimeError('spend cap reached')
        try:
            r = _oa.c.responses.create(**kw); u = r.usage
            cached = getattr(getattr(u, 'input_tokens_details', None), 'cached_tokens', 0) or 0
            spend.add(model if flex else (model + '@standard' if model + '@standard' in PRICE else model), u.input_tokens, cached, u.output_tokens,
                      note='flex' if flex else 'standard')
            return r.output_text or '', {'in': u.input_tokens, 'cache': cached, 'out': u.output_tokens}
        except Exception as e:
            if k < 3 and not_served(e): time.sleep(30 * (k + 1)); continue
            raise


def parse_obj(t):
    return json.loads(t[t.find('{'):t.rfind('}') + 1])


def ask_json(model, text, spend, **kw):
    """openai_call + JSON parse; one re-ask when a served reply is not JSON, then None."""
    for k in range(2):
        t, u = openai_call(model, text, spend, **kw)
        try: return parse_obj(t), t, u
        except Exception:
            if k == 1: return None, t, u


def final_score(text):
    """UMBRELA-style judge output: the last 'final score: n' (n in 0-3), else None."""
    m = re.findall(r'final score[^0-9]{0,6}([0-3])', text or '', re.I)
    return int(m[-1]) if m else None


def run_items(stage, items, fn, workers, outdir):
    """fn(item) -> record dict, saved as outdir/<id>.json; items already saved are skipped (resumable). Returns (todo, saved, failed, secs)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (outdir / f"{it['id']}.json").exists()]
    print(f'[{stage}] items {len(items)} | to do {len(todo)} | workers {workers}', flush=True)
    done, failed, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in todo}
        for f in as_completed(futs):
            it = futs[f]
            try:
                rec = f.result()
                if rec is not None:
                    tmp = outdir / f"{it['id']}.tmp"; tmp.write_text(json.dumps(rec, ensure_ascii=False)); tmp.replace(outdir / f"{it['id']}.json")
                    done += 1
            except Exception as e:
                failed += 1; print(f"[{stage}] {it['id']} FAILED (not retried): {type(e).__name__}: {str(e)[:200]}", flush=True)
            if (done + failed) % 50 == 0: print(f'[{stage}] {done + failed}/{len(todo)} | {time.time() - t0:.0f}s', flush=True)
    return len(todo), done, failed, time.time() - t0


def records(d, prefix=''):
    """{stem: record} for every <prefix>*.json in d (files starting with '_' and summary.json are skipped)."""
    return {f.stem: json.load(open(f)) for f in Path(d).glob(f'{prefix}*.json') if not f.name.startswith('_') and f.name != 'summary.json'}


def topic_catalog():
    """OpenAlex topic catalog ($OPENALEX/topics_catalog.json, corpus/openalex_extract.py topics): {short topic id 'T123': entry}."""
    return {x['id'].rsplit('/', 1)[-1]: x for x in json.load(open(env('OPENALEX') / 'topics_catalog.json'))}


def topic_field_rows():
    """[(topic URL as in works_cols, OpenAlex field name)]: the field of a paper = the field of its rank-1 topic."""
    return [('https://openalex.org/' + t, x['field']['display_name']) for t, x in topic_catalog().items()]


def base_papers(C):
    """{field: [q corpusid, ...]} of the run: config q_pool (sample_base_papers.py) in pool order without the ids listed in config
    exclude_base_papers (a JSON list), the first q_per_field per field if set."""
    sel = json.load(open(C['q_pool']))['per_field']; k = C.get('q_per_field')
    ex = {int(c) for c in json.load(open(C['exclude_base_papers']))} if C.get('exclude_base_papers') else set()
    sel = {f: [int(c) for c in v if int(c) not in ex] for f, v in sel.items()}
    return {f: v[:k] if k else v for f, v in sel.items()}


def stage2_dirs(C):
    """(Stage 2 input dir of the run, shared cache dir with jats_cache/ s2orc_q/ fulltext_cache/)"""
    i, c = Path(C['work']) / 'stage2_input', Path(C['stage2_cache'])
    for d in (i, c / 'jats_cache', c / 's2orc_q', c / 'fulltext_cache'): d.mkdir(parents=True, exist_ok=True)
    return i, c
