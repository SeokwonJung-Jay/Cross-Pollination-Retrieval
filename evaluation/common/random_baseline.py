"""Exact random-ranking baseline for a pool of N docs with g golds (Q-level, any gold): expected Recall@k and MRR, and pool size.
Recorded next to every retrieval result."""
from math import comb
def random_metrics(N, g, ks=(1, 5, 10)):
    if g <= 0 or N <= 0: return {**{f'recall{k}': 0.0 for k in ks}, 'mrr': 0.0}
    out = {f'recall{k}': (1 - comb(N - g, k) / comb(N, k)) if k < N else 1.0 for k in ks}
    # P(first gold at rank r) = C(N-r, g-1) / C(N, g)
    out['mrr'] = sum(comb(N - r, g - 1) / comb(N, g) / r for r in range(1, N - g + 2))
    return out
def summarize(pools):
    """pools: list of (N, g) per Q -> macro means + median pool size"""
    import statistics
    ms = [random_metrics(N, g) for N, g in pools]
    s = {k: round(statistics.mean(m[k] for m in ms), 4) for k in ms[0]}
    s['pool_median'] = statistics.median(N for N, _ in pools); return s
