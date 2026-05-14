"""
Skip policies for Phase B compute-skipping experiments.

Each policy takes a per-token cosine similarity tensor [B, L] and
returns a boolean mask [B, L] where True means "skip this token at
this layer" (i.e., reuse the previous step's output).
"""

from __future__ import annotations
from typing import Optional
import torch


class SkipPolicy:
    """Abstract base."""
    name: str = "abstract"

    def compute_mask(self, sim: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"


class NoSkipPolicy(SkipPolicy):
    name = "baseline"
    def compute_mask(self, sim):
        return torch.zeros_like(sim, dtype=torch.bool)


class TokenCossimPolicy(SkipPolicy):
    def __init__(self, threshold: float):
        assert 0.0 < threshold < 1.0
        self.threshold = float(threshold)
        self.name = f"token_cossim_{threshold:g}"

    def compute_mask(self, sim):
        return sim > self.threshold


class TokenTopKPolicy(SkipPolicy):
    """Keep top-k% most-changed (lowest-sim); skip the rest."""
    def __init__(self, k_percent: float):
        assert 0 < k_percent < 100
        self.k_percent = float(k_percent)
        self.name = f"token_topk_{int(k_percent)}"

    def compute_mask(self, sim):
        # k=25 means: keep the 25% lowest-sim (most-changed) tokens, skip the rest 75%.
        # → threshold = k-th percentile of sim;  skip = (sim > threshold).
        q = self.k_percent / 100.0
        sim_f = sim.float() if sim.dtype != torch.float32 else sim
        thr = torch.quantile(sim_f, q=q, dim=-1, keepdim=True)
        return sim_f > thr


class LayerCossimPolicy(SkipPolicy):
    """Aggregate per-token sims (avg or max), skip whole layer if > threshold."""
    def __init__(self, threshold: float, agg: str = "avg"):
        assert 0.0 < threshold < 1.0
        assert agg in ("avg", "max")
        self.threshold = float(threshold)
        self.agg = agg
        self.name = f"layer_cossim_{agg}_{threshold:g}"

    def compute_mask(self, sim):
        if self.agg == "avg":
            agg_sim = sim.mean(dim=-1)
        else:
            agg_sim = sim.max(dim=-1).values
        skip_whole = agg_sim > self.threshold  # [B]
        return skip_whole.unsqueeze(-1).expand_as(sim)


def make_policy(spec: dict) -> SkipPolicy:
    n = spec['name']
    if n in ('baseline', 'none'):
        return NoSkipPolicy()
    if n == 'token_cossim':
        return TokenCossimPolicy(threshold=spec['threshold'])
    if n == 'token_topk':
        return TokenTopKPolicy(k_percent=spec['k'])
    if n == 'layer_cossim':
        return LayerCossimPolicy(threshold=spec['threshold'],
                                  agg=spec.get('agg', 'avg'))
    raise ValueError(f"Unknown policy name: {n!r}")


ALL_SETTINGS = (
    [{'name': 'baseline'}]
    + [{'name': 'token_cossim', 'threshold': t}
       for t in (0.995, 0.99, 0.98, 0.97, 0.96)]
    + [{'name': 'token_topk', 'k': k}
       for k in (25, 50)]
    + [{'name': 'layer_cossim', 'threshold': t, 'agg': 'avg'}
       for t in (0.999, 0.995, 0.99, 0.98, 0.97)]
    + [{'name': 'layer_cossim', 'threshold': t, 'agg': 'max'}
       for t in (0.999, 0.995, 0.99, 0.98, 0.97)]
)
assert len(ALL_SETTINGS) == 18


if __name__ == "__main__":
    print("Sanity-checking all 18 settings...")
    torch.manual_seed(42)
    sim = 0.5 + 0.5 * torch.rand(2, 32)
    for spec in ALL_SETTINGS:
        policy = make_policy(spec)
        mask = policy.compute_mask(sim)
        assert mask.shape == sim.shape
        assert mask.dtype == torch.bool
        skip_rate = mask.float().mean().item()
        print(f"  {policy.name:35s}  skip_rate={skip_rate:.3f}")
    print(f"\nTotal: {len(ALL_SETTINGS)} settings, all OK.")
