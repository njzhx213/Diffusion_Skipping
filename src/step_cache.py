"""
StepCacheManager + hook attachment.

This is the central abstraction for our diffusion-skipping project.
V1 (this file): logging-only mode (policy=None).
  - Captures H_in / attn_out / H_mid / mlp_out / temp at each denoising step.
  - Computes per-token cosine similarity vs. previous step.
  - Accumulates sim data in memory; dump to .npz at end of sample.

Later: add token-skip / layer-skip policies into the same hooks.
"""
from __future__ import annotations

import os
from policies import SkipPolicy, NoSkipPolicy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Per-step, per-layer sim records (one entry per layer per step)."""
    sample_id: int
    step_id: int                  # global step counter within a sample
    block_id: int
    is_first_step_in_block: bool
    layer_id: int
    # Per-token cosine similarity (vs prev step), shape: [B, L]
    sim_H_in: Optional[np.ndarray] = None     # used for token/layer skip decision
    sim_H_mid: Optional[np.ndarray] = None    # FFN motivation
    sim_attn_out: Optional[np.ndarray] = None # column motivation (attn)
    sim_mlp_out: Optional[np.ndarray] = None  # column motivation (mlp)
    sim_temp: Optional[np.ndarray] = None     # column motivation (FFN temp)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class StepCacheManager:
    """
    Manages per-step caches and similarity logging for one diffusion-LM run.

    Lifecycle (called by generation_functions.batch_sample):
      manager.start_new_sample(sample_id)
        manager.start_new_block(block_id)
          manager.start_new_step()   # before each forward
          (model forward; hooks fire)
          manager.finalize_step()    # after each forward
        ...
      manager.finalize_sample()  # dumps .npz
    """

    def __init__(
        self,
        num_layers: int = 28,
        policy=None,
        threshold: Optional[float] = None,
        dump_dir: Optional[str] = None,
    ):
        self.num_layers = num_layers
        # Phase B: policy is now a SkipPolicy instance.
        # Strings, None: fall back to NoSkipPolicy (Phase A behavior).
        if policy is None or isinstance(policy, str):
            self.policy = NoSkipPolicy()
        else:
            self.policy = policy
        self.threshold = threshold
        self.dump_dir = dump_dir
        # Phase B: skip-policy state
        self.skip_mask: Dict[int, Optional[torch.Tensor]] = {}
        self.prev_attn_out_cache: Dict[int, torch.Tensor] = {}
        self.prev_mlp_out_cache: Dict[int, torch.Tensor] = {}
        self.skipped_per_step_layer: Dict[Tuple[int, int], int] = {}
        self.total_steps_this_sample: int = 0
        self.is_first_step_in_block: bool = False

        # === per-sample state ===
        self.sample_id: int = -1
        self.records: List[StepRecord] = []
        # global step counter within current sample (resets per sample)
        self.step_id_global: int = -1

        # === per-block state ===
        self.block_id: int = -1
        self.local_step_in_block: int = -1

        # === per-step state ===
        # buffers filled by hooks during current forward
        # key: (layer_id, name) where name in {H_in, attn_out, H_mid, mlp_out, temp}
        self.this_step: Dict[Tuple[int, str], torch.Tensor] = {}
        self.prev_step: Dict[Tuple[int, str], torch.Tensor] = {}

    # -------------------- Sample / Block / Step lifecycle --------------------

    def start_new_sample(self, sample_id: int):
        self.sample_id = sample_id
        self.records = []
        # Phase B: reset per-sample skip state
        self.skip_mask = {}
        self.prev_attn_out_cache = {}
        self.prev_mlp_out_cache = {}
        self.skipped_per_step_layer = {}
        self.total_steps_this_sample = 0
        self.is_first_step_in_block = False
        self.step_id_global = -1
        self.block_id = -1
        self.this_step.clear()
        self.prev_step.clear()

    def start_new_block(self, block_id: int):
        self.block_id = block_id
        self.local_step_in_block = -1
        # cross-block: clear prev. (Per task spec: each block's first step
        # must not be skipped; clearing prev guarantees this naturally.)
        self.prev_step.clear()
        # Phase B: block boundary invalidates reuse cache; mark first step.
        self.is_first_step_in_block = True
        self.prev_attn_out_cache = {}
        self.prev_mlp_out_cache = {}
        self.skip_mask = {}

    def start_new_step(self):
        """Call before each forward pass."""
        self.step_id_global += 1
        self.local_step_in_block += 1
        self.this_step.clear()  # ready to receive hook outputs
        # Phase B: per-step tracking
        self.total_steps_this_sample += 1
        self.skip_mask = {}

    def finalize_step(self):
        """Call after each forward pass.

        Computes sim vs prev_step (if exists), creates StepRecord(s),
        and rolls this_step -> prev_step.
        """
        is_first = (self.local_step_in_block == 0)

        if not is_first and len(self.prev_step) > 0:
            # compute sim for each layer
            for layer_id in range(self.num_layers):
                rec = StepRecord(
                    sample_id=self.sample_id,
                    step_id=self.step_id_global,
                    block_id=self.block_id,
                    is_first_step_in_block=is_first,
                    layer_id=layer_id,
                )
                for name in ['H_in', 'H_mid', 'attn_out', 'mlp_out', 'temp']:
                    k = (layer_id, name)
                    if k in self.this_step and k in self.prev_step:
                        sim = self._cosine_sim(self.this_step[k], self.prev_step[k])
                        setattr(rec, f'sim_{name}', sim)
                self.records.append(rec)

        # roll: this -> prev (transfer ownership, then clear this)
        self.prev_step = self.this_step
        self.this_step = {}
        # Phase B: after first step of a block completes, subsequent steps may skip
        self.is_first_step_in_block = False

    def finalize_sample(self):
        """Call after a sample finishes. Dumps records to .npz if dump_dir set."""
        if self.dump_dir is None or len(self.records) == 0:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f'sample_{self.sample_id:04d}.npz')

        # gather into compact arrays for storage
        N = len(self.records)
        meta = {
            'sample_id': np.array([r.sample_id for r in self.records], dtype=np.int32),
            'step_id': np.array([r.step_id for r in self.records], dtype=np.int32),
            'block_id': np.array([r.block_id for r in self.records], dtype=np.int32),
            'layer_id': np.array([r.layer_id for r in self.records], dtype=np.int32),
            'is_first_step': np.array(
                [r.is_first_step_in_block for r in self.records], dtype=bool),
        }
        # Pack per-record sim arrays: each is a [B, L] array but L can vary
        # by step. We store as a python object array of np.ndarrays; numpy
        # handles this transparently with allow_pickle.
        sim_arrays = {}
        for name in ['H_in', 'H_mid', 'attn_out', 'mlp_out', 'temp']:
            sim_arrays[f'sim_{name}'] = np.array(
                [getattr(r, f'sim_{name}') for r in self.records],
                dtype=object,
            )

        np.savez_compressed(path, **meta, **sim_arrays)

    # -------------------- helpers --------------------

    @staticmethod
    def _cosine_sim(this_t: torch.Tensor, prev_t: torch.Tensor) -> np.ndarray:
        """Per-token cosine similarity, return shape [B, L] as fp32 numpy."""
        # bf16 -> fp32 for numerical stability in cosine_similarity
        if this_t.shape != prev_t.shape:
            # length may differ if sub-block forward shape varies; align by
            # taking the overlap on the last-but-one dim
            min_len = min(this_t.shape[-2], prev_t.shape[-2])
            this_t = this_t[..., -min_len:, :]
            prev_t = prev_t[..., -min_len:, :]
        sim = F.cosine_similarity(this_t.float(), prev_t.float(), dim=-1)  # [B, L]
        return sim.detach().cpu().numpy().astype(np.float32)

    # -------------------- hook target API --------------------

    def record_tensor(self, layer_id: int, name: str, tensor: torch.Tensor):
        """Called by hooks to deposit a tensor for the current step."""
        # detach + keep on GPU (cheap); we only convert to numpy in
        # finalize_step, and only the sim result (small).
        self.this_step[(layer_id, name)] = tensor.detach()
        # Phase B: when H_in arrives, decide skip mask for this layer
        # (in time for the attn / mlp hooks to read it).
        if name == 'H_in':
            self._decide_skip_mask_for_layer(layer_id, tensor)

    def _decide_skip_mask_for_layer(self, layer_id: int, h_in: torch.Tensor):
        """Phase B: compute skip mask for this (step, layer) from H_in vs prev H_in."""
        prev = self.prev_step.get((layer_id, 'H_in'))
        if prev is None or prev.shape != h_in.shape:
            # First step or shape mismatch: cannot decide; no skip.
            self.skip_mask[layer_id] = None
            return
        sim = F.cosine_similarity(h_in.float(), prev.float(), dim=-1)  # [B, L]
        if self.is_first_step_in_block:
            mask = torch.zeros_like(sim, dtype=torch.bool)
        else:
            mask = self.policy.compute_mask(sim)
        self.skip_mask[layer_id] = mask
        # Count skipped tokens for this (step, layer); used for FLOPs accounting.
        self.skipped_per_step_layer[(self.step_id_global, layer_id)] = int(mask.sum().item())

    def _maybe_substitute_output(self, layer_id: int, name: str, output: torch.Tensor):
        """
        Phase B: called from attn/mlp hooks. If we have a valid mask and
        a previous-step cache of compatible shape, blend the previous output
        in at masked positions. Otherwise return output unchanged.
        Always update the cache (so next step can reuse the latest state).
        """
        mask = self.skip_mask.get(layer_id)
        cache_dict = (self.prev_attn_out_cache if name == 'attn_out'
                      else self.prev_mlp_out_cache)
        prev_cache = cache_dict.get(layer_id)
        if mask is None or prev_cache is None or not mask.any():
            cache_dict[layer_id] = output.detach()
            return output
        if prev_cache.shape != output.shape:
            cache_dict[layer_id] = output.detach()
            return output
        # Blend: at mask=True positions, take prev; otherwise current
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1] broadcastable to [B, L, H]
        new_out = torch.where(mask_expanded, prev_cache, output)
        cache_dict[layer_id] = new_out.detach()
        return new_out

    def dump_skip_stats(self):
        """Phase B: dump per-sample skip statistics."""
        if self.dump_dir is None:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        items = sorted(self.skipped_per_step_layer.items())
        if items:
            step_ids = np.array([k[0] for k, _ in items], dtype=np.int32)
            layer_ids = np.array([k[1] for k, _ in items], dtype=np.int32)
            skipped_counts = np.array([v for _, v in items], dtype=np.int32)
        else:
            step_ids = np.zeros(0, dtype=np.int32)
            layer_ids = np.zeros(0, dtype=np.int32)
            skipped_counts = np.zeros(0, dtype=np.int32)
        path = os.path.join(self.dump_dir, f'skip_stats_{self.sample_id:04d}.npz')
        # Phase B note: per task spec 3.b, we log per (step, layer) the
        # number of tokens NOT reused (i.e., extra FLOPs). We dump both
        # reused_counts (mask=True) and not_reused_counts for clarity.
        np.savez_compressed(
            path,
            sample_id=np.array([self.sample_id], dtype=np.int32),
            total_steps=np.array([self.total_steps_this_sample], dtype=np.int32),
            step_ids=step_ids,
            layer_ids=layer_ids,
            reused_counts=skipped_counts,
            # not_reused_counts is the complement; we leave the per-record
            # block size to be filled by the aggregator since L can vary.
        )


# ---------------------------------------------------------------------------
# Hook attachment
# ---------------------------------------------------------------------------

def attach_hooks(model, manager: StepCacheManager) -> List:
    """
    Attach all logging hooks to a Fast_dLLM_QwenForCausalLM model.
    Returns a list of hook handles (caller can .remove() them).

    Convention: model.model.layers is the list of DecoderLayers.
    """
    handles = []
    layers = model.model.layers
    assert len(layers) == manager.num_layers, (
        f"manager.num_layers={manager.num_layers} but model has {len(layers)} layers")

    for layer_idx, layer in enumerate(layers):
        handles.extend(_attach_layer_hooks(layer, layer_idx, manager))
    return handles


def _attach_layer_hooks(layer, layer_idx: int, manager: StepCacheManager) -> List:
    """Attach 5 hooks to one decoder layer."""
    handles = []

    # 1) H_in: output of input_layernorm
    handles.append(layer.input_layernorm.register_forward_hook(
        _make_post_hook(manager, layer_idx, 'H_in')))

    # 2) attn_out: output of self_attn
    handles.append(layer.self_attn.register_forward_hook(
        _make_post_hook(manager, layer_idx, 'attn_out')))

    # 3) H_mid: output of post_attention_layernorm
    handles.append(layer.post_attention_layernorm.register_forward_hook(
        _make_post_hook(manager, layer_idx, 'H_mid')))

    # 4) mlp_out: output of mlp
    handles.append(layer.mlp.register_forward_hook(
        _make_post_hook(manager, layer_idx, 'mlp_out')))

    # 5) temp: input of mlp.down_proj
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(
        _make_pre_hook(manager, layer_idx, 'temp')))

    return handles


def _make_post_hook(manager: StepCacheManager, layer_idx: int, name: str):
    """Forward post-hook: capture module output (+ Phase B optional substitute)."""
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return  # match Phase A behavior for non-Tensor outputs
        # 1. Capture (Phase A behavior)
        manager.record_tensor(layer_idx, name, output)
        # 2. Phase B: maybe substitute output for attn_out / mlp_out
        if name in ('attn_out', 'mlp_out'):
            new_out = manager._maybe_substitute_output(layer_idx, name, output)
            if new_out is not output:
                return new_out
        return None
    return hook


def _make_pre_hook(manager: StepCacheManager, layer_idx: int, name: str):
    """Forward pre-hook: capture module input (the first positional arg)."""
    def hook(module, inputs):
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            manager.record_tensor(layer_idx, name, inputs[0])
    return hook


# ---------------------------------------------------------------------------
# Convenience: context manager for clean attach/detach
# ---------------------------------------------------------------------------

class HookSession:
    """Use as: `with HookSession(model, manager): ...`"""
    def __init__(self, model, manager: StepCacheManager):
        self.model = model
        self.manager = manager
        self.handles = []

    def __enter__(self):
        self.handles = attach_hooks(self.model, self.manager)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []


# ===========================================================================
# Attention Weight Histogram Collector (for 1c-ii motivation plot)
# ===========================================================================
# Attached to each Fast_dLLM_QwenAttention module as .stats_collector.
# The patched modeling.py calls _compute_attention_stats(...) which in turn
# calls collector.record(layer_id, attn_weights) when .enabled is True.

class AttnWeightHistogramCollector:
    """Log-scale histogram of attention weights, 40 bins from 1e-4 to 1."""

    # 40 bin edges: 10 per decade × 4 decades
    _BIN_EDGES = np.concatenate([
        np.logspace(-4, -3, 11)[:-1],
        np.logspace(-3, -2, 11)[:-1],
        np.logspace(-2, -1, 11)[:-1],
        np.logspace(-1,  0, 11),
    ])  # length 41 → 40 bins

    def __init__(self, num_layers=28, dump_path=None):
        self.num_layers = num_layers
        self.dump_path = dump_path
        self.counts = np.zeros((num_layers, 40), dtype=np.int64)
        self.below_min = np.zeros(num_layers, dtype=np.int64)
        self.above_max = np.zeros(num_layers, dtype=np.int64)
        self.total_records = 0
        self.enabled = False

    def record(self, layer_id, attn_weights):
        if not self.enabled:
            return
        flat = attn_weights.detach().float().flatten().cpu().numpy()
        below = (flat < self._BIN_EDGES[0]).sum()
        above = (flat > self._BIN_EDGES[-1]).sum()
        in_range = flat[(flat >= self._BIN_EDGES[0]) & (flat <= self._BIN_EDGES[-1])]
        if in_range.size > 0:
            hist, _ = np.histogram(in_range, bins=self._BIN_EDGES)
            self.counts[layer_id] += hist
        self.below_min[layer_id] += int(below)
        self.above_max[layer_id] += int(above)
        self.total_records += 1

    def dump(self):
        if self.dump_path is None:
            return
        os.makedirs(os.path.dirname(self.dump_path), exist_ok=True)
        np.savez_compressed(
            self.dump_path,
            counts=self.counts,
            bin_edges=self._BIN_EDGES,
            below_min=self.below_min,
            above_max=self.above_max,
            total_records=np.array([self.total_records]),
        )
