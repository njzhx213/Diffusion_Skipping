"""
Plot 1a-ii and 1a-iii (FULL VERSION, no right-tail truncation).
Strictly follows Cheng-jhih's clarification: "average the metric over samples"
— per (step, layer) cell, mean is taken only over samples that have data
at that step, without truncating any step.

Reads: logs/motivation_100/sample_*.npz
Output: figs/02_layer_motivation_mean.png   ← official deliverable
        figs/03_layer_motivation_var.png    ← official deliverable
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = "logs/motivation_100"
OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.npz")))
print(f"Found {len(files)} sample files")

buckets = defaultdict(list)
max_step = 0
for fpath in files:
    data = np.load(fpath, allow_pickle=True)
    step_ids = data['step_id']
    layer_ids = data['layer_id']
    sim_H_in = data['sim_H_in']
    
    for i in range(len(step_ids)):
        sim_arr = np.asarray(sim_H_in[i], dtype=np.float32).flatten()
        mean_sim = float(sim_arr.mean())
        s = int(step_ids[i])
        l = int(layer_ids[i])
        buckets[(s, l)].append(mean_sim)
        max_step = max(max_step, s)
    
    fi = int(os.path.basename(fpath).split('_')[1].split('.')[0])
    if fi % 20 == 0:
        print(f"  Loaded {os.path.basename(fpath)}")

print(f"\nMax step = {max_step}, total cells = {len(buckets)}")

n_layers = 28
n_steps = max_step + 1
mean_mat = np.full((n_layers, n_steps), np.nan, dtype=np.float32)
var_mat = np.full((n_layers, n_steps), np.nan, dtype=np.float32)

for (s, l), vals in buckets.items():
    arr = np.asarray(vals, dtype=np.float32)
    mean_mat[l, s] = arr.mean()
    var_mat[l, s] = arr.var()

print(f"\nMatrix shape: {mean_mat.shape}")
print(f"  mean range: [{np.nanmin(mean_mat):.4f}, {np.nanmax(mean_mat):.4f}]")
print(f"  var  range: [{np.nanmin(var_mat):.6f}, {np.nanmax(var_mat):.6f}]")
print(f"  fraction NaN: {np.isnan(mean_mat).mean():.2%}")

# === Plot mean (no truncation) ===
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(mean_mat, aspect='auto', origin='upper', cmap='viridis',
               vmin=0.5, vmax=1.0, interpolation='nearest')
ax.set_xlabel('Denoising step')
ax.set_ylabel('Layer')
ax.set_title('Mean per-step per-layer cos sim of H_in '
             '(averaged over 100 GSM8K samples per step)')
xtick_step = max(1, n_steps // 20)
ax.set_xticks(np.arange(0, n_steps, xtick_step))
ax.set_xticklabels([str(i) for i in range(0, n_steps, xtick_step)])
ax.set_yticks(np.arange(0, n_layers, 4))
ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('cosine similarity (mean across samples)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_layer_motivation_mean.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR}/02_layer_motivation_mean.png")
plt.close()

# === Plot variance (no truncation) ===
fig, ax = plt.subplots(figsize=(14, 6))
var_vmax = float(np.nanpercentile(var_mat, 99))
im = ax.imshow(var_mat, aspect='auto', origin='upper', cmap='magma',
               vmin=0, vmax=var_vmax, interpolation='nearest')
ax.set_xlabel('Denoising step')
ax.set_ylabel('Layer')
ax.set_title('Variance of per-step per-layer cos sim of H_in (across 100 samples)')
ax.set_xticks(np.arange(0, n_steps, xtick_step))
ax.set_xticklabels([str(i) for i in range(0, n_steps, xtick_step)])
ax.set_yticks(np.arange(0, n_layers, 4))
ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('variance (clipped to 99th pct)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_layer_motivation_var.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/03_layer_motivation_var.png")
plt.close()

print("\n✅ Done.")
