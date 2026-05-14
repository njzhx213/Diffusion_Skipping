"""
Plot 1a-ii and 1a-iii: Layer skipping motivation, mean and variance over 100 samples.
  x-axis: denoising step
  y-axis: layer (0-27)
  heatmap: similarity (mean / variance over samples)

Reads: logs/motivation_100/sample_*.npz
Output: figs/02_layer_motivation_mean.png
        figs/03_layer_motivation_var.png
        figs/04_layer_motivation_count.png  (sanity: samples per step)
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "logs/motivation_100"
OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load all npz files ===
files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.npz")))
print(f"Found {len(files)} sample files")

# Per-(step, layer) accumulator
# key: (step_id, layer_id), value: list of mean-over-tokens sim values
from collections import defaultdict
buckets = defaultdict(list)

max_step = 0
for fpath in files:
    data = np.load(fpath, allow_pickle=True)
    step_ids = data['step_id']
    layer_ids = data['layer_id']
    sim_H_in = data['sim_H_in']
    # sim_H_in could be (N, 1, L) ndarray OR (N,) object array of (1, L) arrays
    
    for i in range(len(step_ids)):
        sim_arr = sim_H_in[i]
        # mean over tokens
        if isinstance(sim_arr, np.ndarray):
            mean_sim = float(sim_arr.mean())
        else:
            mean_sim = float(np.asarray(sim_arr).mean())
        
        s = int(step_ids[i])
        l = int(layer_ids[i])
        buckets[(s, l)].append(mean_sim)
        max_step = max(max_step, s)
    
    if int(os.path.basename(fpath).split('_')[1].split('.')[0]) % 20 == 0:
        print(f"  Loaded {os.path.basename(fpath)}")

print(f"\nLoaded all files. Max step = {max_step}")
print(f"Total (step, layer) cells: {len(buckets)}")

# === Build 2D matrices ===
n_layers = 28
n_steps = max_step + 1

mean_mat = np.full((n_layers, n_steps), np.nan, dtype=np.float32)
var_mat = np.full((n_layers, n_steps), np.nan, dtype=np.float32)
count_mat = np.zeros((n_layers, n_steps), dtype=np.int32)

for (s, l), vals in buckets.items():
    arr = np.asarray(vals, dtype=np.float32)
    mean_mat[l, s] = arr.mean()
    var_mat[l, s] = arr.var()
    count_mat[l, s] = len(arr)

print(f"\nMatrix stats:")
print(f"  mean: range [{np.nanmin(mean_mat):.4f}, {np.nanmax(mean_mat):.4f}]")
print(f"  var:  range [{np.nanmin(var_mat):.6f}, {np.nanmax(var_mat):.6f}]")
print(f"  Sample count per step (range): "
      f"{count_mat[0].min()} - {count_mat[0].max()}")

# === Decide which step range to actually plot ===
# Drop steps where fewer than N_MIN samples have data (right tail unreliable)
N_MIN = 20  # need at least 20 samples (out of 100) for the average to be meaningful
sample_count_per_step = count_mat[0]  # layer 0 sample count, same for all layers
valid_mask = sample_count_per_step >= N_MIN
last_valid_step = int(np.where(valid_mask)[0].max())
print(f"\nUsing steps 0 to {last_valid_step} (≥{N_MIN} samples each)")
print(f"  Discarded {n_steps - 1 - last_valid_step} steps with <{N_MIN} samples")

mean_mat_plot = mean_mat[:, :last_valid_step + 1]
var_mat_plot = var_mat[:, :last_valid_step + 1]

# === Plot mean heatmap ===
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(
    mean_mat_plot, aspect='auto', origin='upper', cmap='viridis',
    vmin=0.5, vmax=1.0, interpolation='nearest',
)
ax.set_xlabel('Denoising step')
ax.set_ylabel('Layer')
ax.set_title(f'Mean per-step per-layer cos sim of H_in (avg over up to 100 GSM8K samples)')
ax.set_yticks(np.arange(0, n_layers, 4))
ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])
xtick_step = max(1, mean_mat_plot.shape[1] // 20)
ax.set_xticks(np.arange(0, mean_mat_plot.shape[1], xtick_step))
ax.set_xticklabels([str(i) for i in range(0, mean_mat_plot.shape[1], xtick_step)])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('cosine similarity (mean)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_layer_motivation_mean.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR}/02_layer_motivation_mean.png")

# === Plot variance heatmap ===
fig, ax = plt.subplots(figsize=(14, 6))
# variance is small; use log scale or stretched range
im = ax.imshow(
    var_mat_plot, aspect='auto', origin='upper', cmap='magma',
    vmin=0, vmax=float(np.nanpercentile(var_mat_plot, 99)),
    interpolation='nearest',
)
ax.set_xlabel('Denoising step')
ax.set_ylabel('Layer')
ax.set_title(f'Variance of per-step per-layer cos sim of H_in (across up to 100 samples)')
ax.set_yticks(np.arange(0, n_layers, 4))
ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])
ax.set_xticks(np.arange(0, var_mat_plot.shape[1], xtick_step))
ax.set_xticklabels([str(i) for i in range(0, var_mat_plot.shape[1], xtick_step)])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('variance (clipped to 99th pct)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_layer_motivation_var.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/03_layer_motivation_var.png")

# === Plot sample count (sanity / supplementary) ===
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(sample_count_per_step, color='steelblue')
ax.axhline(N_MIN, color='red', linestyle='--', linewidth=1,
           label=f'min threshold N={N_MIN}')
ax.axvline(last_valid_step, color='red', linestyle=':', linewidth=1)
ax.set_xlabel('Denoising step')
ax.set_ylabel('Number of samples with data')
ax.set_title('Sample coverage per step (right tail thins because long samples are rare)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_layer_motivation_count.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/04_layer_motivation_count.png")

print("\n✅ Done.")
