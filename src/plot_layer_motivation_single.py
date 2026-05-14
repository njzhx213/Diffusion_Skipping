"""
Plot 1a-i: Layer skipping motivation, single sample.
  x-axis: denoising step
  y-axis: layer (0-27)
  heatmap: per-step per-layer mean cosine similarity of H_in

Reads: logs/motivation_5/sample_0000.npz
Output: figs/01_layer_motivation_single.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_PATH = "logs/motivation_5/sample_0000.npz"
OUT_PATH = "figs/01_layer_motivation_single.png"
os.makedirs("figs", exist_ok=True)

# Load
print(f"Loading {SAMPLE_PATH}")
data = np.load(SAMPLE_PATH, allow_pickle=True)
print(f"  Keys: {list(data.keys())}")
print(f"  Records: {len(data['sample_id'])}")

# Records are flat: each row is (step_id, layer_id, sim_H_in shape [1, L])
# We want to build a 2D matrix [num_layers, num_steps] where each cell is
# the *mean* sim across all tokens for that (step, layer).

step_ids = data['step_id']
layer_ids = data['layer_id']
sim_H_in = data['sim_H_in']  # shape (N,) of (1, L) arrays, or (N, 1, L) array

# Convert: each record -> a scalar (mean over tokens)
sim_scalars = np.array([s.mean() for s in sim_H_in], dtype=np.float32)
print(f"  Sim per-record: shape={sim_scalars.shape}, "
      f"mean={sim_scalars.mean():.4f}, min={sim_scalars.min():.4f}")

# Determine dimensions
all_layers = sorted(set(layer_ids.tolist()))
all_steps = sorted(set(step_ids.tolist()))
n_layers = len(all_layers)
n_steps = len(all_steps)
step_to_col = {s: i for i, s in enumerate(all_steps)}
layer_to_row = {l: i for i, l in enumerate(all_layers)}

# Build matrix
mat = np.full((n_layers, n_steps), np.nan, dtype=np.float32)
for i in range(len(step_ids)):
    r = layer_to_row[int(layer_ids[i])]
    c = step_to_col[int(step_ids[i])]
    mat[r, c] = sim_scalars[i]

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(
    mat,
    aspect='auto',
    origin='upper',
    cmap='viridis',
    vmin=0.5, vmax=1.0,
    interpolation='nearest',
)
ax.set_xlabel('Denoising step')
ax.set_ylabel('Layer')
ax.set_title('Per-step per-layer mean cos sim of H_in (1 GSM8K sample, baseline)')

# Reasonable tick spacing
xtick_step = max(1, n_steps // 20)
ax.set_xticks(np.arange(0, n_steps, xtick_step))
ax.set_xticklabels([str(all_steps[i]) for i in range(0, n_steps, xtick_step)])
ax.set_yticks(np.arange(0, n_layers, 4))
ax.set_yticklabels([str(all_layers[i]) for i in range(0, n_layers, 4)])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('cosine similarity')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")

# Print some stats for sanity
print(f"\nMatrix stats:")
print(f"  Shape: {mat.shape}")
print(f"  Mean: {np.nanmean(mat):.4f}")
print(f"  Min: {np.nanmin(mat):.4f}")
print(f"  Max: {np.nanmax(mat):.4f}")
print(f"  Fraction NaN: {np.isnan(mat).mean():.2%}")
