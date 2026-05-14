"""
Plot 1b-i: Token skipping motivation, single sample.
  x-axis: global token position in generated region
  y-axis: layer (0-27)
  heatmap: mean cos sim of H_in, averaged over all denoising steps

Reads: logs/motivation_100/sample_0000.npz
Output: figs/05_token_motivation_single.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_PATH = "logs/motivation_100/sample_0000.npz"
OUT_PATH = "figs/05_token_motivation_single.png"
BLOCK_SIZE = 32
os.makedirs("figs", exist_ok=True)

# === Load ===
print(f"Loading {SAMPLE_PATH}")
data = np.load(SAMPLE_PATH, allow_pickle=True)
print(f"  Records: {len(data['sample_id'])}")

step_ids = data['step_id']
block_ids = data['block_id']
layer_ids = data['layer_id']
sim_H_in = data['sim_H_in']

max_block = int(block_ids.max())
min_block = int(block_ids.min())
n_token_positions = (max_block + 1) * BLOCK_SIZE
n_layers = 28
print(f"  Block range: {min_block} to {max_block}")
print(f"  Token positions: 0 to {n_token_positions - 1}")

# === Accumulate per (layer, global_token_pos) ===
acc = np.zeros((n_layers, n_token_positions), dtype=np.float64)
count = np.zeros((n_layers, n_token_positions), dtype=np.int32)

for i in range(len(step_ids)):
    block = int(block_ids[i])
    layer = int(layer_ids[i])
    sim_arr = np.asarray(sim_H_in[i], dtype=np.float32).flatten()
    L = sim_arr.shape[0]
    start = block * BLOCK_SIZE
    end = min(start + L, n_token_positions)
    acc[layer, start:end] += sim_arr[:end - start]
    count[layer, start:end] += 1

# === Compute mean (avoid div by 0) ===
mean_mat = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)
mask = count > 0
mean_mat[mask] = (acc[mask] / count[mask]).astype(np.float32)

print(f"\nBefore trim:")
print(f"  Shape: {mean_mat.shape}")
print(f"  NaN cells: {np.isnan(mean_mat).sum()} / {mean_mat.size}")

# === Trim leading all-NaN columns ===
all_nan_cols = np.all(np.isnan(mean_mat), axis=0)
first_valid = int(np.argmax(~all_nan_cols))  # first column with any data
mean_mat = mean_mat[:, first_valid:]
token_offset = first_valid
print(f"\nTrimmed {first_valid} leading all-NaN cols")
print(f"  New shape: {mean_mat.shape}")
print(f"  Mean (over non-NaN): {np.nanmean(mean_mat):.4f}")
print(f"  Min: {np.nanmin(mean_mat):.4f}")
print(f"  Max: {np.nanmax(mean_mat):.4f}")

# === Plot ===
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(
    mean_mat, aspect='auto', origin='upper', cmap='viridis',
    vmin=0.5, vmax=1.0, interpolation='nearest',
)
ax.set_xlabel('Token position (in generated region)')
ax.set_ylabel('Layer')
ax.set_title('Per-token per-layer mean cos sim of H_in '
             '(1 GSM8K sample, avg over denoising steps)')

# X ticks: show absolute token position (with offset)
xtick_step = max(1, mean_mat.shape[1] // 20)
xt = np.arange(0, mean_mat.shape[1], xtick_step)
ax.set_xticks(xt)
ax.set_xticklabels([str(i + token_offset) for i in xt])

# Y ticks: every 4 layers
ax.set_yticks(np.arange(0, n_layers, 4))
ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])

# Block boundaries (shift by token_offset)
for b in range(1, max_block + 1):
    x = b * BLOCK_SIZE - 0.5 - token_offset
    if 0 < x < mean_mat.shape[1]:
        ax.axvline(x, color='white', linewidth=0.3, alpha=0.5)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('cosine similarity (mean over steps)')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")
