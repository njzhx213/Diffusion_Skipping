"""
Plot 1b-ii and 1b-iii (FULL VERSION, no truncation).
Token-position motivation, averaged over 100 samples per (layer, token_pos).
Cheng-jhih's clarification: average over samples that have data at that
position, without filtering by sample count.

Reads: logs/motivation_100/sample_*.npz
Output: figs/06_token_motivation_mean.png   ← official deliverable
        figs/07_token_motivation_var.png    ← official deliverable
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = "logs/motivation_100"
OUT_DIR = "figs"
BLOCK_SIZE = 32
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.npz")))
print(f"Found {len(files)} samples")

# Cross-sample running sum + sum-of-squares per (layer, token_pos)
sum_per_cell = defaultdict(float)
sumsq_per_cell = defaultdict(float)
cnt_per_cell = defaultdict(int)
max_block = 0

for fi, fpath in enumerate(files):
    data = np.load(fpath, allow_pickle=True)
    block_ids = data['block_id']
    layer_ids = data['layer_id']
    sim_H_in = data['sim_H_in']
    
    # Within this sample, mean over steps for each (layer, token_pos)
    sample_acc = defaultdict(float)
    sample_cnt = defaultdict(int)
    
    for i in range(len(block_ids)):
        block = int(block_ids[i])
        layer = int(layer_ids[i])
        sim_arr = np.asarray(sim_H_in[i], dtype=np.float32).flatten()
        max_block = max(max_block, block)
        for k in range(sim_arr.shape[0]):
            tok_pos = block * BLOCK_SIZE + k
            sample_acc[(layer, tok_pos)] += float(sim_arr[k])
            sample_cnt[(layer, tok_pos)] += 1
    
    # Aggregate sample-mean to cross-sample
    for key, s in sample_acc.items():
        m = s / sample_cnt[key]
        sum_per_cell[key] += m
        sumsq_per_cell[key] += m ** 2
        cnt_per_cell[key] += 1
    
    if fi % 20 == 0:
        print(f"  Processed {os.path.basename(fpath)}")

n_layers = 28
n_token_positions = (max_block + 1) * BLOCK_SIZE
mean_mat = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)
var_mat = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)

for (l, t), s in sum_per_cell.items():
    n = cnt_per_cell[(l, t)]
    if n > 0:
        m = s / n
        mean_mat[l, t] = m
        var_mat[l, t] = max(0.0, sumsq_per_cell[(l, t)] / n - m**2)

# Only trim leading all-NaN columns (token positions 0-N never have any data
# because they're prefill); these are not "truncation," they have no data at all.
all_nan_cols = np.all(np.isnan(mean_mat), axis=0)
first_valid = int(np.argmax(~all_nan_cols))
mean_mat_plot = mean_mat[:, first_valid:]
var_mat_plot = var_mat[:, first_valid:]
token_offset = first_valid

print(f"\nFull token range: 0 to {n_token_positions - 1}")
print(f"  Leading all-NaN cols (no data ever): {first_valid}")
print(f"  After leading NaN trim: shape = {mean_mat_plot.shape}")
print(f"  Mean range: [{np.nanmin(mean_mat_plot):.4f}, {np.nanmax(mean_mat_plot):.4f}]")
print(f"  Var  range: [{np.nanmin(var_mat_plot):.6f}, {np.nanmax(var_mat_plot):.6f}]")

# === Helper ===
def plot_heatmap(matrix, title, cmap, vmin, vmax, cbar_label, out_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect='auto', origin='upper', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Token position (in generated region)')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    xtick_step = max(1, matrix.shape[1] // 20)
    xt = np.arange(0, matrix.shape[1], xtick_step)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(i + token_offset) for i in xt])
    ax.set_yticks(np.arange(0, n_layers, 4))
    ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

plot_heatmap(
    mean_mat_plot,
    'Mean per-token per-layer cos sim of H_in '
    '(averaged over 100 GSM8K samples per token position)',
    cmap='viridis', vmin=0.5, vmax=1.0,
    cbar_label='cosine similarity (mean across samples)',
    out_path=f'{OUT_DIR}/06_token_motivation_mean.png',
)

var_vmax = float(np.nanpercentile(var_mat_plot, 99))
plot_heatmap(
    var_mat_plot,
    'Variance of per-token per-layer cos sim of H_in (across 100 samples)',
    cmap='magma', vmin=0, vmax=var_vmax,
    cbar_label='variance (clipped to 99th pct)',
    out_path=f'{OUT_DIR}/07_token_motivation_var.png',
)

print("\n✅ Done.")
