"""
Plot 1b-ii / 1b-iii / 1b-iv:
  1b-ii: mean of per-token per-layer cos sim of H_in, averaged over 100 samples
  1b-iii: variance across samples
  1b-iv: H_in sim vs attn_out sim scatter (Pearson correlation)

For mean/var: per-token-position heatmap (x=token position, y=layer)
  Each (layer, token_pos) cell is the mean (or var) across all samples that
  generated that position.

For scatter: collect (mean-over-tokens H_in sim, mean-over-tokens attn_out sim)
  for each record in each sample → one point per record.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = "logs/motivation_100"
OUT_DIR = "figs"
BLOCK_SIZE = 32
N_MIN_SAMPLES = 20  # drop columns with fewer than this many samples (right tail)
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.npz")))
print(f"Found {len(files)} samples")

# Cross-sample running sum / sum-of-squares per (layer, token_pos)
sum_per_cell  = defaultdict(float)
sumsq_per_cell = defaultdict(float)
cnt_per_cell  = defaultdict(int)

# 1b-iv: collect per-record (H_mean, attn_mean) pairs
H_for_scatter = []
ATT_for_scatter = []

max_block = 0

for fi, fpath in enumerate(files):
    data = np.load(fpath, allow_pickle=True)
    block_ids = data['block_id']
    layer_ids = data['layer_id']
    sim_H_in = data['sim_H_in']
    sim_attn = data['sim_attn_out']

    # Within this sample, compute per-(layer, token_pos) mean across steps
    sample_acc  = defaultdict(float)
    sample_cnt  = defaultdict(int)

    for i in range(len(block_ids)):
        block = int(block_ids[i])
        layer = int(layer_ids[i])
        sim_arr = np.asarray(sim_H_in[i], dtype=np.float32).flatten()
        sim_att = np.asarray(sim_attn[i], dtype=np.float32).flatten()
        L = sim_arr.shape[0]
        max_block = max(max_block, block)
        for k in range(L):
            tok_pos = block * BLOCK_SIZE + k
            sample_acc[(layer, tok_pos)] += float(sim_arr[k])
            sample_cnt[(layer, tok_pos)] += 1
        # Scatter: per-record means
        H_for_scatter.append(float(sim_arr.mean()))
        ATT_for_scatter.append(float(sim_att.mean()))

    # Aggregate sample-level means into cross-sample accumulator
    for key, s in sample_acc.items():
        c = sample_cnt[key]
        sample_mean = s / c
        sum_per_cell[key]  += sample_mean
        sumsq_per_cell[key] += sample_mean ** 2
        cnt_per_cell[key]  += 1

    if fi % 20 == 0:
        print(f"  Processed {os.path.basename(fpath)}")

# === Build full matrices ===
n_layers = 28
n_token_positions = (max_block + 1) * BLOCK_SIZE

mean_mat = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)
var_mat  = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)
count_mat = np.zeros((n_layers, n_token_positions), dtype=np.int32)

for (l, t), s in sum_per_cell.items():
    n = cnt_per_cell[(l, t)]
    if n > 0:
        m = s / n
        mean_mat[l, t] = m
        var_mat[l, t]  = max(0.0, sumsq_per_cell[(l, t)] / n - m**2)
        count_mat[l, t] = n

# === Trim leading + trailing low-sample columns ===
all_nan_cols = np.all(np.isnan(mean_mat), axis=0)
first_valid = int(np.argmax(~all_nan_cols))

sample_count_per_col = count_mat[0]  # representative
valid_mask = sample_count_per_col >= N_MIN_SAMPLES
last_valid = int(np.where(valid_mask)[0].max()) if np.any(valid_mask) else len(valid_mask) - 1

mean_mat = mean_mat[:, first_valid:last_valid + 1]
var_mat  = var_mat[:, first_valid:last_valid + 1]
count_mat = count_mat[:, first_valid:last_valid + 1]
token_offset = first_valid

print(f"\nFinal shape: {mean_mat.shape}")
print(f"  Token offset: {token_offset}, last token: {token_offset + mean_mat.shape[1] - 1}")
print(f"  Sample count in range: {count_mat[0].min()} - {count_mat[0].max()}")
print(f"  Mean range: [{np.nanmin(mean_mat):.4f}, {np.nanmax(mean_mat):.4f}]")
print(f"  Var range: [{np.nanmin(var_mat):.6f}, {np.nanmax(var_mat):.6f}]")

# === Plotting helper ===
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

# 1b-ii: mean
plot_heatmap(
    mean_mat,
    'Mean per-token per-layer cos sim of H_in (avg over up to 100 GSM8K samples)',
    cmap='viridis', vmin=0.5, vmax=1.0,
    cbar_label='cosine similarity (mean)',
    out_path=f'{OUT_DIR}/06_token_motivation_mean.png',
)

# 1b-iii: variance
var_vmax = float(np.nanpercentile(var_mat, 99))
plot_heatmap(
    var_mat,
    'Variance of per-token per-layer cos sim of H_in (across samples)',
    cmap='magma', vmin=0, vmax=var_vmax,
    cbar_label='variance (clipped to 99th pct)',
    out_path=f'{OUT_DIR}/07_token_motivation_var.png',
)

# === 1b-iv: scatter ===
H_arr = np.array(H_for_scatter, dtype=np.float32)
ATT_arr = np.array(ATT_for_scatter, dtype=np.float32)
print(f"\nScatter data: {len(H_arr)} points")
rho = float(np.corrcoef(H_arr, ATT_arr)[0, 1])
print(f"  Pearson correlation ρ = {rho:.4f}")

# Subsample for plotting density
N_PLOT = 100000
if len(H_arr) > N_PLOT:
    idx = np.random.RandomState(42).choice(len(H_arr), N_PLOT, replace=False)
    H_plot, ATT_plot = H_arr[idx], ATT_arr[idx]
else:
    H_plot, ATT_plot = H_arr, ATT_arr

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(H_plot, ATT_plot, s=1, alpha=0.1, color='steelblue')
ax.plot([-0.1, 1.05], [-0.1, 1.05], 'r--', linewidth=1, alpha=0.7, label='y = x')
ax.set_xlabel('Similarity of H_in (mean over tokens per record)')
ax.set_ylabel('Similarity of attn_out (mean over tokens per record)')
ax.set_title(f'H_in sim vs AttnOut sim\nρ = {rho:.3f}  (n={len(H_arr)} records)')
ax.set_xlim(-0.1, 1.05)
ax.set_ylim(-0.1, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_H_vs_AttnOut_scatter.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/08_H_vs_AttnOut_scatter.png")
plt.close()

print("\n✅ Done.")
