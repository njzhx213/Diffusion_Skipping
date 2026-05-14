"""
Plot 1c-ii: Attention score range histogram.

Reads: logs/attn_weight_100/attn_weight_hist.npz
Output: figs/09_attn_weight_histogram.png

X-axis: 40 log-scale bins from 1e-4 to 1.0
Y-axis: 28 layers
Color: log(count + 1) — log color scale because counts span ~6 orders of magnitude
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "logs/attn_weight_100/attn_weight_hist.npz"
OUT_PATH = "figs/09_attn_weight_histogram.png"
os.makedirs("figs", exist_ok=True)

# Load
print(f"Loading {DATA_PATH}")
data = np.load(DATA_PATH)
counts = data['counts']         # [28, 40]
bin_edges = data['bin_edges']   # 41 edges
below_min = data['below_min']   # [28]
above_max = data['above_max']   # [28]
total_records = int(data['total_records'][0])

print(f"  Counts shape: {counts.shape}")
print(f"  Total hist records: {total_records}")
print(f"  Per-layer count range: [{counts.sum(axis=1).min()}, {counts.sum(axis=1).max()}]")
print(f"  Below min sum: {below_min.sum()}, Above max sum: {above_max.sum()}")

# === Plot ===
# Use log color scale because counts span huge range
counts_log = np.log10(counts + 1)  # +1 to avoid log(0)

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(
    counts_log,
    aspect='auto',
    origin='upper',
    cmap='viridis',
    interpolation='nearest',
)
ax.set_xlabel('Attention weight value range (log scale)')
ax.set_ylabel('Layer')
ax.set_title('Pre-softmax attention weight distribution per layer\n'
             '(100 GSM8K samples, all decoding steps, all heads, all q-k pairs)')

# X ticks: show every 4th bin's center value
xtick_indices = np.arange(0, 40, 4)
xtick_labels = [f"{bin_edges[i]:.0e}" for i in xtick_indices]
ax.set_xticks(xtick_indices)
ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

# Y ticks: every 4 layers
ax.set_yticks(np.arange(0, 28, 4))
ax.set_yticklabels([str(i) for i in range(0, 28, 4)])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('log10(count + 1)')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")
plt.close()

# === Also print supplementary stats for writeup ===
print("\n=== Supplementary stats (for writeup) ===")
total_per_layer = counts.sum(axis=1) + below_min + above_max
print(f"\n{'Layer':>5} | {'% below 1e-4':>13} | {'% in-range':>11} | {'% above 1':>10}")
print("-" * 50)
for i in range(28):
    pct_below = 100 * below_min[i] / total_per_layer[i]
    pct_in = 100 * counts[i].sum() / total_per_layer[i]
    pct_above = 100 * above_max[i] / total_per_layer[i]
    print(f"{i:>5} | {pct_below:>12.2f}% | {pct_in:>10.2f}% | {pct_above:>9.2f}%")
