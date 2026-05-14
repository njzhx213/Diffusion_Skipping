"""
Plot 1c-i (attn_out sim) and 1c-iii (FFN temp sim).
Both are heatmaps of mean per-(step, layer) sim across 100 GSM8K samples.
Structure identical to plot_layer_motivation_100.py but for different
sim fields.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = "logs/motivation_100"
OUT_DIR = "figs"
N_MIN_SAMPLES = 20
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.npz")))
print(f"Found {len(files)} samples")


def aggregate_and_plot(sim_field, title, out_path):
    """Read sim_field across all samples, build mean matrix, plot heatmap."""
    print(f"\n=== Processing {sim_field} → {out_path} ===")
    buckets = defaultdict(list)
    max_step = 0
    
    for fi, fpath in enumerate(files):
        data = np.load(fpath, allow_pickle=True)
        step_ids = data['step_id']
        layer_ids = data['layer_id']
        sim_arr_all = data[sim_field]
        
        for i in range(len(step_ids)):
            s = int(step_ids[i])
            l = int(layer_ids[i])
            sim_per_token = np.asarray(sim_arr_all[i], dtype=np.float32).flatten()
            mean_sim = float(sim_per_token.mean())
            buckets[(s, l)].append(mean_sim)
            max_step = max(max_step, s)
        
        if fi % 20 == 0:
            print(f"  Loaded {os.path.basename(fpath)}")
    
    n_layers = 28
    n_steps = max_step + 1
    mean_mat = np.full((n_layers, n_steps), np.nan, dtype=np.float32)
    count_mat = np.zeros((n_layers, n_steps), dtype=np.int32)
    
    for (s, l), vals in buckets.items():
        arr = np.asarray(vals, dtype=np.float32)
        mean_mat[l, s] = arr.mean()
        count_mat[l, s] = len(arr)
    
    # Trim by sample-count threshold (same as 1a-ii)
    sample_count_per_step = count_mat[0]
    valid_mask = sample_count_per_step >= N_MIN_SAMPLES
    if not np.any(valid_mask):
        print(f"  WARNING: no steps with ≥{N_MIN_SAMPLES} samples!")
        return
    last_valid = int(np.where(valid_mask)[0].max())
    mean_mat = mean_mat[:, :last_valid + 1]
    
    print(f"  Final shape: {mean_mat.shape}")
    print(f"  Mean range: [{np.nanmin(mean_mat):.4f}, {np.nanmax(mean_mat):.4f}]")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mean_mat, aspect='auto', origin='upper', cmap='viridis',
                   vmin=0.5, vmax=1.0, interpolation='nearest')
    ax.set_xlabel('Denoising step')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    xtick_step = max(1, mean_mat.shape[1] // 20)
    ax.set_xticks(np.arange(0, mean_mat.shape[1], xtick_step))
    ax.set_xticklabels([str(i) for i in range(0, mean_mat.shape[1], xtick_step)])
    ax.set_yticks(np.arange(0, n_layers, 4))
    ax.set_yticklabels([str(i) for i in range(0, n_layers, 4)])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('cosine similarity (mean)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


# 1c-i: Attention output similarity
aggregate_and_plot(
    sim_field='sim_attn_out',
    title='Mean per-step per-layer cos sim of attn_out (avg over 100 GSM8K samples)',
    out_path=f'{OUT_DIR}/10_attn_out_motivation.png',
)

# 1c-iii: FFN temp similarity
aggregate_and_plot(
    sim_field='sim_temp',
    title='Mean per-step per-layer cos sim of FFN temp (avg over 100 GSM8K samples)\n'
          'temp = act_fn(gate_proj(x)) * up_proj(x)',
    out_path=f'{OUT_DIR}/11_ffn_temp_motivation.png',
)

print("\n✅ Done.")
