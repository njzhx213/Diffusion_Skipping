"""
Improve Phase B plots based on task spec:
  - Main fig: add "Better" arrow, Pareto cluster ellipse
  - Replace bar chart with a TABLE (per task spec 2.b)
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib import patches as mpatches

LOG_ROOT = Path("/home/njzhx/Fast-dLLM/v2/logs/skip_exp")
FIG_ROOT = Path("/home/njzhx/Fast-dLLM/v2/figs")
PHASE_A_ROOT = Path("/home/njzhx/Fast-dLLM/v2/logs/motivation_100")

NUM_LAYERS = 28
BLOCK_SIZE = 32
BASELINE_ACC = 80.0


def categorize(name):
    if name == 'baseline':
        return 'baseline'
    if name.startswith('token_cossim'):
        return 'token_cossim'
    if name.startswith('token_topk'):
        return 'token_topk'
    if name.startswith('layer_cossim_avg'):
        return 'layer_avg'
    if name.startswith('layer_cossim_max'):
        return 'layer_max'
    return 'unknown'


def compute_flops_reduction(reuse_rate, avg_steps, baseline_steps):
    skip_factor = avg_steps * (1 - reuse_rate) / baseline_steps
    return (1 - skip_factor) * 100


def get_baseline_avg_steps():
    files = sorted(PHASE_A_ROOT.glob('sample_*.npz'))
    all_steps = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        all_steps.append(len(np.unique(d['step_id'])))
    return np.mean(all_steps)


def collect():
    baseline_steps = get_baseline_avg_steps()
    summaries = [{
        'setting_name': 'baseline',
        'category': 'baseline',
        'accuracy': BASELINE_ACC,
        'avg_steps': baseline_steps,
        'reuse_rate': 0.0,
        'flops_reduction': 0.0,
    }]
    for d in sorted(LOG_ROOT.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / 'summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        s['category'] = categorize(s['setting_name'])
        s['flops_reduction'] = compute_flops_reduction(
            s.get('reuse_rate', 0), s.get('avg_steps', baseline_steps), baseline_steps
        )
        summaries.append(s)
    return summaries, baseline_steps


# =============================================================================
# Main figure: enhanced scatter with "Better" arrow + Pareto cluster
# =============================================================================
def plot_main(summaries, baseline_steps):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    
    style = {
        'baseline':     {'marker': '*', 'color': '#1f77b4', 's': 320, 'label': 'Baseline'},
        'token_cossim': {'marker': 's', 'color': '#2ca02c', 's': 110, 'label': 'Token cossim (5 thresholds)'},
        'token_topk':   {'marker': 'o', 'color': '#ff7f0e', 's': 130, 'label': 'Token TopK'},
        'layer_avg':    {'marker': '^', 'color': '#d62728', 's': 130, 'label': 'Layer-level avg'},
        'layer_max':    {'marker': 'v', 'color': '#9467bd', 's': 130, 'label': 'Layer-level max'},
    }
    
    # Pre-compute jitter for overlapping clusters
    # token_cossim 5 points all sit at (~96.7, ~82) - spread along x-axis
    cossim_idx_map = {
        'token_cossim_0.995': 0,
        'token_cossim_0.99':  1,
        'token_cossim_0.98':  2,
        'token_cossim_0.97':  3,
        'token_cossim_0.96':  4,
    }
    # layer_max 5 points all sit at (100, 0) - spread along y-axis
    layermax_idx_map = {
        'layer_cossim_max_0.999': 0,
        'layer_cossim_max_0.995': 1,
        'layer_cossim_max_0.99':  2,
        'layer_cossim_max_0.98':  3,
        'layer_cossim_max_0.97':  4,
    }
    
    def get_jitter(s):
        name = s['setting_name']
        if name in cossim_idx_map:
            idx = cossim_idx_map[name]
            # Spread 5 points horizontally: -4, -2, 0, +2, +4 (each 1.5% apart in x)
            # Stack 5 points vertically at same FLOPs reduction, spread on acc axis
            # idx 0..4 → y offset -4, -2, 0, +2, +4 (so 5 points span 8 acc-units)
            return (0, (idx - 2) * 2.0)
        if name in layermax_idx_map:
            idx = layermax_idx_map[name]
            # Spread 5 points vertically (acc dim), and slight x offset to push them inside
            return (-(idx + 1) * 1.0, (idx - 2) * 1.8)
        return (0, 0)
    
    seen = set()
    for s in summaries:
        cat = s['category']
        sty = style[cat]
        label = sty['label'] if cat not in seen else None
        seen.add(cat)
        dx, dy = get_jitter(s)
        ax.scatter(s['flops_reduction'] + dx, s['accuracy'] + dy,
                   marker=sty['marker'], color=sty['color'], s=sty['s'],
                   label=label, edgecolors='black', linewidth=1.2, alpha=0.85, zorder=4)
        s['_plot_x'] = s['flops_reduction'] + dx
        s['_plot_y'] = s['accuracy'] + dy
    
    # Add minimal annotations (threshold values, hide overlaps)
    # Special offsets for clustered groups (token_cossim, layer_max) to spread labels
    cossim_anno_offsets = {
        'token_cossim_0.995': (8, 0),     # all to the right, vertically aligned with point
        'token_cossim_0.99':  (8, 0),
        'token_cossim_0.98':  (8, 0),
        'token_cossim_0.97':  (8, 0),
        'token_cossim_0.96':  (8, 0),
    }
    layermax_anno_offsets = {
        'layer_cossim_max_0.999': (8, -4),
        'layer_cossim_max_0.995': (8, -4),
        'layer_cossim_max_0.99':  (8, -4),
        'layer_cossim_max_0.98':  (8, -4),
        'layer_cossim_max_0.97':  (8, -4),
    }
    
    for s in summaries:
        if s['category'] == 'baseline':
            continue
        # Short label
        if 'cossim' in s['setting_name']:
            anno = s['setting_name'].split('_')[-1]
        elif 'topk' in s['setting_name']:
            anno = f"k={s['setting_name'].split('_')[-1]}"
        else:
            continue
        
        pos = (s.get('_plot_x', s['flops_reduction']), s.get('_plot_y', s['accuracy']))
        # Choose offset based on setting
        name = s['setting_name']
        if name in cossim_anno_offsets:
            offset = cossim_anno_offsets[name]
        elif name in layermax_anno_offsets:
            offset = layermax_anno_offsets[name]
        else:
            offset = (8, 5)
        ax.annotate(anno, pos, xytext=offset, textcoords='offset points', 
                    fontsize=8.5, color='#444', fontweight='bold')
    
    # Baseline horizontal line
    ax.axhline(BASELINE_ACC, color='gray', linestyle='--', linewidth=1, alpha=0.6,
               label=f'Baseline acc ({BASELINE_ACC:.0f}%)')
    
    # ★ Better arrow (lower-left → upper-right)
    arrow = FancyArrowPatch(
        (45, 30), (85, 70),
        arrowstyle='->', mutation_scale=30,
        color='#1f77b4', linewidth=2.5, alpha=0.8, zorder=5
    )
    ax.add_patch(arrow)
    ax.text(65, 33, 'Better', fontsize=14, fontweight='bold', color='#1f77b4',
            rotation=45, ha='center', va='center')
    
    # ★ Pareto frontier cluster: encircle the high-acc + high-FLOPs-reduction points
    # The "good" points: token_cossim, topk_50, topk_25
    # Center: ~(80, 82), spread to (97, 85)
    pareto_ellipse = Ellipse(
        xy=(82, 83), width=45, height=15,
        edgecolor='green', facecolor='lightgreen', alpha=0.18,
        linewidth=2, linestyle='-', zorder=2
    )
    ax.add_patch(pareto_ellipse)
    ax.text(82, 93, 'High-FLOPs-reduction Pareto cluster',
            fontsize=10, color='darkgreen', ha='center', style='italic')
    
    # Note about jitter (small text under plot)
    fig.text(0.5, 0.02,
             "Note: 5 token_cossim points (~97% FLOPs↓, ~82% acc) and 5 layer_max points (~100% FLOPs↓, ~0% acc) "
             "are jittered for visibility; their true positions are identical (see Table).",
             ha='center', fontsize=8, style='italic', color='#666')
    ax.set_xlabel('FLOPs reduction compared to baseline (%)', fontsize=12)
    ax.set_ylabel('GSM8K accuracy (%)', fontsize=12)
    ax.set_title('Phase B: Accuracy vs FLOPs Reduction (18 settings)', fontsize=13.5, pad=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.9)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-7, 100)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # leave bottom margin for footnote
    path = FIG_ROOT / 'phase_b_acc_flops_v2.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# =============================================================================
# Table figure: matplotlib table for average denoising steps
# =============================================================================
def plot_table(summaries):
    """Matplotlib table summarizing all 18 settings: avg steps + acc + FLOPs%."""
    
    # Order rows
    order = {
        'baseline': 0,
        'token_cossim': 1, 'token_topk': 2,
        'layer_avg': 3, 'layer_max': 4,
    }
    summaries_sorted = sorted(
        summaries,
        key=lambda s: (order[s['category']], -s.get('reuse_rate', 0))
    )
    
    cell_data = []
    cell_colors = []
    
    color_map = {
        'baseline':     '#e6f0fa',
        'token_cossim': '#e8f5e8',
        'token_topk':   '#fff4e6',
        'layer_avg':    '#fdecec',
        'layer_max':    '#f3eaf7',
    }
    
    for s in summaries_sorted:
        cell_data.append([
            s['setting_name'],
            f"{s.get('avg_steps', 0):.1f}",
            f"{s['accuracy']:.1f}%",
            f"{s.get('reuse_rate', 0)*100:.1f}%",
            f"{s['flops_reduction']:.1f}%",
        ])
        c = color_map.get(s['category'], 'white')
        cell_colors.append([c, c, c, c, c])
    
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axis('off')
    table = ax.table(
        cellText=cell_data,
        cellColours=cell_colors,
        colLabels=['Setting', 'Avg Steps', 'Accuracy', 'Reuse Rate', 'FLOPs↓'],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)
    # Bold header
    for i in range(5):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_facecolor('#444')
    
    plt.title('Phase B: Average Denoising Steps and Accuracy by Setting',
              fontsize=13.5, pad=15)
    plt.tight_layout()
    path = FIG_ROOT / 'phase_b_steps_table.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def main():
    summaries, baseline_steps = collect()
    print(f"Baseline avg steps: {baseline_steps:.1f}")
    print(f"Total settings: {len(summaries)}")
    plot_main(summaries, baseline_steps)
    plot_table(summaries)


if __name__ == "__main__":
    main()
