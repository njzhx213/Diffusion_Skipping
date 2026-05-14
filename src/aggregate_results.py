"""
Aggregate all 18 setting results (1 baseline + 17 Phase B settings).
Compute FLOPs reduction and produce acc-FLOPs scatter + table.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

LOG_ROOT = Path("/home/njzhx/Fast-dLLM/v2/logs/skip_exp")
PHASE_A_ROOT = Path("/home/njzhx/Fast-dLLM/v2/logs/motivation_100")

# Constants
BASELINE_AVG_STEPS = None  # Computed from Phase A data, see main()
NUM_LAYERS = 28
BLOCK_SIZE = 32

# Baseline value taken from Phase A; 80/100 accuracy
BASELINE_ACC = 80.0

# ---------------------------------------------------------------------------
# Phase A baseline: synthesize from motivation_100 data
# ---------------------------------------------------------------------------
def get_baseline_summary():
    """Estimate baseline stats from Phase A logs."""
    # Phase A logged sim records but not "correct" labels per sample.
    # We can count records to verify total steps, but acc=80.0% is known.
    files = sorted(PHASE_A_ROOT.glob('sample_*.npz'))
    n_total_samples = len(files)
    total_steps = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        # Each record = (step, layer), so total / num_layers = total steps
        n_records = len(d['step_id'])
        sample_steps = n_records // NUM_LAYERS
        total_steps += sample_steps
    return {
        'setting_name': 'baseline',
        'accuracy': BASELINE_ACC,
        'avg_steps': total_steps / n_total_samples if n_total_samples > 0 else BASELINE_AVG_STEPS,
        'reuse_rate': 0.0,
        'category': 'baseline',
    }


# ---------------------------------------------------------------------------
# Per-setting: load summary + categorize
# ---------------------------------------------------------------------------
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
    """
    Compute FLOPs reduction relative to baseline.
    
    baseline total computation = baseline_steps × num_layers × block_size × per_token_FLOPs
    skip total computation     = avg_steps × num_layers × block_size × (1 - reuse_rate) × per_token_FLOPs
    
    reduction = 1 - skip/baseline
              = 1 - (avg_steps × (1 - reuse_rate)) / baseline_steps
    """
    skip_factor = avg_steps * (1 - reuse_rate) / baseline_steps
    reduction = 1 - skip_factor
    return reduction * 100  # percent


def collect_all_settings():
    summaries = [get_baseline_summary()]
    for d in sorted(LOG_ROOT.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / 'summary.json'
        if not summary_path.exists():
            print(f"  skip {d.name}: no summary.json")
            continue
        with open(summary_path) as f:
            s = json.load(f)
        s['category'] = categorize(s['setting_name'])
        summaries.append(s)
    return summaries


# ---------------------------------------------------------------------------
# Compute FLOPs reduction for each
# ---------------------------------------------------------------------------
def main():
    global BASELINE_AVG_STEPS
    summaries = collect_all_settings()
    
    # Use the baseline's actual avg_steps as reference
    baseline = next(s for s in summaries if s['category'] == 'baseline')
    BASELINE_AVG_STEPS = baseline['avg_steps']
    
    print(f"Loaded {len(summaries)} settings")
    print(f"Baseline avg_steps: {BASELINE_AVG_STEPS:.1f}\n")
    
    # Compute FLOPs reduction
    for s in summaries:
        s['flops_reduction'] = compute_flops_reduction(
            s.get('reuse_rate', 0),
            s.get('avg_steps', BASELINE_AVG_STEPS),
            BASELINE_AVG_STEPS,
        )
    
    # ----- Print table -----
    print(f"{'Setting':<28s} {'Acc%':>6s} {'Steps':>7s} {'Reuse%':>7s} {'FLOPs↓':>8s}")
    print("─" * 65)
    
    # Order: baseline, token_cossim (descending threshold), topk, layer_avg, layer_max
    def sort_key(s):
        order = {
            'baseline': 0,
            'token_cossim': 1,
            'token_topk': 2,
            'layer_avg': 3,
            'layer_max': 4,
        }
        return (order.get(s['category'], 99), -s.get('reuse_rate', 0))
    
    summaries_sorted = sorted(summaries, key=sort_key)
    
    for s in summaries_sorted:
        name = s['setting_name']
        acc = s['accuracy']
        steps = s.get('avg_steps', 0)
        reuse = s.get('reuse_rate', 0) * 100
        flops = s['flops_reduction']
        print(f"  {name:<26s} {acc:>5.1f}% {steps:>7.1f} {reuse:>6.1f}% {flops:>7.1f}%")
    
    # ----- Save aggregated CSV -----
    csv_path = LOG_ROOT.parent / 'phase_b_aggregate.csv'
    with open(csv_path, 'w') as f:
        f.write("setting,category,accuracy,avg_steps,reuse_rate,flops_reduction\n")
        for s in summaries_sorted:
            f.write(f"{s['setting_name']},{s['category']},{s['accuracy']:.1f},"
                    f"{s.get('avg_steps', 0):.1f},{s.get('reuse_rate', 0)*100:.1f},"
                    f"{s['flops_reduction']:.1f}\n")
    print(f"\nSaved CSV: {csv_path}")
    
    # ----- Plot acc-FLOPs scatter -----
    fig, ax = plt.subplots(figsize=(10, 7))
    
    style = {
        'baseline':     {'marker': '*', 'color': 'black', 's': 250, 'label': 'Baseline'},
        'token_cossim': {'marker': 's', 'color': 'tab:blue', 's': 120, 'label': 'Token cossim'},
        'token_topk':   {'marker': 'o', 'color': 'tab:green', 's': 120, 'label': 'Token TopK'},
        'layer_avg':    {'marker': '^', 'color': 'tab:orange', 's': 120, 'label': 'Layer avg'},
        'layer_max':    {'marker': 'v', 'color': 'tab:red', 's': 120, 'label': 'Layer max'},
    }
    
    seen_labels = set()
    for s in summaries_sorted:
        cat = s['category']
        sty = style[cat]
        label = sty['label'] if cat not in seen_labels else None
        seen_labels.add(cat)
        
        ax.scatter(s['flops_reduction'], s['accuracy'],
                   marker=sty['marker'], color=sty['color'], s=sty['s'],
                   label=label, edgecolors='black', linewidth=1, alpha=0.8, zorder=3)
        
        # Annotate with setting name (short)
        if cat == 'baseline':
            anno = 'baseline'
        elif 'cossim' in s['setting_name']:
            anno = s['setting_name'].split('_')[-1]  # threshold value
        elif 'topk' in s['setting_name']:
            anno = f"k={s['setting_name'].split('_')[-1]}"
        else:
            anno = ''
        ax.annotate(anno, (s['flops_reduction'], s['accuracy']),
                    xytext=(7, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('FLOPs reduction (%)', fontsize=12)
    ax.set_ylabel('GSM8K accuracy (%)', fontsize=12)
    ax.set_title('Phase B: accuracy vs FLOPs reduction (18 settings)', fontsize=13)
    ax.axhline(BASELINE_ACC, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
               label=f'Baseline acc ({BASELINE_ACC}%)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 100)
    
    plt.tight_layout()
    fig_path = LOG_ROOT.parent.parent / 'figs' / 'phase_b_acc_flops.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {fig_path}")
    
    # ----- Plot avg steps comparison -----
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    names = [s['setting_name'] for s in summaries_sorted]
    steps = [s.get('avg_steps', 0) for s in summaries_sorted]
    colors = [style[s['category']]['color'] for s in summaries_sorted]
    
    bars = ax2.bar(range(len(names)), steps, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(BASELINE_AVG_STEPS, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'Baseline ({BASELINE_AVG_STEPS} steps)')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=60, ha='right', fontsize=9)
    ax2.set_ylabel('Average denoising steps', fontsize=11)
    ax2.set_title('Phase B: avg denoising steps per sample (lower is better, but at the cost of acc)', fontsize=12)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig2_path = LOG_ROOT.parent.parent / 'figs' / 'phase_b_avg_steps.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {fig2_path}")
    
    plt.close('all')
    print("\nDone.")


if __name__ == "__main__":
    main()
