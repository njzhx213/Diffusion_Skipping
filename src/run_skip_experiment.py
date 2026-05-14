"""
Phase B: Run compute-skipping experiment for all 17 settings (excluding baseline).

For each setting:
  - 100 GSM8K test samples
  - Track: accuracy, avg denoising steps, per (step, layer) reused counts
  - Dump per-sample skip stats to logs/skip_exp/<setting_name>/
  - Resume support: skip settings with done.flag

Usage:
  python run_skip_experiment.py                  # run all settings
  python run_skip_experiment.py --setting NAME   # run only one setting
  python run_skip_experiment.py --list           # list all settings
  python run_skip_experiment.py --redo           # ignore done.flag, redo all
"""
from __future__ import annotations
import sys, time, types, os, re, json, argparse, traceback
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions
from step_cache import StepCacheManager, HookSession
from policies import make_policy, ALL_SETTINGS

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"
LOG_ROOT = Path("/home/njzhx/Fast-dLLM/v2/logs/skip_exp")
N_SAMPLES = 100
MAX_NEW_TOKENS = 512


# -----------------------------------------------------------------------------
# Answer extraction (matches Phase A)
# -----------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract \\boxed{...} answer; fallback to last number in text."""
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        ans = m.group(1).strip()
        # strip dollar signs and commas
        ans = ans.replace('$', '').replace(',', '').strip()
        return ans
    # Fallback: last number
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""


def normalize_answer(text: str) -> str:
    """Normalize for comparison: int if possible, else stripped.
    Robust to nan/inf and other malformed numbers."""
    s = str(text).strip().replace(',', '').replace('$', '')
    try:
        f = float(s)
        # NaN check (NaN != NaN)
        if f != f:
            return s
        # inf check
        if abs(f) == float('inf'):
            return s
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError, OverflowError):
        return s


def extract_gt(answer_text: str) -> str:
    """Extract gt from GSM8K answer field ('... #### N')."""
    return answer_text.split('####')[-1].strip()


# -----------------------------------------------------------------------------
# Single setting run
# -----------------------------------------------------------------------------

def run_one_setting(model, tokenizer, ds, spec: dict, redo: bool = False) -> dict:
    """Run a single setting on N_SAMPLES samples."""
    policy = make_policy(spec)
    setting_name = policy.name
    setting_dir = LOG_ROOT / setting_name
    done_flag = setting_dir / "done.flag"
    summary_path = setting_dir / "summary.json"

    print(f"\n{'='*70}")
    print(f"Setting: {setting_name}")
    print(f"Policy: {policy}")
    print(f"Output: {setting_dir}")
    print(f"{'='*70}")

    if done_flag.exists() and not redo:
        # Load existing summary and return
        if summary_path.exists():
            with open(summary_path) as f:
                prev = json.load(f)
            print(f"  ✓ Already done. acc={prev.get('accuracy', '?'):.1f}%, "
                  f"avg_steps={prev.get('avg_steps', '?'):.1f}")
            return prev
        print(f"  Already done (no summary). Skipping.")
        return {'setting_name': setting_name, 'status': 'skipped (done.flag exists)'}

    setting_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample tracking
    n_correct = 0
    total_steps = 0
    total_tokens = 0
    total_reused = 0
    total_decisions = 0

    t_start = time.time()
    for sample_id in range(N_SAMPLES):
        question = ds[sample_id]['question'] + (
            " Please reason step by step, and put your final answer within \\boxed{}."
        )
        gt = extract_gt(ds[sample_id]['answer'])
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
        min_len = input_ids.shape[1]

        # Fresh manager per sample (resets all state)
        manager = StepCacheManager(num_layers=28, policy=policy,
                                    dump_dir=str(setting_dir))
        model.step_cache_manager = manager
        model.current_sample_id = sample_id

        try:
            manager.start_new_sample(sample_id)
            t0 = time.time()
            with HookSession(model, manager):
                with torch.no_grad():
                    generated = model.mdm_sample(
                        input_ids=input_ids, tokenizer=tokenizer,
                        block_size=32, small_block_size=8,
                        max_new_tokens=MAX_NEW_TOKENS, mask_id=151665,
                        min_len=min_len, seq_len=seq_len,
                        use_block_cache=False, threshold=1.0,
                    )
            elapsed = time.time() - t0
            n_new = generated[0][min_len:].shape[0]
            answer = tokenizer.decode(generated[0][min_len:], skip_special_tokens=True)
            pred = extract_answer(answer)
            correct = (normalize_answer(pred) == normalize_answer(gt))

            # Per-sample stats
            steps = manager.total_steps_this_sample
            reused = sum(manager.skipped_per_step_layer.values())
            decisions = len(manager.skipped_per_step_layer)

            # Dump skip_stats_NNNN.npz with sample-level info
            _dump_with_extras(manager, sample_id, n_new, correct, pred, gt)

            n_correct += int(correct)
            total_steps += steps
            total_tokens += n_new
            total_reused += reused
            total_decisions += decisions

            mark = '✓' if correct else '✗'
            print(f"  [{sample_id+1:3d}/{N_SAMPLES}] gen={n_new:3d} steps={steps:3d} "
                  f"reused={reused:6d} time={elapsed:5.1f}s ({n_new/elapsed:4.1f} tok/s) "
                  f"{mark} gt={gt} pred={pred}")

        except Exception as e:
            print(f"  [{sample_id+1:3d}/{N_SAMPLES}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Continue with next sample
            continue

        # Free GPU mem
        del manager
        torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start

    # Aggregate
    accuracy = 100.0 * n_correct / N_SAMPLES
    avg_steps = total_steps / N_SAMPLES if N_SAMPLES > 0 else 0
    avg_tokens = total_tokens / N_SAMPLES if N_SAMPLES > 0 else 0
    avg_reused = total_reused / N_SAMPLES if N_SAMPLES > 0 else 0
    reuse_rate = (total_reused / (total_decisions * 32)) if total_decisions > 0 else 0

    summary = {
        'setting_name': setting_name,
        'spec': spec,
        'n_samples': N_SAMPLES,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'avg_steps': avg_steps,
        'avg_tokens': avg_tokens,
        'avg_reused_per_sample': avg_reused,
        'total_reused': total_reused,
        'total_decisions': total_decisions,
        'reuse_rate': reuse_rate,
        'elapsed_seconds': elapsed_total,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    done_flag.touch()

    print(f"\n  → {setting_name}: acc={accuracy:.1f}%, avg_steps={avg_steps:.1f}, "
          f"reuse_rate={reuse_rate:.1%}, time={elapsed_total/60:.1f}min")
    return summary


def _dump_with_extras(manager, sample_id, n_tokens, correct, pred, gt):
    """Extend the basic dump with sample-level metadata."""
    if manager.dump_dir is None:
        return
    items = sorted(manager.skipped_per_step_layer.items())
    if items:
        step_ids = np.array([k[0] for k, _ in items], dtype=np.int32)
        layer_ids = np.array([k[1] for k, _ in items], dtype=np.int32)
        reused_counts = np.array([v for _, v in items], dtype=np.int32)
    else:
        step_ids = np.zeros(0, dtype=np.int32)
        layer_ids = np.zeros(0, dtype=np.int32)
        reused_counts = np.zeros(0, dtype=np.int32)
    path = os.path.join(manager.dump_dir, f'skip_stats_{sample_id:04d}.npz')
    np.savez_compressed(
        path,
        sample_id=np.array([sample_id], dtype=np.int32),
        total_steps=np.array([manager.total_steps_this_sample], dtype=np.int32),
        n_tokens_generated=np.array([n_tokens], dtype=np.int32),
        correct=np.array([correct], dtype=bool),
        pred_answer=np.array([str(pred)]),
        gt_answer=np.array([str(gt)]),
        step_ids=step_ids,
        layer_ids=layer_ids,
        reused_counts=reused_counts,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default=None,
                        help="Run only this setting name (e.g. 'token_cossim_0.99')")
    parser.add_argument("--list", action="store_true",
                        help="List all settings and exit")
    parser.add_argument("--redo", action="store_true",
                        help="Ignore done.flag, redo all settings")
    args = parser.parse_args()

    # Settings to run (skip baseline; Phase A already has this)
    all_settings = ALL_SETTINGS[1:]  # excludes baseline

    if args.list:
        print("Settings:")
        for i, spec in enumerate(all_settings):
            policy = make_policy(spec)
            print(f"  {i+1:2d}. {policy.name}  ({spec})")
        return

    if args.setting:
        filtered = [s for s in all_settings if make_policy(s).name == args.setting]
        if not filtered:
            print(f"No setting named {args.setting!r}. Use --list to see all.")
            return
        all_settings = filtered

    print(f"Will run {len(all_settings)} settings × {N_SAMPLES} samples each.")
    print(f"Log root: {LOG_ROOT}")
    print()

    # Load model + dataset once
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
    )

    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  {len(ds)} samples available; using first {N_SAMPLES}")

    # Run all
    all_results = []
    t_master = time.time()
    for spec in all_settings:
        try:
            summary = run_one_setting(model, tokenizer, ds, spec, redo=args.redo)
            all_results.append(summary)
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping. Resume by re-running same command.")
            break
        except Exception as e:
            print(f"FATAL error in setting {spec}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_master
    print(f"\n{'='*70}")
    print(f"DONE. Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"{'='*70}")
    print(f"\n{'Setting':<35s} {'Acc%':>6s} {'Steps':>7s} {'Reuse%':>7s} {'Time':>7s}")
    print("-" * 70)
    for r in all_results:
        if 'accuracy' not in r:
            continue
        print(f"  {r['setting_name']:<33s} "
              f"{r['accuracy']:>5.1f}% {r['avg_steps']:>7.1f} "
              f"{r['reuse_rate']*100:>6.1f}% "
              f"{r['elapsed_seconds']/60:>6.1f}m")


if __name__ == "__main__":
    main()
