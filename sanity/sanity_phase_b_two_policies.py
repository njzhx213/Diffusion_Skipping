"""
Phase B sanity #3 + #4:
  - TokenCossimPolicy(threshold=0.99)
  - LayerCossimPolicy(threshold=0.99, agg='avg')

Both runs use sample 0 (Janet, gt=18).
Compares skip rates to expectations from Phase A statistics:
  - sim mean ≈ 0.977, fraction > 0.99 ≈ 43% (Phase A H_in)
  - => TokenCossim 0.99 should skip ~43% tokens
  - => LayerCossim avg 0.99: per-layer mean is usually < 0.99 (mean=0.977),
       so 很少 layer 会被整层跳; expected very low skip rate
"""
import sys, time, types
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions
from step_cache import StepCacheManager, HookSession
from policies import TokenCossimPolicy, LayerCossimPolicy

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"

print("Loading model (once for two tests)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

# === Load dataset (once) ===
print("Loading GSM8K sample 0...")
ds = load_dataset("openai/gsm8k", "main", split="test")
question = ds[0]["question"] + (
    " Please reason step by step, and put your final answer within \\boxed{}."
)
messages = [{"role": "user", "content": question}]
prompt_text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
min_len = input_ids.shape[1]


def run_one(policy_name, policy):
    """Run sample 0 with the given policy, print stats."""
    print(f"\n{'='*70}")
    print(f"Test: {policy_name}")
    print(f"Policy: {policy}")
    print(f"{'='*70}")
    
    manager = StepCacheManager(num_layers=28, policy=policy, dump_dir=None)
    model.step_cache_manager = manager
    model.current_sample_id = 0
    
    manager.start_new_sample(0)
    t0 = time.time()
    with HookSession(model, manager):
        with torch.no_grad():
            generated = model.mdm_sample(
                input_ids=input_ids, tokenizer=tokenizer,
                block_size=32, small_block_size=8,
                max_new_tokens=512, mask_id=151665,
                min_len=min_len, seq_len=seq_len,
                use_block_cache=False, threshold=1.0,
            )
    elapsed = time.time() - t0
    n_new = generated[0][min_len:].shape[0]
    answer = tokenizer.decode(generated[0][min_len:], skip_special_tokens=True)
    
    # Extract answer
    import re
    m = re.search(r'\\boxed\{(\d+)\}', answer)
    pred = m.group(1) if m else "NOT FOUND"
    
    # Skip stats
    counts_arr = np.array(list(manager.skipped_per_step_layer.values()))
    n_decisions = len(counts_arr)
    total_skipped = int(counts_arr.sum())
    mean_skipped = float(counts_arr.mean()) if len(counts_arr) > 0 else 0
    nonzero = int((counts_arr > 0).sum())
    
    print(f"\n  Time: {elapsed:.1f}s, tokens generated: {n_new}, speed: {n_new/elapsed:.1f} tok/s")
    print(f"  Total denoising steps: {manager.total_steps_this_sample}")
    print(f"  Answer: \\boxed{{{pred}}}  (gt=18)  {'✅' if pred == '18' else '❌'}")
    print()
    print(f"  Skip stats:")
    print(f"    Total decisions:      {n_decisions}")
    print(f"    Total tokens skipped: {total_skipped}")
    print(f"    Decisions w/ skip>0:  {nonzero}/{n_decisions}")
    print(f"    Mean skipped/decision:{mean_skipped:.1f}")
    print(f"    Skip count range:     [{counts_arr.min() if len(counts_arr)>0 else 0}, "
          f"{counts_arr.max() if len(counts_arr)>0 else 0}]")
    
    return {
        'policy_name': policy_name,
        'time': elapsed,
        'tokens': n_new,
        'steps': manager.total_steps_this_sample,
        'answer': pred,
        'total_decisions': n_decisions,
        'total_skipped': total_skipped,
        'mean_skipped': mean_skipped,
    }


# === Run two tests ===
results = []
results.append(run_one("TokenCossim(0.99)", TokenCossimPolicy(threshold=0.99)))
results.append(run_one("LayerCossim(0.99, avg)", LayerCossimPolicy(threshold=0.99, agg='avg')))

# === Summary table ===
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Policy':<35s} {'Steps':>6s} {'Skipped':>10s} {'Mean/dec':>10s} {'Answer':>8s}")
print("-"*70)
for r in results:
    print(f"  {r['policy_name']:<33s} {r['steps']:>6d} {r['total_skipped']:>10d} "
          f"{r['mean_skipped']:>10.1f} {r['answer']:>8s}")

print()
print("Compare to:")
print(f"  Baseline (NoSkip):                246 steps, 0 skipped, answer=18")
print(f"  TopK(k=25):                       277 steps, 176577 skipped, answer=18")
print(f"  Expected TokenCossim(0.99): ~43% skip rate (from Phase A stats)")
print(f"  Expected LayerCossim avg 0.99: very low skip rate "
      f"(mean sim < 0.99 in most layers)")
