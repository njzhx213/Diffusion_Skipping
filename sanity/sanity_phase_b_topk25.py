"""
Phase B sanity check #2: TokenTopKPolicy(k=25) should actually skip tokens.

Runs sample 0 of GSM8K with TopK k=25 → 75% of tokens get reused each step.

Expected:
  - skipped_per_step_layer total >> 0 (in fact 75% of all decisions)
  - skipped count after first step of each block ≈ 0.75 × num_tokens × 28 layers
  - Generation might be slower or faster (depends on whether we save time)
  - Answer might be WRONG (because aggressive skipping degrades quality)
  
  The key test: skipping is happening, hooks are modifying outputs.
"""
import sys, time, types
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions
from step_cache import StepCacheManager, HookSession
from policies import TokenTopKPolicy

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

print("Setting up manager with TokenTopKPolicy(k=25)...")
policy = TokenTopKPolicy(k_percent=25)
manager = StepCacheManager(num_layers=28, policy=policy, dump_dir=None)
model.step_cache_manager = manager
print(f"Policy: {manager.policy}")

print("Loading GSM8K sample 0 (Janet ducks)...")
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
model.current_sample_id = 0

print()
print("Generating with TopK(k=25) skip...")
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
print(f"\nDone in {elapsed:.1f}s, {n_new} tokens, {n_new/elapsed:.1f} tok/s")

answer = tokenizer.decode(generated[0][min_len:], skip_special_tokens=True)
print(f"\nAnswer (first 400 chars):")
print(answer[:400])

# === Sanity checks ===
print("\n" + "="*60)
print("SANITY CHECKS (TopK k=25 should skip ~75% of decisions):")
print("="*60)

total_skipped = sum(manager.skipped_per_step_layer.values())
n_decisions = len(manager.skipped_per_step_layer)
n_entries = n_decisions

# Skipped per layer-step
# Average skipped tokens per (step, layer) where decision made
import numpy as np
counts_arr = np.array(list(manager.skipped_per_step_layer.values()))
nonzero_count = (counts_arr > 0).sum()
mean_skipped = counts_arr.mean() if len(counts_arr) > 0 else 0
max_skipped = counts_arr.max() if len(counts_arr) > 0 else 0
min_skipped = counts_arr.min() if len(counts_arr) > 0 else 0

print(f"\n1. Total decisions made (step, layer): {n_decisions}")
print(f"2. Total skipped tokens: {total_skipped}")
print(f"3. Mean skipped per decision: {mean_skipped:.1f}")
print(f"4. Skipped count range: [{min_skipped}, {max_skipped}]")
print(f"5. Decisions with skipped > 0: {nonzero_count}/{n_decisions}")
print(f"6. total_steps_this_sample: {manager.total_steps_this_sample}")
print()
# k=25 means skip top 75% sim, so 24/32 ≈ 24 token per decision (if all tokens decided)
expected_per_dec = 32 * 0.75  # 24
print(f"Expected skipped per decision: ~{expected_per_dec} (if 32 tokens, skip 75%)")
print()

# Check skip rate
if mean_skipped > 10:
    print("✅ Mean skipped > 10 → skipping IS happening")
else:
    print(f"⚠️ Mean skipped = {mean_skipped} → skipping not as expected")

import re
m = re.search(r'\\boxed\{(\d+)\}', answer)
pred = m.group(1) if m else "NOT FOUND"
print(f"\nAnswer prediction: \\boxed{{{pred}}}  (Phase A baseline: 18)")
if pred == "18":
    print("  Surprising: answer still correct despite 75% skip")
else:
    print("  Expected: aggressive skip degrades answer quality")
