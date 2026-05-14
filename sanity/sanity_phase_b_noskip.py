"""
Phase B sanity check #1: NoSkipPolicy should behave identically to baseline.

Runs sample 0 of GSM8K with full Phase B infrastructure (manager + hooks)
but uses NoSkipPolicy. Expected:
  - Answer = 72 (correct)
  - Speed ≈ 6 tok/s (same as Phase A motivation logging)
  - skipped_per_step_layer counts = all 0
"""
import sys, time, types
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions
from step_cache import StepCacheManager, HookSession
from policies import NoSkipPolicy

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

print("Setting up manager with NoSkipPolicy...")
policy = NoSkipPolicy()
manager = StepCacheManager(num_layers=28, policy=policy, dump_dir=None)
model.step_cache_manager = manager
print(f"Policy: {manager.policy}")

print("Loading GSM8K sample 0...")
ds = load_dataset("openai/gsm8k", "main", split="test")
question = ds[0]["question"]
question_text = (
    question + " Please reason step by step, and put your final answer "
    "within \\boxed{}."
)
messages = [{"role": "user", "content": question_text}]
prompt_text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
min_len = input_ids.shape[1]
model.current_sample_id = 0

print(f"Prompt: {input_ids.shape[1]} tokens")
print()
print("Generating with hooks attached (NoSkip)...")

manager.start_new_sample(0)
t0 = time.time()
with HookSession(model, manager):
    with torch.no_grad():
        generated = model.mdm_sample(
            input_ids=input_ids,
            tokenizer=tokenizer,
            block_size=32,
            small_block_size=8,
            max_new_tokens=512,
            mask_id=151665,
            min_len=min_len,
            seq_len=seq_len,
            use_block_cache=False,
            threshold=1.0,
        )
elapsed = time.time() - t0
n_new = generated[0][min_len:].shape[0]
print(f"\nGeneration done in {elapsed:.1f}s, {n_new} tokens, {n_new/elapsed:.1f} tok/s")

answer = tokenizer.decode(generated[0][min_len:], skip_special_tokens=True)
print(f"\nAnswer (first 600 chars):\n{answer[:600]}")

# === Sanity checks ===
print("\n" + "="*60)
print("SANITY CHECKS:")
print("="*60)

# 1. Answer contains "\\boxed{72}"
import re
m = re.search(r'\\boxed\{(\d+)\}', answer)
ans = m.group(1) if m else "NOT FOUND"
correct = (ans == "72")
print(f"1. Answer: \\boxed{{{ans}}}  (expected 72) → {'✓' if correct else '✗ FAIL'}")

# 2. skipped_per_step_layer should be all 0 (NoSkip)
total_skipped = sum(manager.skipped_per_step_layer.values())
n_entries = len(manager.skipped_per_step_layer)
print(f"2. skipped_per_step_layer entries: {n_entries}, total skipped: {total_skipped}")
print(f"   (expected: total = 0 because NoSkipPolicy) → {'✓' if total_skipped == 0 else '✗ FAIL'}")

# 3. total_steps_this_sample > 0
print(f"3. total_steps_this_sample: {manager.total_steps_this_sample}  → "
      f"{'✓' if manager.total_steps_this_sample > 0 else '✗ FAIL'}")

# 4. Records collected
n_records = len(manager.records)
print(f"4. records collected: {n_records}  (expected ~6000)")

# 5. prev_attn_out_cache filled
n_attn_cache = len(manager.prev_attn_out_cache)
print(f"5. prev_attn_out_cache: {n_attn_cache} layers cached (expected 28)")

# 6. skip_mask: should be all-False or None
all_no_skip = True
for layer_id, mask in manager.skip_mask.items():
    if mask is not None and mask.any():
        all_no_skip = False
        break
print(f"6. All skip_mask are None or all-False: "
      f"{'✓' if all_no_skip else '✗ FAIL'}")

print()
print(f"OVERALL: {'✅ PASS' if correct and total_skipped == 0 else '❌ FAIL'}")
