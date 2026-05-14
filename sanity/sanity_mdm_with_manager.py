"""
Sanity check: run 1 GSM8K sample through mdm_sample with StepCacheManager
attached. Goal: verify the integrated logging produces sane sim records.
"""
import sys
import time
import types
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions
from step_cache import StepCacheManager, HookSession

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"

QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May? "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

print("Loading tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

# Bind mdm_sample
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

# === Create manager (logging-only) ===
manager = StepCacheManager(
    num_layers=28,
    policy=None,
    dump_dir="logs/motivation_test",  # don't dump for this sanity check; we'll inspect in-memory
)
model.step_cache_manager = manager
model.current_sample_id = 0  # so manager picks it up at start_new_sample

# === Prepare prompt ===
messages = [{"role": "user", "content": QUESTION}]
prompt_text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
min_len = input_ids.shape[1]
print(f"Input length: {min_len} tokens")

# === Run with HookSession ===
print("\nRunning mdm_sample with logging hooks attached...")
torch.cuda.reset_peak_memory_stats()
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
peak_vram = torch.cuda.max_memory_allocated() / 1e9

# === Inspect ===
new_tokens = generated[0][min_len:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"DONE in {elapsed:.1f}s")
print(f"Peak VRAM: {peak_vram:.2f} GB")
print(f"Generated {len(new_tokens)} new tokens at {len(new_tokens)/elapsed:.1f} tok/s")
print(f"\nAnswer (last 200 chars):\n{answer[-200:]}")

# === Inspect manager state ===
print(f"\n{'='*60}")
print(f"=== Manager logging summary ===")
print(f"Total records collected: {len(manager.records)}")
print(f"Global step count: {manager.step_id_global + 1}")
print(f"Final block id: {manager.block_id}")

if len(manager.records) > 0:
    # Distribution of records across (block, step)
    from collections import Counter
    block_counter = Counter(r.block_id for r in manager.records)
    print(f"\nRecords per block (each block produces num_layers x num_recorded_steps):")
    for bid in sorted(block_counter):
        print(f"  Block {bid}: {block_counter[bid]} records  "
              f"({block_counter[bid] // 28} steps × 28 layers)")
    
    # Sim statistics
    import numpy as np
    all_sim_H_in = np.concatenate(
        [r.sim_H_in.flatten() for r in manager.records if r.sim_H_in is not None]
    )
    all_sim_attn = np.concatenate(
        [r.sim_attn_out.flatten() for r in manager.records if r.sim_attn_out is not None]
    )
    print(f"\nSimilarity statistics (all (step, layer, token) datapoints):")
    print(f"  sim_H_in:     mean={all_sim_H_in.mean():.4f}, "
          f"median={np.median(all_sim_H_in):.4f}, "
          f"min={all_sim_H_in.min():.4f}, max={all_sim_H_in.max():.4f}")
    print(f"  sim_attn_out: mean={all_sim_attn.mean():.4f}, "
          f"median={np.median(all_sim_attn):.4f}, "
          f"min={all_sim_attn.min():.4f}, max={all_sim_attn.max():.4f}")
    
    # Fraction above each threshold (relevant to task)
    print(f"\nFraction of sim_H_in > threshold (per-token):")
    for thr in [0.96, 0.97, 0.98, 0.99, 0.995]:
        frac = (all_sim_H_in > thr).mean()
        print(f"  > {thr}: {frac:.2%}")

print("\n✅ Done.")
