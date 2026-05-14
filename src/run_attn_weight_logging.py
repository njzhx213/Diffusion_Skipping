"""
Logging attention weight distribution for 1c-ii motivation plot.

Uses modeling.py's _compute_attention_stats method (added via patch).
We attach an AttnWeightHistogramCollector to every self_attn module,
and toggle it on/off around each denoising step via the manager lifecycle.
"""
import os
import sys
import time
import types
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions
from step_cache import StepCacheManager, AttnWeightHistogramCollector

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"
DUMP_PATH = "logs/attn_weight_100/attn_weight_hist.npz"
N_SAMPLES = 100

os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

# === Set up collector and attach to every attention module ===
collector = AttnWeightHistogramCollector(num_layers=28, dump_path=DUMP_PATH)
for layer in model.model.layers:
    layer.self_attn.stats_collector = collector
print(f"Attached collector to {len(model.model.layers)} attention modules")

# Dummy manager for lifecycle gating (toggles collector around denoising forwards)
manager = StepCacheManager(num_layers=28, policy=None, dump_dir=None)

# Monkey-patch manager to toggle collector
_orig_start_step = manager.start_new_step
_orig_finalize_step = manager.finalize_step
def start_step_with_aw():
    collector.enabled = True
    _orig_start_step()
def finalize_step_with_aw():
    _orig_finalize_step()
    collector.enabled = False
manager.start_new_step = start_step_with_aw
manager.finalize_step = finalize_step_with_aw

model.step_cache_manager = manager

# === Load dataset ===
print("Loading GSM8K...")
ds = load_dataset("openai/gsm8k", "main", split="test")
print(f"  Using first {N_SAMPLES}")

# === Run ===
t_start = time.time()
total_tokens = 0
n_failed = 0

for sample_idx in range(N_SAMPLES):
    try:
        question = ds[sample_idx]["question"]
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

        model.current_sample_id = sample_idx

        t0 = time.time()
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
        n_new = (generated[0][min_len:]).shape[0]
        total_tokens += n_new
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{sample_idx+1:3d}/{N_SAMPLES}] gen={n_new:3d} "
              f"time={elapsed:5.1f}s ({n_new/elapsed:5.1f} tok/s) "
              f"vram={peak_vram:.2f}GB hist_records={collector.total_records}",
              flush=True)

        if (sample_idx + 1) % 10 == 0:
            t_so_far = time.time() - t_start
            eta = t_so_far / (sample_idx + 1) * (N_SAMPLES - sample_idx - 1)
            print(f"  --- {sample_idx+1}/{N_SAMPLES} done, "
                  f"elapsed={t_so_far/60:.1f}min, ETA={eta/60:.1f}min",
                  flush=True)

    except Exception as e:
        n_failed += 1
        print(f"[{sample_idx+1:3d}] FAILED: {e}", flush=True)
        torch.cuda.empty_cache()
        continue

t_total = time.time() - t_start
print(f"\n{'='*70}")
print(f"DONE. Total: {t_total/60:.1f} min")
print(f"Tokens: {total_tokens}, avg speed: {total_tokens/t_total:.1f} tok/s")
print(f"Histogram records (hook fires): {collector.total_records}")
print(f"Per-layer total counts: "
      f"min={collector.counts.sum(axis=1).min()}, "
      f"max={collector.counts.sum(axis=1).max()}")
print(f"  Below min bin total: {collector.below_min.sum()}")
print(f"  Above max bin total: {collector.above_max.sum()}")

collector.dump()
print(f"\nDumped histogram to: {DUMP_PATH}")
