"""
Motivation logging on 100 GSM8K samples.

Each sample's sim records are dumped to logs/motivation_100/sample_NNNN.npz
immediately after generation, so a crash never loses past progress.
Already-dumped samples are skipped (resumable).
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
from step_cache import StepCacheManager, HookSession

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"
DUMP_DIR = "logs/motivation_100"
N_SAMPLES = 100

os.makedirs(DUMP_DIR, exist_ok=True)

print(f"Output dir: {DUMP_DIR}")
print(f"Target: {N_SAMPLES} samples")

# === Check for already-dumped samples (resumable) ===
done = set()
for f in os.listdir(DUMP_DIR):
    if f.startswith('sample_') and f.endswith('.npz'):
        idx = int(f.split('_')[1].split('.')[0])
        done.add(idx)
if done:
    print(f"Resume mode: {len(done)} samples already dumped, skipping them")

# === Load dataset ===
print("Loading GSM8K test set...")
ds = load_dataset("openai/gsm8k", "main", split="test")
print(f"  Available: {len(ds)} questions; using first {N_SAMPLES}")

# === Load model ===
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

manager = StepCacheManager(num_layers=28, policy=None, dump_dir=DUMP_DIR)
model.step_cache_manager = manager

# === Run ===
t_run_start = time.time()
total_tokens = 0
n_correct = 0
n_failed = 0
times = []

with HookSession(model, manager):
    for sample_idx in range(N_SAMPLES):
        # Skip if already done
        if sample_idx in done:
            continue

        try:
            question = ds[sample_idx]["question"]
            gt_answer = ds[sample_idx]["answer"]  # GSM8K provides #### N at the end
            # extract gt number for checking
            try:
                gt_num = gt_answer.split('####')[-1].strip().replace(',', '')
            except Exception:
                gt_num = None

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
            times.append(elapsed)

            new_tokens = generated[0][min_len:]
            n_new = len(new_tokens)
            total_tokens += n_new
            answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Quick correctness check: does the boxed answer contain gt_num?
            import re
            match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
            pred_num = None
            if match:
                pred_str = match.group(1).strip().replace(',', '').replace('$', '')
                # extract first number
                m2 = re.search(r'-?\d+\.?\d*', pred_str)
                if m2:
                    pred_num = m2.group(0)
            is_correct = (pred_num is not None and gt_num is not None
                          and pred_num == gt_num)
            if is_correct:
                n_correct += 1

            # Light progress line every sample
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            status = "✓" if is_correct else "✗"
            print(f"[{sample_idx+1:3d}/{N_SAMPLES}] gen={n_new:3d} "
                  f"time={elapsed:5.1f}s ({n_new/elapsed:5.1f} tok/s) "
                  f"vram={peak_vram:.2f}GB {status} "
                  f"gt={gt_num} pred={pred_num}", flush=True)

            # Mid-run summary every 10
            if (sample_idx + 1) % 10 == 0:
                t_so_far = time.time() - t_run_start
                done_so_far = sample_idx + 1 - len(done)  # newly done
                if done_so_far > 0:
                    eta = t_so_far / done_so_far * (N_SAMPLES - sample_idx - 1)
                    print(f"  --- {sample_idx+1}/{N_SAMPLES} done, "
                          f"elapsed={t_so_far/60:.1f}min, "
                          f"avg={t_so_far/done_so_far:.1f}s/sample, "
                          f"ETA={eta/60:.1f}min, "
                          f"correct so far: {n_correct}/{sample_idx+1}", flush=True)

        except Exception as e:
            n_failed += 1
            print(f"[{sample_idx+1:3d}/{N_SAMPLES}] FAILED: {e}", flush=True)
            torch.cuda.empty_cache()
            continue

# === Final summary ===
t_total = time.time() - t_run_start
n_attempted = N_SAMPLES - len(done)
print(f"\n{'='*70}")
print(f"DONE.")
print(f"Total time: {t_total/60:.1f} min ({t_total:.1f}s)")
print(f"Samples attempted (this run): {n_attempted}")
print(f"Total tokens generated: {total_tokens}")
if n_attempted > 0:
    print(f"Average speed: {total_tokens / t_total:.1f} tok/s")
    print(f"Average time per sample: {t_total / n_attempted:.1f}s")
print(f"Correctness: {n_correct}/{N_SAMPLES - n_failed - len(done)} "
      f"(baseline accuracy)")
print(f"Failed: {n_failed}")

# Count final dumped files
all_files = [f for f in os.listdir(DUMP_DIR) if f.endswith('.npz')]
total_size = sum(os.path.getsize(os.path.join(DUMP_DIR, f))
                 for f in all_files) / 1e6
print(f"\nDumped: {len(all_files)} npz files, total {total_size:.0f} MB")
print(f"  Avg per sample: {total_size / max(1, len(all_files)):.1f} MB")
