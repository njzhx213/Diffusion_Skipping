"""
Minimal sanity check: 跑通 mdm_sample 走完 1 个 GSM8K 样本。
不走 lm-eval 框架，直接 import 模型 + 调 mdm_sample。
目标：验证环境 + 拿到真实的 step/block 数字 + 看显存峰值。
"""
import sys
import time
import types
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 让 Python 能找到 Fast-dLLM/v2 目录下的 generation_functions.py
sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")
import generation_functions

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"

# === 一道随机 GSM8K 题 ===
QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May? "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model (bf16, ~15GB)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")
model.eval()
print(f"  Model loaded in {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 绑定 mdm_sample
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

# === 模拟 eval.py 的 prompt 处理流程 ===
# eval.py 用了 apply_chat_template，我们这里也走一遍保持一致
messages = [{"role": "user", "content": QUESTION}]
prompt_text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
print(f"\nPrompt (after chat template):\n{'-'*60}\n{prompt_text}\n{'-'*60}")

inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
min_len = input_ids.shape[1]
print(f"Input length: {min_len} tokens")

# === 调 mdm_sample ===
# 跟 eval_script.sh 的 GSM8K 配置对齐：threshold=1, bd_size=32, small_block_size=8
print(f"\nRunning mdm_sample (threshold=1.0, block_size=32, small_block_size=8)...")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    generated = model.mdm_sample(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=32,
        small_block_size=8,
        max_new_tokens=512,        # 短一点先验证管线
        mask_id=151665,
        min_len=min_len,
        seq_len=seq_len,
        use_block_cache=False,
        threshold=1.0,             # baseline 设置
    )
elapsed = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / 1e9

# generated 是 dict: {original_idx: tensor}
output_ids = generated[0]
new_tokens = output_ids[min_len:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"DONE in {elapsed:.1f}s")
print(f"Peak VRAM: {peak_vram:.2f} GB")
print(f"Generated tokens: {len(new_tokens)}")
print(f"Throughput: {len(new_tokens)/elapsed:.1f} tok/s")
print(f"{'='*60}")
print(f"\nAnswer:\n{'-'*60}\n{answer}\n{'-'*60}")
