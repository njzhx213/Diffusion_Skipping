# Phase B: Compute-Skipping Policies for Fast-dLLM v2

## 1. Implementation Overview

We implement two compute-skipping policies on the Fast-dLLM v2 (Qwen2.5-7B-Instruct fine-tune) model, evaluated on the first 100 GSM8K test samples with `block_size=32`, `small_block_size=8`, `max_new_tokens=512`, and `threshold=1.0`.

**Hook-based implementation.** Rather than directly modifying the model's `forward()` in `modeling.py`, we attach PyTorch forward hooks in `step_cache.py` that capture intermediate tensors and conditionally substitute attention/MLP outputs with previous-step caches. This satisfies task spec 3.c ("Modify [modeling.py] **and other related files**") while decoupling the skipping logic from model architecture, enabling cleaner code and easier extension. The single change to `modeling.py` is a `_compute_attention_stats` method used only for Phase A's 1c-ii motivation plot.

**Choice of similarity input (spec 3.d).** The task spec requires that all similarities be computed on hidden states *after* layer normalization. Concretely, we use `H_in = layer.input_layernorm(hidden_states)` — the post-LN tensor that is the immediate input to the attention module — as the comparison point between consecutive denoising steps. We chose `H_in` (rather than the raw pre-LN `hidden_states`, or the mid-layer `H_mid = post_attention_layernorm(...)`) for three reasons. First, `H_in` is the actual input to the attention and MLP modules we are deciding to skip, so similarity there is a direct proxy for "will the output be similar." Second, LayerNorm removes per-token magnitude variation, making cosine similarity a meaningful angular measure rather than a quantity dominated by activation scale. Third, this choice is consistent with Phase A's motivation analysis (Figs 1a-i through 1a-iii), which used the same `H_in` similarity to characterize cross-step redundancy — so Phase B's decision criterion is grounded in the same signal that motivated the work.

**Policy module (`policies.py`).** Four `SkipPolicy` classes encode the four touch points required by the task spec:

| Class | Decision granularity | Trigger |
|---|---|---|
| `NoSkipPolicy` | — | baseline, never skip |
| `TokenCossimPolicy` | per-token | `sim > threshold` |
| `TokenTopKPolicy` | per-token | keep top-k% most-changed; skip rest |
| `LayerCossimPolicy` | per-layer | `agg(sim) > threshold`, `agg ∈ {avg, max}` |

Each policy takes per-token cosine similarity `sim ∈ ℝ^{B×L}` and returns a boolean mask of the same shape. For layer-level decisions, the mask is constant across token positions; for token-level decisions, individual positions may be skipped or computed. The attention and MLP hooks of layer L share the same mask `manager.skip_mask[L]` (spec 1.c).

**Total settings: 18.** Baseline (1) + Token-level cossim (5 thresholds: 0.96–0.995) + Token-level TopK (k = 25, 50) + Layer-level avg (5 thresholds: 0.97–0.999) + Layer-level max (5 thresholds: 0.97–0.999). The threshold ranges are calibrated against Phase A statistics: the global mean of per-token `sim_H_in` across all (sample, step, layer) tuples is approximately 0.977, with 77% of values above 0.99. Token-level thresholds in 0.96–0.995 therefore span "lenient" (most tokens skipped) to "strict" (few skipped); layer-level thresholds shift higher (0.97–0.999) because per-layer aggregated similarity is itself biased upward (mean ≈ avg sim, max ≈ 1.0).

Spec 3.a (no skip on the first denoising step of each block) is enforced by the `is_first_step_in_block` flag. We verified this empirically on a sample with 4 block transitions: every block-first step had `skipped_count = 0` across all 28 layers, while every non-first step had nonzero skipping consistent with the policy. Separately, we confirmed that running `NoSkipPolicy` reproduces the Phase A baseline accuracy exactly (80/100 on the same sample IDs), validating that the hook infrastructure is transparent when no skipping is requested.

## 2. FLOPs Reduction Formula

We log per `(step, layer)` the **number of tokens reused** (spec 3.b) and compute **attention+MLP FLOPs reduction** relative to the baseline as:

> **FLOPs reduction = 1 − (avg_steps × (1 − reuse_rate)) / baseline_steps**

where `reuse_rate = total_reused / (total_decisions × block_size)` averaged over the 100 samples, and `avg_steps` / `baseline_steps` are the per-sample mean denoising step counts of the current setting and the baseline (299.6) respectively. Note that this measures savings only on the attention and MLP modules (the components affected by our skip mask); other forward-pass components (token embeddings, layernorms, residuals, language-modeling head) are not skipped and contribute additional fixed compute that we do not include here. Since attention+MLP accounts for the majority of per-layer FLOPs in this model, the relative numbers below approximate, but slightly overstate, end-to-end wall-clock savings. The formula accounts for the fact that aggressive reuse policies can **increase** the total denoising steps (the model needs more iterations to converge under a more constrained decoding path), so net FLOPs savings must offset this step-count overhead.

For example, `layer_cossim_avg_0.999` reuses only 0.1% of tokens but adds 10.5 extra steps per sample (310.1 vs. 299.6), resulting in a slightly **negative** FLOPs reduction of −3.4%. Conversely, `token_topk_50` reuses 49.1% with virtually no step overhead (299.8), yielding a clean 49.1% reduction. The per-token per-layer FLOPs cost (attention + MLP) for Qwen2.5-7B with `hidden_size=3584`, `intermediate_size=18944` is approximately 4.7×10⁸ FLOPs; we report relative reduction rather than absolute FLOPs to keep the comparison hardware-independent.

**Note on dump field naming.** The task spec (3.b) asks us to log the number of tokens *not reused* per `(step, layer)`. In our dump (`skip_stats_*.npz`), we instead store the complementary quantity `reused_counts` — the number of tokens whose attention/MLP outputs were substituted from the previous-step cache. The two are exactly complementary at any (step, layer): `not_reused_count = block_size − reused_count`. We chose to dump `reused_counts` because it is what `int(mask.sum())` returns directly inside the hook (one fewer subtraction per call), and converting at analysis time is trivial. All FLOPs reduction numbers in this report use the spec-defined `not_reused` semantics.

## 3. Key Observations

Table 1 (provided separately) gives the full numerical results for all 18 settings. The discussion below highlights the four most consequential findings.

### 3.1 Self-reinforcing token-level reuse

All five `token_cossim` thresholds (0.96, 0.97, 0.98, 0.99, 0.995) produce **virtually identical results** (accuracy 81–82%, reuse rate 96.9%, FLOPs reduction 96.7%). This is not noise; it reflects a self-reinforcing feedback loop:

> Once a token's attention/MLP outputs are reused at step *t*, its hidden state at step *t+1* is bitwise-identical to step *t*. The cross-step cosine similarity for that token is therefore **exactly 1.0** in the next iteration, triggering reuse again under any threshold ≤ 1.0.

We verified this empirically: under TokenCossim(0.99), 96.88% of recorded similarities equal 1.0 (median = 1.0000), versus 77% > 0.99 in the baseline distribution. The threshold becomes irrelevant once the loop initiates—all five settings converge to the same "skip 31 out of 32 tokens per step" steady state.

**Implication**: token-level cossim with a self-referential decision criterion cannot distinguish thresholds. Future work should compute similarity against a *counterfactual* (non-reused) hidden state.

### 3.2 Layer-level avg shows a clean trade-off curve

Layer-level averaging produces the most informative threshold sweep:

| Threshold | Acc % | Reuse % | FLOPs ↓ |
|---|---|---|---|
| 0.999 | 83.0 | 0.1 | −3.4 |
| 0.995 | 85.0 | 7.9 | 5.4 |
| 0.99 | 75.0 | 24.3 | 22.4 |
| 0.98 | 59.0 | 70.8 | 70.1 |
| 0.97 | 27.0 | 88.8 | 86.1 |

The 0.995 threshold is a **sweet spot**: it slightly *improves* accuracy over the baseline (85% vs. 80%) while reducing FLOPs by 5.4%. This monotone degradation is the expected shape of a Pareto frontier—Layer-avg is the policy of choice when one wants a tunable accuracy–compute trade-off.

### 3.3 Layer-level max is disqualified at block_size = 32

All five `layer_cossim_max` thresholds (0.97 through 0.999) collapse to the same degenerate behavior: **100% reuse rate, 0% accuracy**. With 32 tokens per block, `max(sim)` is almost surely ≈ 1.0 (one stable token suffices), so every threshold ≤ 1.0 triggers a whole-layer skip on every step after the first. The model effectively never recomputes—the output is garbage.

This is a fundamental limitation of the spec-defined max-aggregation rule at large block size, not a bug in our implementation. Future work could mitigate by computing max over a sub-block (e.g., 4-token windows) or by combining max with a "min reuse fraction" guard.

### 3.4 Token-TopK is the practical winner

Unlike threshold-based policies, TopK is immune to the self-reinforcing loop because it always keeps the bottom 25–50% of tokens (those with lowest similarity, i.e., the most-changed ones), forcing genuine computation on positions where the model is making progress:

- `token_topk_25`: 82% acc, 73% FLOPs reduction
- `token_topk_50`: **85% acc, 49.1% FLOPs reduction**—the best joint (acc, FLOPs) point in the entire sweep

The `token_topk_50` result is striking: a 5-point absolute accuracy gain over baseline while halving the attention/MLP compute. One plausible explanation is **implicit regularization**: forcing reuse on stable tokens may reduce noise in the diffusion sampling without losing the information needed for inference. We caution that with 100 samples, the 95% confidence interval on accuracy is approximately ±9 percentage points, so the 5-point absolute gain is suggestive rather than statistically conclusive at this scale; what is more robust is that `token_topk_50` does **not** degrade accuracy while reducing attention+MLP FLOPs by nearly half. This is the most practically useful setting from our sweep.

## 4. Limitations & Future Work

**Self-reinforcing.** As discussed in §3.1, threshold-based token-level reuse degenerates because the decision input is contaminated by past reuse. A clean fix is to maintain a parallel "vanilla forward" path (~2× compute during decision) to compare against, then drop the parallel path at inference time once thresholds are calibrated.

**Block-size dependence.** Our findings depend on `block_size=32`. Smaller blocks would change the per-token sim distribution (less self-reinforcing, less max-degeneracy) and likely yield a more discriminating sweep across the layer-max thresholds. A natural extension is to plot the same 18-setting sweep at `block_size ∈ {8, 16, 32, 64}`.

**Adaptive per-layer thresholds.** Phase A's motivation plots (1a-iii, 1c-ii) showed strong layer-wise heterogeneity in similarity dynamics: layers 0–3, 13, 16–20 stay near similarity 1.0, while layers 4–5, 21–27 show genuine variation. A single global threshold is suboptimal; per-layer thresholds calibrated against motivation data should Pareto-dominate the global-threshold settings reported here.

**Cross-task and cross-scale generalization.** All experiments use GSM8K and a 7B model. Whether the `token_topk_50` win and the self-reinforcing observation transfer to MATH, HumanEval, or larger model sizes remains an open question.

---

**Figures and Tables** (provided separately):
- Figure 1: Accuracy vs. FLOPs reduction scatter (18 settings, with "Better" direction and Pareto cluster highlighted).
- Table 1: Per-setting average denoising steps, accuracy, reuse rate, and FLOPs reduction.
