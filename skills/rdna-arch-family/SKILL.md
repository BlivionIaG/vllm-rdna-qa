---
name: rdna-arch-family
description: use this when a new model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, M-RoPE, DeepSeek v4 / v4.1) on the gfx1030 dest fork. Not for dest landing, HIP/ISA, or HTTP clients.
---

# Arch family

The dest fork uses a **family → tile bind** model. Each kernel
under `csrc/rocm/` is family-scoped; pulling the wrong kernel
into the wrong family produces silent garbage output, not a crash.

Family briefs live at `arch-research/<family>/BRIEF.md` (separate
repo). No papers, no upstream citations — just enough to dispatch.

## 1. Family map

| Family | Examples | State rings | Conv1d | RoPE | Attention |
|---|---|---|---|---|---|
| `qwen4_exp` | Qwen4Exp / Qwen3.8-Flash-Next | HC streams (residual/inject/scale) | — | M-RoPE | dense + QSA + PLE; HC prefill glue, fused decode glue |
| `qwen3_next` (Qwen3.5/3.6/3.8 hybrid) | Qwen3.5-0.8B-exl3, Ornith-1.5-9B, Qwen3.6-27B-AWQ, Qwen3.8-27B-AWQ | GDN | `causal_conv1d_fwd` | M-RoPE (3-axis T,H,W) | GDN + QSA + dense |
| `qwen3_gdn` (Qwen3-Next, GDN only) | Qwen3-Next base | GDN | `causal_conv1d_fwd` | M-RoPE | GDN + dense |
| `glm5_next` | Glm5Next KDA + DSA | KDA, DSA | DSA-flavored | M-RoPE variant | KDA + DSA + dense |
| `deepseek_v4` | DeepSeek-V4 (Lightning) | Lightning | — | 2D-RoPE | MLA |
| `deepseek_v41` | DeepSeek-V4.1 (Engram + CED) | Engram, CED | — | 2D-RoPE | MLA |
| dense-attn-only | Llama, Mistral, Qwen2 | — | — | 1D-RoPE | dense attn |
| AWQ/GPTQ W4A16 | any quant model | — | — | family RoPE | family attn |

`qwen4_exp` ≠ Qwen3.x GDN+QSA ≠ Glm5Next KDA+DSA ≠ DeepSeek v4
≠ DeepSeek v4.1.

Two models can share a model name prefix and have completely
different architecture — the `architectures` field in
`config.json` is the source of truth, not the model name.

## 2. Family-vs-family transplants — don't do these

| Transplant | Why it's wrong |
|---|---|
| Qwen-GDN `gated_rms` → GLM KDA | Bias format and bias dims differ |
| Qwen-GDN `causal_conv1d_fwd` → Mamba / RWKV | state_len and weight ordering differ |
| Qwen M-RoPE → DeepSeek 2D-RoPE | 3-axis vs 2-axis broadcast |
| GLM KDA → Qwen GDN | state update is gated delta rule (different) |
| DeepSeek v4 (Lightning) → v4.1 (Engram + CED) | v4.1 ≠ v4 |

When a new family lands, write the kernels fresh. Do not
copy-paste a similar family's kernel and "tweak the dims".

## 3. State rings ≠ KV cache

GDN, KDA, DSA, Lightning, Engram, CED, indexer — all state
rings are **separate from the KV cache**. They are paged (or
pooled), per-request, and **not shared by APC**.

The dest fork's KV allocator and the state allocator are two
different objects. A state-ring allocation must not route through
the KV allocator. APC state ≠ KV — merging breaks both
correctness and capture.

## 4. Dual page-commit

When a family has two independent state rings (Qwen-GDN has
`ssm_state` and `conv_state`), each ring needs its own page-commit:

- `slot 0 = NULL_BLOCK_ID` for each
- Zero-initialised at **alloc** (page-commit)
- Never a one-shot decode wipe of live GDN state (`388a61b6`)

Skip the page-commit for one ring → the other reads garbage from
a previous request's state. Symptom: "first request correct,
second request correct, third request garbage" — the previous
state leaked through the missing ring.

## 5. Tile binds

| Family / dispatch | Kernel | Where |
|---|---|---|
| Qwen-GDN decode | `gdn_decode_rdna2` (gate: `VLLM_GDN_DECODE_KERNEL`) | `csrc/rocm/gdn_decode_rdna2.cu` |
| Qwen-GDN prefill | `gdn_prefill_{prep,kkt,solve_wy,delta_h,o}_rdna2` (5 chained) | `csrc/rocm/gdn_prefill_*_rdna2.cu` |
| Qwen-GDN Conv1d prefill | `causal_conv1d_fwd_rdna2` (gate: `VLLM_CAUSAL_CONV1D_RDNA2`) | `csrc/rocm/causal_conv1d_rdna2.cu` |
| Qwen-GDN Conv1d decode | `causal_conv1d_update_rdna2` (gate: `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE`) | same |
| Qwen M-RoPE | `mrope_rdna2` | `csrc/rocm/mrope_rdna2.cu` |
| Qwen QSA | `qsa_rdna2` | `csrc/rocm/qsa_rdna2.cu` |
| Dense attn (any family) | `fa_rdna2` (FA-RDNA2) | `csrc/rocm/fa_rdna2.cu` |
| W4A16 dense | `gptq_gemm_rdna2` / `gptq_gemm_rdna2_prefill` | `csrc/rocm/q_gemm_rdna2*.cu` |
| W4A16 MoE | `moe_q_gemm_rdna2` / `moe_w8a16_rdna2` | `csrc/rocm/moe_*.cu` |
| MoE fused | `moe_w8a16_fp8_rdna2` | `csrc/rocm/moe_w8a16_fp8_rdna2.cu` |
| FP8 dense | `gemm_w8a8_fp8_dense_rdna2` | `csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu` |
| MXFP4 MoE / dense | `mxfp4_dot2_{dense,moe}` | `csrc/rocm/mxfp4_dot2_*.cu` |
| EXL3 3bpw | `exl3_gemm_rdna2` / `exl3_decode_trellis_rdna2` / `moe_exl3_gemm_rdna2` / `exl3_hadamard_128` / `exl3_dequant_bits6_mul1` | `csrc/rocm/exl3_*.cu` |
| MLA sparse | `sparse_mla_decode_rdna2` / `sparse_mla_prefill_rdna2` | `csrc/rocm/sparse_mla_rdna2.cu` |
| Indexer | `indexer_paged_mqa_rdna2` | `csrc/rocm/indexer_paged_mqa_rdna2.cu` |
| PLE | `ple_short_conv_rdna2` | `csrc/rocm/ple_short_conv_rdna2.cu` |
| HC prefill (Qwen4Exp / Flash-Next) | `grouped_gemma_rmsnorm_rdna2` + `hc_silu_rdna2` + `hc_gate_mix_rdna2` + `hc_combine_rdna2` + `hc_combine_norm_rdna2` (5 elementwise/affine) | `csrc/rocm/hc_rdna2.cu` |
| Fused decode glue (M ≤ 8) | `rdna_gemv_act` + `rdna_hc_up_gate_mix` + `rdna_se_gate_up_silu` + `rdna_se_down_gated` | `csrc/rocm/rdna_fused_glue.cu` |
| All-reduce (TP>1) | `rdna_allreduce` | `csrc/rocm/rdna_allreduce.{cu,cuh}` |
| RMSNorm | AOT `vllm::rocm_layernorm::rms_norm` | `csrc/rocm/layernorm.cu` |

A kernel being present in the tree does **not** mean it is
enabled. Qwen4Exp HC / QSA / PLE / fused_glue stay **default-off**
until capture-safe (wiki
[qwen4exp-flash-next-hip.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/kernels/qwen4exp-flash-next-hip.md)).
GDN prefill HIP is opt-in (`VLLM_GDN_HIP_PREFILL`).

## 6. CSA2 / Lightning / QSA / MLA — distinct mechanisms

| Mechanism | What it does | Family |
|---|---|---|
| Lightning | v4 sparse / one-step path | `deepseek_v4` |
| CSA2 Full / Reindex / Reuse | v4.1 indexer modes | `deepseek_v41` |
| CED / Engram | v4.1 encoder + host tables | `deepseek_v41` |
| QSA | Qwen sparse attention | Qwen3-Next, Qwen4Exp |
| MLA | compressed KV | DeepSeek v2/v3/v4 decode |

CSA2 ≠ Lightning ≠ QSA ≠ MLA. `deepseek_v4` ≠ `deepseek_v41`.
Don't merge or alias them in the dispatcher.

## 7. New-family checklist

When `config.json` says a model is a new family:

1. Add the family identifier to the dispatcher. Use `architectures`,
   not the model name.
2. Identify state rings — what state beyond KV, where allocated,
   paged or pooled.
3. Identify the attention mechanism — dense attn, MLA, QSA, CSA2,
   Lightning, GDN, KDA. Each has its own kernel.
4. Identify the RoPE variant — 1D, 2D, M-RoPE, none.
5. Identify the conv1d path — Qwen-GDN needs `causal_conv1d`,
   most others don't.
6. Decide tile binds — write fresh or import from a similar
   family with justification.
7. Page-commit every state ring (§4).
8. APC compatibility — state rings do not share; KV cache does.
9. Soak matrix: TP=2/TP=4 × eager/graph × 1k/16k × c=1..16. See
   [rdna-dest-review §7](../rdna-dest-review/SKILL.md).

## 8. bf16 leftovers

Some models ship bf16 weights even when the rest is fp16
(embeddings, norms, GDN in_proj_a/b, conv1d, lm_head).

- **gfx1030**: bf16 has no native instruction. bf16 is rejected at
  upstream `check_if_supports_dtype` if GPU capability < 80, so
  pass `--dtype float16` at serve time and let vLLM cast at load.
  Don't try to make bf16 fast on gfx1030.
- **gfx1100 (RDNA3)**: bf16 has V_DOT2 support — use natively if
  the model ships bf16.
- **Triton fallback** for bf16 on gfx1030 is fine for non-hot-path
  layers. Don't route a bf16 MoE gate through Triton on gfx1030.

`VLLM_RDNA_FORCE_FP16` was an old knob and is no longer read by
vLLM code; the launcher scripts set it for documentation only.
Use `--dtype float16` instead.

## 9. Speculative decoding families

- `DSpark` — DeepSeek-specific, its own dispatcher and state.
  **Off the expert path**: only draft/verify, never routed expert
  forward.
- `MTP` — Qwen-specific.
- `EAGLE` / `Medusa` — their own families.

Each has its own speculative-decoding kernel set. Don't cross.
