---
name: rdna-dest-review
description: use this when reviewing/landing a dest commit on opengfx1030/vllm-rdna rdna_extras.
---

# Dest review

[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Depth:
[rdna-hip-wiki](https://github.com/BlivionIaG/rdna-hip-wiki).

1. Classify Dest / harvest / Leave. Take = extras. Import = fatbin +
   bind. **Never PR `vllm-project/vllm`.** Present on tip ≠ dest.
2. Residuals stay three tracks: mamba-split TP≤2 prefill; in-graph
   W4A16 world>2; APC state ≠ KV.
3. Family binds: **Qwen GDN ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE**.
   `gated_rms` / `causal_conv1d_fwd` Qwen-GDN only (`71a54552` /
   `82b6f18`).
4. **ConfigA dest, ConfigH Leave** (`7ac98a26`). No dormant slower
   configs (`56f67111`). AWQ prefill → GPTQ ConfigA; do not resurrect
   the dead AWQ kernel (`aaaae85` / `1046782`).
5. Contiguous GPTQ act (`c6b5cfb9`). HIP `reshape_and_cache` over
   Triton (`a447b237`).
6. Dest residual 2 = capture-safe W4 HIP at world>2 / `splitting_ops`.
   **Never land** TP≤2 `can_implement` or Triton as dest. Isolate
   eager TP=4 vs graph TP=4 vs TP=2 graph first.
7. TP≤2 / breakable-CG+PYNCCL / `VLLM_ROCM_MOE_SKINNY` are **triage
   or default-off**, not dest (`de8a4e41` / `f86faadb` / `1c1dbee8`).
   `VLLM_RDNA_AR` stays **opt-in** (`e0a7b167`).
8. Soak: TP=2 vs TP=4 × eager vs graph vs APC. **16k×8** = hybrid APC.
