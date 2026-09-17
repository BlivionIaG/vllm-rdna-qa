---
name: rdna-dest-review
description: use this when reviewing/landing a dest commit on opengfx1030/vllm-rdna rdna_extras. Not for HTTP clients, fatbin ISA, or wiki drafts.
---

# Dest review

[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Depth:
[rdna-hip-wiki](https://github.com/BlivionIaG/rdna-hip-wiki).
ISA: [`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md). Capture:
[`rdna-graph-qa`](../rdna-graph-qa/SKILL.md). Pin:
[`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md). Families:
[`rdna-arch-family`](../rdna-arch-family/SKILL.md).

1. Classify Dest / harvest / Leave. Take = extras. Import = fatbin +
   bind. **Never PR `vllm-project/vllm`.** Present on tip ≠ dest.
2. Residuals stay three tracks: mamba-split TP≤2 prefill; in-graph
   W4A16 world>2; APC state ≠ KV.
3. Family binds: **Qwen GDN ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE**.
   Do not transplant tiles.
4. **ConfigA dest, ConfigH Leave** (`7ac98a26`). No dormant slower
   configs (`56f67111`). AWQ prefill → GPTQ ConfigA; do not resurrect
   the dead AWQ kernel (`aaaae85` / `1046782`).
5. Contiguous GPTQ act (`c6b5cfb9`). HIP `reshape_and_cache` over
   Triton (`a447b237`). GDN prefill HIP is **opt-in**
   (`VLLM_GDN_HIP_PREFILL`, `cd1231fd`); default Triton/FLA is not a
   dest W4 land.
6. Dest residual 2 = capture-safe W4 HIP at world>2 / `splitting_ops`.
   **Never land** TP≤2 `can_implement` or Triton **W4/KV** as dest.
   Isolate eager TP=4 vs graph TP=4 vs TP=2 graph first.
7. TP≤2 / `VLLM_ROCM_MOE_SKINNY` are **triage or default-off**, not
   dest (`de8a4e41` / `1c1dbee8`). `VLLM_RDNA_AR` stays **opt-in**
   (`e0a7b167`). Production TP fabric is custom AR under breakable CG
   (`849292ec`); PYNCCL stays triage. HC/QSA/PLE HIP gates stay
   default-off.
8. Soak: TP=2 vs TP=4 × eager vs graph vs APC. **16k×8** = hybrid APC
   cell. Production: PIECEWISE + prefix; `max-num-seqs` 6 until GDN
   batched-decode at 8 is dest (`b78006a`).
