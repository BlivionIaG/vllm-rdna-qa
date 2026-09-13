---
name: rdna-dest-review
description: use this when reviewing/landing a dest commit on opengfx1030/vllm-rdna rdna_extras.
---

# Dest review

Load before landing on
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Depth:
[rdna-hip-wiki](https://github.com/BlivionIaG/rdna-hip-wiki). Zoo:
[hippihx](https://github.com/BlivionIaG/hippihx).

1. Classify: **Dest** vs **harvest** vs **Leave**. **Take** = extras
   wiring. **Import** = hippihx fatbin + bind, not a `.cu` dump.
2. Residuals stay **three tracks**. Never collapse:
   mamba-split TP≤2 prefill; in-graph W4A16 world>2; APC state ≠ KV.
3. **Never land** TP≤2 `can_implement` or Triton as dest.
4. Family binds: **Qwen GDN ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE**.
   Separate hsaco. Do not retarget GDN 16/48 onto KDA 64×128.
5. **Never PR `vllm-project/vllm`.** Present on tip ≠ dest.
6. **Breakable CG + PYNCCL is soak triage**, not dest (eagers GDN+FA).
   Custom AR cannot replay under breakable CG — do not bake PYNCCL as
   dest fabric.
7. Dest for residual 2 = **capture-safe W4 HIP at world>2 /
   `splitting_ops`**, not more graph breaks or Triton.
8. Isolate before landing a gate: eager TP=4 vs graph TP=4 vs TP=2
   graph; `VLLM_DISABLED_KERNELS`; FA off; pynccl vs custom AR. A
   TP=4 graph fail is **not** a TP≤2 `can_implement`.
9. Soak cells: TP=2 vs TP=4 × eager vs graph vs APC. **16k×8** is the
   hybrid APC cell.
10. Also load: [`rdna-graph-qa`](../rdna-graph-qa/SKILL.md),
    [`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md),
    [`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md),
    [`rdna-arch-family`](../rdna-arch-family/SKILL.md).
