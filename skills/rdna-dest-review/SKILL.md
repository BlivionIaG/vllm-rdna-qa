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

1. Classify the change: **Dest** (signed `rdna_extras` path) vs
   **harvest** (observed / Later) vs **Leave** (reject). **Take** =
   extras wiring. **Import** = hippihx fatbin + bind, not a `.cu` dump.
   Present on tip ≠ dest.
2. Keep residuals as **three tracks**. Never collapse into one ticket,
   env, or Triton path:
   - mamba-split TP≤2 prefill
   - in-graph W4A16 world>2
   - APC state ≠ KV
3. **Never land** TP≤2 `can_implement` as dest (TP=2 is a soak cell).
   **Never land Triton as dest** (one HIP / hippihx V1 consume).
4. Family binds: **Qwen GDN ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE**.
   Separate hsaco. Do not retarget GDN 16/48 onto KDA 64×128.
5. Also load: [`rdna-graph-qa`](../rdna-graph-qa/SKILL.md),
   [`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md),
   [`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md),
   [`rdna-arch-family`](../rdna-arch-family/SKILL.md).
