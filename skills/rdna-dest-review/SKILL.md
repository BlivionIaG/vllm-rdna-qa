---
name: rdna-dest-review
description: use this when reviewing/landing a dest commit on opengfx1030/vllm-rdna rdna_extras.
---

# Dest review

Read this before landing anything on
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Depth:
[rdna-hip-wiki](https://github.com/BlivionIaG/rdna-hip-wiki). Op zoo:
[hippihx](https://github.com/BlivionIaG/hippihx).

## Dest vs harvest vs Leave

| Verb | Meaning |
|---|---|
| **Dest** | signed serve path on `rdna_extras` |
| **Harvest** | observed / Later — not dest-signed |
| **Leave** | reject |
| **Take** | accept into dest extras |
| **Import** | consume a hippihx fatbin through a bind — not a `.cu` dump |

Present on tip ≠ dest. Dual HIP copies are how serve bugs accrete.

## Residuals — never collapse

Three tracks. One ticket / env / Triton path does **not** close them.

1. **mamba-split TP≤2 prefill**
2. **in-graph W4A16 world>2**
3. **APC state ≠ KV**

## Never land

- **TP≤2 `can_implement`** as dest. TP=2 is a soak cell, not the ceiling.
- **Triton as dest.** One HIP / hippihx V1 consume path. Triton is harvest.

## Family binds

**Qwen GDN ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE.** Separate hsaco.
Do not retarget GDN 16/48 onto KDA 64×128. For the registry (Qwen3.x /
Glm5Next / `deepseek_v4` vs `v41`) load
[`rdna-arch-family`](../rdna-arch-family/SKILL.md).

## Also load

- Graph / capture → [`rdna-graph-qa`](../rdna-graph-qa/SKILL.md)
- HIP/ISA / fatbin → [`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md)
- ROCm / AR / cache wipe → [`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md)
