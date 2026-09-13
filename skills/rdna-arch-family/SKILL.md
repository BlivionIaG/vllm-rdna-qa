---
name: rdna-arch-family
description: use this when a model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, deepseek_v4 vs v41).
---

# Arch family

Briefs only: `arch-research/<family>/BRIEF.md`. No papers. Tile binds:
[`rdna-dest-review`](../rdna-dest-review/SKILL.md).

1. **Never PARK across** families. A parked bind, graph, hsaco, or
   scheduler slot for one row must not run another.
2. Registry — do not alias:
   - `qwen4_exp` ≠ Qwen3.x GDN+QSA (qwen4_exp is harvest / Leave as dest HIP)
   - Qwen3.x GDN+QSA ≠ Glm5Next KDA+DSA
   - `deepseek_v4` (lightning / sparse MLA) ≠ `deepseek_v41` (**CED / CSA2**)
3. Hybrid heaps: **state rings ≠ KV pages ≠ Engram ≠ indexer.**
   Prefix-cache / chunked-prefill must split state vs KV.
4. Attn binds: **Full / Reindex / Reuse ≠ lightning indexer ≠ QSA ≠ MLA.**
5. **Leftover BF16** and **MTP-DSpark** stay off expert GEMM. Experts
   are dest W4 / DOT. Leftover is `moe/leftover_bf16`. MTP is serve.
