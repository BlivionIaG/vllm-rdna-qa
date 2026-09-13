---
name: rdna-arch-family
description: use this when a model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, deepseek_v4 vs v41).
---

# Arch family

Briefs: `arch-research/<family>/BRIEF.md`. No papers.

1. **Never PARK** across the registry: `qwen4_exp` ≠ Qwen3.x GDN+QSA
   ≠ Glm5Next KDA+DSA ≠ `deepseek_v4` ≠ `deepseek_v41`.
2. Dual page-commit + **APC state ≠ KV**. Heaps: state rings ≠ KV
   pages ≠ Engram ≠ indexer.
3. Tile binds **Qwen-GDN only** (`gated_rms` / `causal_conv1d_fwd`).
4. **CSA2 modes ≠ lightning ≠ QSA ≠ MLA.** V4.1 = Engram+CED, not v4
   lightning.
5. Leftover BF16 / DSpark stay **off the expert path**.
