---
name: rdna-arch-family
description: use this when a model family hits the board (Qwen GDN/QSA, GLM KDA/DSA, deepseek_v4 vs v41).
---

# Arch family

**Never PARK across** families. A parked bind, cached graph, hsaco, or
scheduler slot for one row must not run another. Briefs only — no
papers: `arch-research/<family>/BRIEF.md`.

## Registry

| Family | Class | Not |
|---|---|---|
| `qwen4_exp` | harvest / Leave as dest HIP | Qwen3.x GDN+QSA |
| Qwen3.x GDN+QSA | GDN + Qwen sparse | Glm5Next |
| Glm5Next KDA+DSA | KDA state + DSA KV | Qwen GDN pages |
| `deepseek_v4` | lightning / sparse MLA | `deepseek_v41` |
| `deepseek_v41` | **CED / CSA2** | `deepseek_v4` |

Tile binds (GDN ≠ KDA ≠ M-RoPE ≠ 2D-RoPE):
[`rdna-dest-review`](../rdna-dest-review/SKILL.md).

## Hybrid heaps

**State rings ≠ KV pages ≠ Engram ≠ indexer.** Prefix-cache /
chunked-prefill must split state vs KV (APC residual). Do not PARK
Engram on an indexer slot.

Attn binds: **Full / Reindex / Reuse ≠ lightning indexer ≠ QSA ≠ MLA.**

## Off the expert path

**Leftover BF16** and **MTP-DSpark** stay off expert GEMM. Experts
are dest W4 / DOT. Leftover is `moe/leftover_bf16`. MTP is serve.
