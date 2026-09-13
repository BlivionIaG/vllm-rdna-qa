---
name: rdna-silicon-gate
description: use this when reviewing HIP/ISA on a dest commit or fatbin.
---

# Silicon gate

ISA / tile lock for dest HIP and hippihx fatbins. Depth stays in
[rdna-hip-wiki silicon](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon)
— do not paste the wiki here.

## Dest ISA

- Inner ops: **`fdot2`** / `v_dot2c`, **`sdot4`**. No `fdot2.bf16`.
- **wave32** on gfx1030 dest. Bind on arch + wave, not GFX name alone.
- **LDS ≤ 64 KiB** / WG. 64 × 4 B banks. Pad on the tile README.
- **`K_STEP` matches packed DOT.** Dest large-M prefill is ConfigA
  (`K_STEP=32`). **ConfigH (`K_STEP=64`) is Leave.**
- One `--offload-arch` per fatbin. **Never `HSA_OVERRIDE`.** Never
  load a foreign ISA.

## Leave

| Leave | Why |
|---|---|
| WMMA / MFMA / TMA | not gfx1030 dest |
| `#ifdef WMMA` on shared DOT | gfx110x overlay is a **separate** object |
| **µs / tok/s as dest** | not a silicon contract |

Spill or bad ISel is a drop. Dump VGPR / LDS / `waves_per_eu` /
`v_dot2c` on the tile README — wiki occupancy pages, not this skill.

Runtime pin / Uncached AR / cache wipe:
[`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md).
