---
name: rdna-silicon-gate
description: use this when reviewing HIP/ISA on a dest commit or fatbin.
---

# Silicon gate

Depth:
[rdna-hip-wiki silicon](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon).
Do not paste the wiki.

1. Inner ops: **`fdot2` / `sdot4`**, wave32. No `fdot2.bf16`. Bind on
   arch + wave, not GFX name alone.
2. **LDS ≤ 64 KiB** / WG. 64 × 4 B banks. Pad on the tile README.
3. **`K_STEP` matches packed DOT.** Dest large-M prefill = ConfigA
   (`K_STEP=32`). **ConfigH (`K_STEP=64`) is Leave.**
4. One `--offload-arch` per fatbin. **Never `HSA_OVERRIDE`.** Never
   load a foreign ISA.
5. **Leave** WMMA / MFMA / TMA, `#ifdef WMMA` on shared DOT, and
   **µs / tok/s as dest**. Spill or bad ISel is a drop.
6. Runtime pin / Uncached AR / cache wipe:
   [`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md).
