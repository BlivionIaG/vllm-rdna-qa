---
name: rdna-silicon-gate
description: use this when reviewing HIP/ISA on a dest commit or fatbin.
---

# Silicon gate

Depth:
[rdna-hip-wiki silicon](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon).
Do not paste the wiki. No tok/s.

1. Inner ops: **`fdot2` / `sdot4`**, wave32. No `fdot2.bf16`. Bind on
   arch + wave.
2. **LDS ≤ 64 KiB** / WG. 64 × 4 B banks.
3. **`K_STEP` matches packed DOT.** ConfigA (`K_STEP=32`) dest.
   ConfigH (`K_STEP=64`) Leave.
4. One `--offload-arch` per fatbin. **Never `HSA_OVERRIDE`.**
5. **Leave** WMMA / MFMA / TMA and `#ifdef WMMA` on shared DOT.
6. Family binds: Qwen GDN tiles (`gated_rms` / `causal_conv1d_fwd`)
   ≠ GLM KDA ≠ M-RoPE ≠ DeepSeek 2D-RoPE. Separate hsaco; **never
   transplant**.
7. Page-commit / spill: outs and GDN state need **zeros not empty**.
   Reject `.sgpr_spill_count > 0` / VGPR spill as latency poison.
8. **gfx1013 Later:** BC-250 is Cyan Skillfish, not gfx906. **VERIFY
   wave size** before sharing `dot.hpp` with gfx1030. Never
   `HSA_OVERRIDE`.
9. Runtime pin / Uncached AR / cache wipe:
   [`rdna-hip-runtime`](../rdna-hip-runtime/SKILL.md).
