---
name: rdna-silicon-gate
description: use this when reviewing HIP/ISA on a dest commit or fatbin. Not for landing classification, ROCm pin, or HTTP clients.
---

# Silicon gate

Depth:
[rdna-hip-wiki silicon](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon).
No wiki paste. No tok/s. Landing:
[`rdna-dest-review`](../rdna-dest-review/SKILL.md).

1. **`fdot2` / `sdot4`**, wave32. One `--offload-arch`. Never
   `HSA_OVERRIDE`. Leave WMMA / TMA / `#ifdef WMMA` on shared DOT.
2. **`K_STEP` = packed DOT coverage.** ConfigA dest. ConfigH half-K
   Leave (`7ac98a26`).
3. **LDS=0** ConfigA/C OK for W4 prefill. Else LDS ≤ 64 KiB. AWQ
   `zero_offset=0` via `use_v2_format` (`aaaae85`).
4. Out-stride + `NULL_BLOCK` on `causal_conv1d_fwd` (`82b6f18`).
   **M-RoPE ≠ 2D-RoPE** (`cf055ad1`). Qwen GDN tiles ≠ GLM KDA.
   Separate hsaco; never transplant.
5. Outs / GDN state: **zeros not empty**. Reject
   `.sgpr_spill_count > 0` / VGPR spill.
6. **gfx1013 Later:** BC-250 is Cyan Skillfish, not gfx906. VERIFY
   wave before sharing `dot.hpp` with gfx1030.
