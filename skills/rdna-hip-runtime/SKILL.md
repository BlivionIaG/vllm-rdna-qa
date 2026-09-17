---
name: rdna-hip-runtime
description: use this when checking ROCm pin / capture runtime / AR Uncached / cache wipe after tip move. Not for tile ISA, family registry, or HTTP clients.
---

# HIP runtime

Wiki:
[rdna-allreduce.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md),
[toolchain](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/toolchain).
Capture contract: [`rdna-graph-qa`](../rdna-graph-qa/SKILL.md).

1. **Live pin ROCm 7.14:** `hipcc --offload-arch=gfx1030 -O3` wave32.
   **WGP-only pin.** Nightly / TheRock = watch only.
2. Arenas / zeros / D2H: graph-qa. Hard-fail leftover `_ensure`.
   `hipStreamIsCapturing` + AR timeout stay **host-side**.
3. Uncached staging for **opt-in** `VLLM_RDNA_AR`. Leave Finegrained.
   Persist CAPTURE vs allocator recycle. Production TP fabric is
   custom AR under breakable CG (`849292ec`).
4. Engram: `hipHostMalloc`. **Leave UVA.**
5. After tip move: wipe `~/.cache/vllm` and torch HIP `*.so`. One
   arch per fatbin. No `HSA_OVERRIDE`.
