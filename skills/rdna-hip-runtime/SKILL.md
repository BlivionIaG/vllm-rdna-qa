---
name: rdna-hip-runtime
description: use this when checking ROCm pin / capture runtime / AR Uncached / cache wipe after tip move.
---

# HIP runtime

Wiki:
[rdna-allreduce.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md),
[toolchain](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/toolchain).

1. **Live pin ROCm 7.14:** `hipcc --offload-arch=gfx1030 -O3` wave32.
   **WGP-only pin.** Nightly / TheRock = watch only.
2. Pre-alloc before `BeginCapture`. No malloc / `.item()` / D2H /
   `WaitValue*`. Hard-fail leftover `_ensure`.
   `hipStreamIsCapturing` + AR timeout stay **host-side**.
3. Uncached AR staging. Leave Finegrained. Persist CAPTURE vs
   allocator recycle.
4. Engram: `hipHostMalloc`. **Leave UVA.** PYNCCL = triage.
5. After tip move: wipe `~/.cache/vllm` and torch HIP `*.so`. One
   arch per fatbin. No `HSA_OVERRIDE`.
