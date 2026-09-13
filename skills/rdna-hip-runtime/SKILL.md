---
name: rdna-hip-runtime
description: use this when checking ROCm pin / capture runtime / AR Uncached / cache wipe after tip move.
---

# HIP runtime

Wiki:
[rdna-allreduce.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md),
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md),
[toolchain](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/toolchain).

1. **Live pin ROCm 7.14:** `hipcc --offload-arch=gfx1030 -O3` wave32.
   Nightly / TheRock = **watch only**. Never a silent dest bump.
2. **Pre-alloc before `BeginCapture`.** No `malloc` / `.item()` / D2H /
   `WaitValue*` in capture. **Hard-fail leftover `_ensure` under
   capture** (do not probe). See
   [`rdna-graph-qa`](../rdna-graph-qa/SKILL.md).
3. **Uncached AR staging.** Leave Finegrained. **Persist CAPTURE**
   arenas vs allocator recycle (mempool reuse is a dest bug).
4. After tip moves: wipe `~/.cache/vllm` and torch HIP
   `torch_extensions/*.so`. One `--offload-arch` per fatbin. **No
   `HSA_OVERRIDE`.** [`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md).
