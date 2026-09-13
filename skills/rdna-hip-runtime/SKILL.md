---
name: rdna-hip-runtime
description: use this when checking ROCm pin / capture runtime / AR Uncached / cache wipe after tip move.
---

# HIP runtime

Wiki:
[rdna-allreduce.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md),
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

1. **Pin ROCm 7.14**, `hipcc`, gfx1030, wave32. Match hippihx
   `HIPPIHX_ROCM_PIN`. Do not silently retarget dest.
2. Pre-alloc + zero + persist CAPTURE **before** `BeginCapture`.
   Illegal on the captured stream: `hipMalloc` / grow, `.item()` /
   D2H, `WaitValue*`. See
   [`rdna-graph-qa`](../rdna-graph-qa/SKILL.md).
3. AR staging is `hipDeviceMallocUncached`. **Leave Finegrained**
   (silent no-op on this part). Push, not pull. `VLLM_RDNA_AR=0`
   default.
4. After dest extras or a fatbin moves:
   `rm -rf ~/.cache/vllm ~/.cache/torch_extensions`. Stale
   `torch_extensions` is not dest-signed.
5. One `--offload-arch` per fatbin. No `HSA_OVERRIDE`.
   [`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md).
