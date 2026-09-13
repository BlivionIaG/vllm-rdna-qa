---
name: rdna-hip-runtime
description: use this when checking ROCm pin / capture runtime / AR Uncached / cache wipe after tip move.
---

# HIP runtime

Not a second wiki. Cite
[rdna-allreduce.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md)
and [graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

## Pin

**ROCm 7.14**, `hipcc`, **gfx1030**, **wave32**. hippihx:
`HIPPIHX_ROCM_PIN`. Do not silently retarget dest.

## Capture runtime

Pre-alloc + zero + persist CAPTURE **before** `BeginCapture`. Illegal
on the captured stream: `hipMalloc` / grow, `.item()` / D2H,
`hipStreamWaitValue*` / `hipWaitValue*`. See
[`rdna-graph-qa`](../rdna-graph-qa/SKILL.md).

## AR staging

Dest `rdna_ar` staging is
`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`.
**Leave Finegrained** — `hipDeviceMallocFinegrained` is a silent no-op
on this part. Push, not pull. Gate stays opt-in (`VLLM_RDNA_AR=0`).

## Cache wipe after tip move

After dest extras or a fatbin moves:

```bash
rm -rf ~/.cache/vllm ~/.cache/torch_extensions
```

A green soak on a stale `torch_extensions` blob is not dest-signed.

One `--offload-arch` per fatbin. No `HSA_OVERRIDE`.
[`rdna-silicon-gate`](../rdna-silicon-gate/SKILL.md).
