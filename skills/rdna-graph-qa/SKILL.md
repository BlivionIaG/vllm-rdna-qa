---
name: rdna-graph-qa
description: use this when debugging or reviewing cudagraph/HIP graph capture on gfx1030 dest.
---

# Graph QA

Wiki:
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

1. **Zeros, not empty.** `torch.empty` / `torch::empty` on gfx1030
   reads as garbage. Dest: `zeros` for any buffer a captured kernel
   reads.
2. **No `.item()` / D2H under capture.** Host sync aborts capture at
   startup. Gate probes with
   `if not torch.cuda.is_current_stream_capturing()`.
3. **Arenas + persist CAPTURE before `BeginCapture`.** Allocate, zero,
   persist first. Capture bakes pointers. `bind` never allocates.
4. **`Tensor!` outs** in `TORCH_LIBRARY`. Plain `Tensor` can drop the
   write (silent wrong output).
5. **Hard-fail `_ensure*` under capture.** If it would D2H or alloc,
   fail — do not probe. Warmup commits pages; capture must not grow
   them. APC state ≠ KV →
   [`rdna-dest-review`](../rdna-dest-review/SKILL.md) residual 3.
