---
name: rdna-graph-qa
description: use this when debugging or reviewing cudagraph/HIP graph capture on gfx1030 dest.
---

# Graph QA (gfx1030 dest)

Capture vs page-commit. Wiki:
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

## Zeros, not empty

`torch.empty` / `torch::empty` on gfx1030 reads as **garbage**, not
zeros. Dest: `zeros` for any buffer a captured kernel will read. The
buffer fix is capture-safe. A `.item()` probe is not.

## No `.item()` / D2H under capture

`.item()`, blocking D2H, and host sync abort capture
(`hipErrorStreamCaptureUnsupported`) at **startup**. Gate probes:

```python
if not torch.cuda.is_current_stream_capturing():
    ...
```

## Arenas + persist CAPTURE before `BeginCapture`

Allocate, zero, persist CAPTURE arenas **before** `BeginCapture`.
Capture bakes pointers. `bind` never allocates.

## `Tensor!` outs

HIP out args in `TORCH_LIBRARY` are `Tensor!`. Plain `Tensor` can drop
the write (silent wrong output).

## Hard-fail `_ensure` under capture

Any dest `_ensure*` (page commit, state slot, workspace) that would
D2H or alloc under capture must **hard-fail**, not probe. Warmup
commits pages; capture must not grow or inspect them.

APC state is not KV — load
[`rdna-dest-review`](../rdna-dest-review/SKILL.md) residual 3.
