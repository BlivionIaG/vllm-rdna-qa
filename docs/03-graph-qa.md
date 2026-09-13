# Graph QA

Capture vs the gfx1030 page-commit guard. Wiki companion:
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

Two separate rules, one lock:

1. Caller-provided outs are **`Tensor!`**.
2. Never device→host on the captured / per-step decode path.

Not an occupancy change. Not a DOT / LDS-tile change.

## Zeros, not empty

gfx1030 hands back **uncommitted pages**: `torch.empty` / `torch::empty`
that is never fully written reads as **garbage**, not zeros. Capture
bakes addresses; replay faults or silently wrong.

Dest allocate path: `torch.zeros` / `torch::zeros` (or equivalent
zero-fill) for any buffer a captured kernel will read.

| Surface (harvest) | Allocate |
|---|---|
| FA `O` / `O_partial` / `M_partial` | `torch::zeros` |
| GDN scratch / `final_state` / prep outs | `torch.zeros` |
| EXL3 dequant / workspace | `zeros`, not `empty` |
| hippihx `plan` scratch | `zeroed=1`; serve zeros it |

The **buffer** fix (allocate zeroed) is capture-safe. The **probe** fix
(`.item()` then `zero_()`) is not.

`TODO(engine):` inventory dest `empty` still on captured outs. Each row
is a graph-QA fail until zeros.

## No `.item()` / D2H under capture

`.item()`, blocking `hipMemcpy` D2H, and `hipStreamSynchronize` are
illegal inside stream capture (`hipErrorStreamCaptureUnsupported`).
Failure lands at **startup** (capture time) and reads like a load
failure.

Gate probes. Do not delete them:

```python
if not torch.cuda.is_current_stream_capturing():
    ...  # .item() probe, then zero_()
```

Same class: `rdna_ar_timed_out()` and any per-replay `isnan().any()`
must stay off the default captured path. Prefer committing pages at
warmup over probing every decode.

`TODO(engine):` list dest probes still unsated (`VLLM_LOG_GDN_PTRS`,
`VLLM_GDN_DBG`, EXL3 debug, `VLLM_CG_NAN_INPUT_CHECK`). Default-off
is the dest contract.

## Arenas + persist CAPTURE before `BeginCapture`

Page-commit and workspace arenas must be **allocated, zeroed, and
persisted** before `cudaGraph` / HIP graph `BeginCapture`.

- Capture bakes pointers. A first-touch alloc inside the captured
  region is a dest bug.
- Persist keepalive for zeros / scratch stays on. Do not “optimize”
  it away because eager looked fine.
- CAPTURE-lifetime arenas are serve-owned. `bind` never allocates.

`TODO(engine):` dest sequence: arena create → zero → persist CAPTURE →
`BeginCapture` → replay. Document the extras file that owns each step.
`TODO(engine):` APC / piecewise graph: same rule; state slots are not
exempt (see [04-residuals.md](04-residuals.md) residual 3).

## `Tensor!` outs

An op that writes a caller buffer but declares plain `Tensor` lies to
functionalization: the write can be dropped. Symptom is **silent wrong
output**, not a crash.

Dest schema: every HIP out arg is `Tensor!` (`TORCH_LIBRARY`).

`TODO(engine):` schema audit vs dest `torch_bindings.cpp`. Hadamard
`Tensor! output` was the stale-schema lesson; do not leave a second one.
