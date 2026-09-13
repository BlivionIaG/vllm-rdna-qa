# HIP runtime

Dest box contract. Not a second
[rdna-hip-wiki](https://github.com/BlivionIaG/rdna-hip-wiki). Cite
[silicon/](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon)
for occupancy / waitcnt / AR protocol.

## Pin

| Knob | Dest |
|---|---|
| ROCm | **7.14** (`hipcc`) |
| Arch | **gfx1030** |
| Wave | **wave32** |

Do not silently retarget dest tiles to another toolchain. hippihx pin
is `HIPPIHX_ROCM_PIN`. Same number here.

`TODO(silicon):` confirm extras build and hippihx still match 7.14.

## Pre-alloc before `BeginCapture`

Allocate, zero, and persist CAPTURE arenas **before** HIP / CUDA graph
`BeginCapture`. Same lock as [03-graph-qa.md](03-graph-qa.md). Runtime
rule: no first-touch `hipMalloc` on the captured stream.

## Illegal inside capture

| Call | Why |
|---|---|
| `hipMalloc` / `hipMallocAsync` / grow | pointer bake; mempool reuse |
| `.item()` / D2H / host sync | `hipErrorStreamCaptureUnsupported` |
| `hipStreamWaitValue*` / `hipWaitValue*` | host/device wait not capture-safe |

Gate probes. Do not delete them. See graph QA for the
`is_current_stream_capturing()` pattern.

`TODO(engine):` dest grep for `WaitValue` / `hipMalloc` on captured
paths (AR timeout, GDN probes, split-K).

## Uncached AR staging — Leave Finegrained

leapdragon / dest `rdna_ar`: staging is
`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`.
**`hipDeviceMallocFinegrained` is a silent no-op on this part.** Leave
it. Push, not pull. AR stays opt-in (`VLLM_RDNA_AR=0`). Wiki:
[rdna-allreduce.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md).

`TODO(silicon):` dest-signed confirm Finegrained still no-op on V620
7.14.

## Cache wipe after tip moves

After dest extras (or hippihx fatbin) moves, wipe stale JIT / AOT:

```bash
rm -rf ~/.cache/vllm ~/.cache/torch_extensions
```

A green soak on an old `torch_extensions` blob is not dest-signed.

`TODO(engine):` list other caches that pin hsaco (`HIP_CACHE`,
`~/.triton`) — wipe vs Leave.

## One arch per fatbin — no `HSA_OVERRIDE`

Same as [01-isa-dest.md](01-isa-dest.md). One `--offload-arch` per
artifact. Never `HSA_OVERRIDE_GFX_VERSION`. Runtime load of a foreign
ISA is Leave.
