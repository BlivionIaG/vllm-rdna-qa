---
name: rdna-graph-qa
description: use this when debugging or reviewing cudagraph / HIP graph capture on gfx1030 dest. Not for ROCm pin, fatbin ISA, or HTTP clients.
---

# Graph QA

Cudagraph on ROCm gfx1030 is not the same as CUDA. Faults are
silent (host VAs in the 0x7f... range), symptoms are garbage
tokens or first-replay crashes, and the standard cudagraph recipes
don't apply.

Depth: [rdna-hip-wiki graph-capture](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

## 1. Capture-time invariants

Inside the captured region, these are hard fails:

| Invariant | Why | Where it leaks |
|---|---|---|
| No `cudaMalloc` / `cudaMallocAsync` | Captures warmup pointer; replay uses a different allocator slot | `torch.empty`, custom allocators |
| No `.item()` / `.tolist()` / sync D2H | Forces `cudaDeviceSynchronize`, illegal during capture | `isnan`, `argmax`, debug prints |
| No `cudaStreamWaitValue*` | Captures a value that's gone by replay | producer/consumer handshake |
| No lazy `_ensure_*` | First call inside capture runs init, captures its pointer | `_ensure_kv_cache`, `_ensure_workspace` |
| No `register_buffer` post-init | Mutates the captured state dict | — |

The atomic counters in `csrc/rocm/torch_bindings.cpp` track capture
state at runtime:

```cpp
std::atomic<int> g_rdna2_graph_capturing{0};   // set before BeginCapture
std::atomic<int> g_rdna2_capture_frozen{0};    // set after warmup replay
```

If `g_rdna2_graph_capturing`, the lazy path must no-op and the
pre-allocated arena must already be sized correctly.

## 2. Arena pre-allocation

Every state tensor the captured graph reads must be allocated
**before** `BeginCapture`, at the **maximum** size the graph will
ever need:

```python
# at init, NOT inside forward — zeros, not empty (page-commit)
self.kv_cache_arena = torch.zeros(max_bs, num_kv_heads, ...)
self.state_arena   = torch.zeros(max_bs, state_dim, ...)

# during capture: slice from the arena, no allocation
kv = self.kv_cache_arena[:current_batch]
```

ROCm's caching allocator isn't guaranteed to return the same
pointer for two same-shape calls inside the same capture — the
arena sidesteps the question.

**Pre-allocated outputs and state tensors must be `torch.zeros`,
not `torch.empty`.** RDNA2's `hipMalloc` returns virtual address
space with uncommitted physical pages; the kernel reads from these
and faults at `Memory access fault ... Page not present or
supervisor privilege`.

## 3. Persist hooks — must be CALLED

The arena pattern has one footgun: if the kernel that fills the
arena is "registered but uncalled" in the captured graph, replay
reads stale (or zero) data.

**Symptom**: `output is zero, no NaN, no fault` — looks like a
kitchen-sink kernel bug but is a missed persist call.

**Rule**: every arena-filling kernel must be **inside** the
captured graph, **eager** every step, or
`@eager_break_during_capture`-decorated. Any other path means the
captured graph references a stale arena.

No shared `PersistBuf` across live projection outs (`c350fa21`).
Write the **CAPTURE** slot on replay (`f7761592`); eager slot only
for larger prefill.

Never a one-shot decode wipe of live GDN state (`388a61b6`) —
alloc zeros, do not sanitizer-zero after prefill.

## 4. `eager_break_during_capture` — admission, not a fix

Use the decorator for kernels that are genuinely unsafe to capture:
Triton JIT kernels with `extern fn` scratch, dynamic-shape tensor
indexing ops, samplers that allocate per step.

It is **not** a workaround for "this kernel crashes inside capture,
let me just decorator it". The fix for crashes inside capture is a
captured-safe kernel. Decorating-by-default means the captured
graph no longer reflects the kernel under test.

Hybrid GDN/PLE capture mode: [rdna-dest-review §7](../rdna-dest-review/SKILL.md).
`VLLM_CG_NAN_INPUT_CHECK` default **off** (D2H class, `b7c77f08`).

## 5. State rings ≠ KV cache

GDN, KDA, DSA, Lightning, Engram, CED, indexer — these are all
**state rings separate from the KV cache**. They are paged (or
pooled), they are per-request, and they are **not shared by APC**.
Full state-ring map and dual page-commit ritual:
[rdna-arch-family §3, §4](../rdna-arch-family/SKILL.md).

The KV allocator and the state allocator are two different objects.
A state-ring allocation must **not** route through the KV allocator.

## 6. Family binds in capture path

`causal_conv1d_fwd`, `gated_rms` are **Qwen-GDN only**. Qwen4Exp
HC/QSA/PLE HIP is a different family bind and stays default-off.
Don't transplant GDN tiles into GLM KDA. Separate `.hsaco` per
family. Cross-ref:
[rdna-arch-family §2](../rdna-arch-family/SKILL.md).

## 7. Cudagraph garbage triage

| Symptom | First check |
|---|---|
| Constant "duct" / repeating token | `qwen_gdn_attention_core` output binding — state-arena slot uninitialised or wrong state tensor |
| Logprobs NaN | First kernel wrote to a non-contiguous output. Check the op schema's `Tensor!` mutation annotation — missing `!` makes torch.compile functionalise the op and drop the in-place write |
| First-replay `Memory Fault Error ... 0x7f...` | Host VA leaked. Search for `tensor.cpu()` or `.item()` in a debug print |
| `HSA_STATUS_ERROR_EXCEPTION` from `at::native::index_elementwise_kernel` with grid ≥ 1M | Tensor-index op captured with stale pointer. Decorate the index op or pre-allocate |

## 8. Bisect flags

```bash
# Confirm the bug is cudagraph-specific
vllm serve ... --enforce-eager                # clean → cudagraph bug

# Confirm it's breakable-CG-specific
VLLM_USE_BREAKABLE_CUDAGRAPH=0 vllm serve ... \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'

# Capture-time nan check (D2H per kernel — debug only)
VLLM_CG_NAN_INPUT_CHECK=1 vllm serve ...
```

Always have the `--enforce-eager` number first. If you can't
produce coherent output eagerly, the cudagraph problem is
secondary — fix the kernel first.
