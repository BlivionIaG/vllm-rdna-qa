---
name: rdna-dest-review
description: use this when reviewing or landing a dest commit on opengfx1030/vllm-rdna rdna_extras. Not for HTTP clients, fatbin ISA, or wiki drafts.
---

# Dest review

Dest = the per-kernel, per-dispatch HIP/Triton land-zone for the
gfx1030 (RDNA2) optimised vLLM fork. Branch: `rdna_extras` on
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna).
Each rule here is a bug that already cost time.

## 1. Classify the change first

| Class | What it is | Review depth |
|---|---|---|
| **Dest** | New HIP kernel / dispatcher / register-time gate | Full |
| **Harvest** | Re-imported from upstream or another fork | Spot-check gate only |
| **Leave** | Disabled by env var, A/B only | Confirm no active-path regression |

## 2. Architecture gate

Gate every dest change with `on_gfx10x()` (or stricter
`on_gfx1030()`) in **all four** places:

1. `vllm/platforms/rocm.py` — import side
2. Python dispatch site (where the kernel is chosen)
3. `csrc/rocm/torch_bindings.cpp` — `TORCH_LIBRARY_EXPAND`
4. `CMakeLists.txt` — source list (avoid pulling gfx1030-only `.cu` into CDNA builds)

A missing gate that compiles on CDNA is a silent correctness bug on
MI300X. Dest soak is gfx1030; do not block a dest land on a gfx942
box you do not have.

**Never PR to `vllm-project/vllm`.** Dest extras live on
opengfx1030. "Present on tip" is not dest.

## 3. Family binds

A tile shape that works for Qwen GDN does not work for GLM KDA,
M-RoPE, or DeepSeek 2D-RoPE. The kernel zoo at `csrc/rocm/` is
family-scoped; the dispatcher at `vllm/model_executor/models/`
decides which fires. Full map: [rdna-arch-family](../rdna-arch-family/SKILL.md).

Reject any PR that transplants a family tile across families
without a written justification in the commit body.

## 4. Three residuals stay separate

Every dest kernel touches three residuals that must never collapse:

| Residual | Symptom if collapsed |
|---|---|
| mamba-split prefill TP≤2 | V620-scale prefill loses hybrid split |
| in-graph W4A16 at world_size > 2 | capture-safe W4 HIP at TP=4 needs the gated fast path |
| APC state ≠ KV | state rings (Engram / GDN / indexer) live in separate heaps from KV; merging breaks capture |

If a "simplification" PR tries to merge two of these, reject it.
Use `splitting_ops` to keep the boundary explicit.

## 5. Config hygiene

- **ConfigA dest, ConfigH Leave** for `K_STEP`. ConfigH = half-K is
  diagnostic only — keep it as env-var opt-in, do not promote.
- **No dormant slower configs**. If 4 configs exist but only 1 wins
  on gfx1030, delete the others. Dormant code re-asserts at the
  worst time.
- **AWQ prefill is dead**. Route through the GPTQ ConfigA path.
  Do not resurrect.

## 6. Pre-allocate + contiguous

- **Pre-allocate before `BeginCapture`**. `torch.empty` /
  `cudaMalloc` / `.item()` inside the captured region captures a
  stale pointer and faults on replay.
- **Activation tensors contiguous** when passed to HIP kernels.
  Non-contiguous slices force kernel-side gathers and break LDS
  budgeting.
- **HIP `reshape_and_cache` only** when the dispatch path is HIP.
  Mixing Triton + HIP `reshape_and_cache` in the same captured
  graph produces constant-token garbage output.

Full capture invariants: [rdna-graph-qa](../rdna-graph-qa/SKILL.md).

## 7. Soak matrix

Pass = coherent output, no NaN, no faults. **tok/s is not dest.**

| Cell | Notes |
|---|---|
| TP=2 / TP=4 × eager | kernel correctness |
| TP=2 / TP=4 × graph | family capture mode (below) |
| APC / 16k shared prefix | hybrid state≠KV cell |

Capture mode is family-specific (dest launchers):

- Qwen3.8-27B hybrid GDN: `FULL_AND_PIECEWISE` + breakable CG
  (`serve_gfx1030_full.sh`)
- Qwen4Exp / Flash-Next: **PIECEWISE** + prefix;
  `FULL_AND_PIECEWISE` is Leave until Qwen4Exp FULL is dest
  (`serve_gfx1030_flashnext.sh`)
- Production `max-num-seqs` **6** until GDN batched-decode at 8
  is dest (`b78006a`)

## 8. Triage / opt-in markers — never promote

Default-off, never dest, unless a launcher already flipped them
with a soak:

- `VLLM_RDNA_AR` (one-shot Uncached AR — **opt-in**, `e0a7b167`)
- `VLLM_ROCM_MOE_SKINNY`
- `VLLM_RDNA_HC_PREFILL_HIP` / `VLLM_RDNA_QSA_HIP` /
  `VLLM_RDNA_PLE_CONV_HIP` / `VLLM_RDNA_FUSED_HC` — wiki lock
  default-off until capture-safe
- `VLLM_GDN_HIP_PREFILL` — GDN prefill HIP is opt-in; default
  Triton/FLA is not a dest W4 land (`cd1231fd`)

Dest-default in launchers (not triage):

- `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` (`849292ec`)
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1` — admission (poison ops leave
  the graph), not a capture fix
- `HSA_FORCE_FINE_GRAIN_PCIE=1`, `GPU_MAX_HW_QUEUES=2`

Env detail: [rdna-hip-runtime §6, §7](../rdna-hip-runtime/SKILL.md).

## 9. Commit-message format

Dest commits must make family + class + cell obvious at a glance:

```
[Qwen-GDN][rdna2] gdn_decode_rdna2: split per-chunk index from global

- g reads stayed global, k/w/u/v_new were chunk-local in the
  same per-chunk loop — chunk 0 worked, chunks 1+ read chunk 0 rows.
- Tests: tests/kernels/test_gdn_prefill_rdna2.py T=64..256 PASS.
- Soak: TP=4 graph 16k×8 no NaN. Do not land tok/s as dest.
```

Keep the kernel history searchable across forks.
