---
name: rdna-dest-review
description: use this when reviewing or landing a dest commit on opengfx1030/vllm-rdna rdna_extras.
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
MI300X. Test on at least one non-RDNA arch before merge (CDNA
gfx942 preferred, else CPU).

**Never PR to `vllm-project/vllm`.** Dest extras live on
opengfx1030. "Present on tip" is not dest.

## 3. Family binds

A tile shape that works for Qwen GDN does not work for GLM KDA,
M-RoPE, or DeepSeek 2D-RoPE. The kernel zoo at `csrc/rocm/` is
family-scoped; the dispatcher at `vllm/model_executor/models/`
decides which fires. Full map: [rdna-arch-family](rdna-arch-family/SKILL.md).

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

Full capture invariants: [rdna-graph-qa](rdna-graph-qa/SKILL.md).

## 7. Soak matrix

Every dest kernel touching the forward path must pass:

| Cell | Pass criteria |
|---|---|
| TP=2 eager, 16k in / 1k out × 8 prompts | No NaN, decode tok/s ≥ previous tip |
| TP=4 eager, 16k in / 1k out × 8 prompts | No NaN, decode tok/s ≥ previous tip |
| TP=2 cg (FULL_AND_PIECEWISE) | Coherent output, no faults |
| TP=4 cg (FULL_AND_PIECEWISE) | Coherent output, no faults |
| APC on, 8 × 16k shared prefix | Coherent, no shared-state NaN |

The hybrid APC cell catches residual state-corruption bugs that
short contexts miss.

## 8. Triage / opt-in markers — never promote

These are default-off, never dest. If a PR promotes any of these
to default-on, the burden of proof is on the PR ("what changed
that makes the larger world safe now?").

- `VLLM_RDNA_AR` (one-shot custom AR — distinct from
  `VLLM_FORCE_CUSTOM_ALL_REDUCE`, which **is** dest-default in
  the launcher)
- `VLLM_ROCM_MOE_SKINNY` (MoE skinny GEMV)
- `VLLM_USE_BREAKABLE_CUDAGRAPH` outside the documented cell

Note: `VLLM_FORCE_CUSTOM_ALL_REDUCE` is dest-default-on in
`scripts/serve_gfx1030_full.sh` (since 2026-09-16, with TP=4 +
breakable CG verified correct). It is **not** in the triage list.
The upstream vLLM env-var default remains `False`; the dest
launcher overrides it. Env vars and behaviour:
[rdna-hip-runtime §6, §7](rdna-hip-runtime/SKILL.md).

## 9. Commit-message format

Dest commits must make family + class + cell obvious at a glance:

```
[Qwen-GDN][rdna2] gdn_decode_rdna2: split per-chunk index from global

- g reads stayed global, k/w/u/v_new were chunk-local in the
  same per-chunk loop — chunk 0 worked, chunks 1+ read chunk 0 rows.
- Tests: tests/kernels/test_gdn_prefill_rdna2.py T=64..256 PASS.
- Soak: TP=4 cg FPP 16k×8 no NaN, decode 31.3 tok/s (was 28.4).
```

Keep the kernel history searchable across forks.
