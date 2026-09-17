---
name: rdna-silicon-gate
description: use this when reviewing HIP / ISA / dot-product instructions on a dest kernel or fatbin for gfx1030.
---

# Silicon gate

RDNA2 (gfx1030) silicon rules for kernel review. Kernels live at
`csrc/rocm/`; family tile binds at `csrc/rocm/{qsa_rdna2,mrope_rdna2,gd*_rdna2}.cu`.

Depth: [rdna-hip-wiki/silicon](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon).

## 1. Wave32, not Wave64

Every dest kernel targets `waveSize = 32`. A Wave64 kernel on
RDNA2 wastes 31 of 32 lanes on the second wave — a 50% efficiency
loss, not 2×.

| Check | How |
|---|---|
| Compile target | `--offload-arch=gfx1030`, one arch per `.so` (see [rdna-hip-runtime §2](rdna-hip-runtime/SKILL.md)) |
| Block size | `block_size_x * num_warps * 32 == block_size`. `num_warps=4` = Wave128, wrong on RDNA2 |
| Wave flag | `#pragma wave32` or `.wave32` in `.cu` (required in some HIP versions) |

**Never use `HSA_OVERRIDE_GFX_VERSION`** to make a CDNA-tuned
kernel "work" on gfx1030. Same ISA class, different register
counts, different LDS budget, different memory ordering — silent
garbage.

## 2. ISA on gfx1030

| ISA | Status | Use |
|---|---|---|
| `V_DOT2_F32_F16` | native | fp16 GEMM inner loop |
| `V_DOT4_I32_IU8` / DP4A | native | int8 / W4A16 GEMM |
| `WMMA` / `TMA` | **not present** | never use directly; Triton AMD path falls back |
| `MFMA` (matrix cores) | **not present** | CDNA only |
| BF16 native ops | **not present on gfx1030** | RDNA3 (gfx1100+) only — force fp16 on gfx1030 |

A kernel compiled for gfx1100 fails to load on gfx1030 with
`hipErrorInvalidImage`. Inspect fatbin:

```bash
/opt/rocm/bin/hipcc --offload-arch=gfx1030 -O3 csrc/rocm/foo.cu -o /tmp/foo.o
/opt/rocm/bin/llvm-objdump -t /tmp/foo.o | grep gfx
# Must show: amdgcn-amd-amdhsa--gfx1030
```

If it shows gfx1100, build env leaked the wrong arch.

## 3. K_STEP = packed DOT coverage

- `K_STEP = K` (full): each program covers the whole K in one
  V_DOT2 chain. **ConfigA = dest**.
- `K_STEP = K/2`: two programs share K. **ConfigH = Leave**
  (diagnostic only).
- `K_STEP = K/4` or finer: wave32 occupancy collapses. Reject.

If a kernel has ConfigH on by default, demote it to env-var opt-in.

## 4. LDS budget

| Path | LDS |
|---|---|
| `BLOCK_M=64, BLOCK_N=64`, fp16 | ~16 KiB/block |
| `BLOCK_M=128, BLOCK_N=128`, fp16 | ~64 KiB/block (max practical) |
| W4A16 prefill (BM=128, BN=128, packed int4) | ~32–48 KiB |
| Anything > 64 KiB | Reject — spills to global |

Use `rocprof-compiler --resource-usage` to inspect. Spills → reduce
`BLOCK_*` or restructure load order.

## 5. AWQ prefill quirk

AWQ stores zero offsets as signed-int8 with a per-group `zero`
field. The RDNA2 AWQ kernel requires `use_v2_format=true` so the
zero_offset is read at position 0; without it the kernel indexes
past the zero table and reads garbage scales.

The dispatcher must check `quant_config.use_v2_format` before
routing to the RDNA2 AWQ kernel. The non-v2 AWQ prefill kernel
is dead — do not resurrect.

## 6. causal_conv1d_fwd out-stride + NULL_BLOCK

The HIP `causal_conv1d_fwd_rdna2` takes an output tensor with its
**own stride** (contiguous, `stride_token = dim`), not the input's
alignment-padded stride (`next_pow2(dim)`).

Common bug: kernel writes to `out[row * dim + col]` when
`out.stride(0) == next_pow2(dim)` — every row overwrites the
previous. Garbage output.

`state_indices` slot 0 must equal `NULL_BLOCK_ID`. If the cache
slot index reads past 0 as a "no init state" sentinel and
`NULL_BLOCK_ID != 0`, the kernel reads a real slot as "no init"
and produces wrong output.

Qwen-GDN only. Mamba / RWKV may differ.

## 7. RoPE variants are not interchangeable

| Variant | Format | Kernel |
|---|---|---|
| M-RoPE (Qwen3.5/3.6/3.8 hybrid) | 3-axis (T, H, W), per-section scale | `mrope_rdna2.cu` |
| 2D-RoPE (DeepSeek, InternVL) | 2-axis (H, W), interleaved | separate — **do not reuse M-RoPE** |
| 1D-RoPE (Llama, Mistral) | single-axis, half-rotation | `apply_rotary_emb` Triton path |

M-RoPE grafted into a 2D-RoPE path silently mis-aligns positions.

## 8. Qwen-GDN ≠ GLM KDA

GDN and KDA look similar (both linear-attention state-space
hybrids) but the kernels are not swappable. Full matrix:
[rdna-arch-family §2](rdna-arch-family/SKILL.md).

A "let's share the tiles" PR is almost always wrong. Family binds
in the dest tree are deliberate.

## 9. Reject VGPR / SGPR spills

```bash
hipcc --save-temps -O3 ... -o /tmp/foo.o
grep -c "sgpr_spill_count" /tmp/foo.s   # > 0 → reject
```

Spills mean the kernel needs more registers than the wave allows
(255 VGPR on gfx1030). Fix by reducing `BLOCK_*` or splitting
the accumulator. Don't ship a kernel with spills.

## 10. µs/tok is not dest

Below 5% decode tok/s gain at the cudagraph-FPP cell, TP=4, 16k×8,
the soak regression risk outweighs the win — it's "leave", not
dest. A kernel that breaks cudagraph, TP=4, or any family is
**never** dest regardless of speed. Soak matrix:
[rdna-dest-review §7](rdna-dest-review/SKILL.md).

## 11. Future-arch note

BC-250 is Cyan Skillfish (gfx1013), not gfx906 and not gfx1030.
Verify wave size and DOT ISA before sharing `dot.hpp` between
gfx1013 and gfx1030 — they are not the same arch.
