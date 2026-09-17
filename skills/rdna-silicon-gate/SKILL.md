---
name: rdna-silicon-gate
description: use this when reviewing HIP / ISA / dot-product instructions on a dest kernel or fatbin for gfx1030. Not for landing classification, ROCm pin, or HTTP clients.
---

# Silicon gate

RDNA2 (gfx1030) silicon rules for kernel review. Kernels live at
`csrc/rocm/`; family tile binds at `csrc/rocm/{qsa_rdna2,mrope_rdna2,gd*_rdna2}.cu`.

Depth: [rdna-hip-wiki/silicon](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon).

## 1. Wave32, not Wave64

gfx1030 dest is **wave32 native, WGP default**. Wave64 is not dest.
`num_warps=4` with wave32 is 128 threads (4×32), not "Wave128".

| Check | How |
|---|---|
| Compile target | `--offload-arch=gfx1030`, one arch per `.so` (see [rdna-hip-runtime §2](../rdna-hip-runtime/SKILL.md)) |
| Block size | threads = waves × 32 |
| Builtin | `__builtin_amdgcn_fdot2` / `__builtin_amdgcn_sdot4` — hipcc will not peephole `__hfma2` into DOT |

**Never use `HSA_OVERRIDE_GFX_VERSION`.** Same family, different
register / LDS / memory ordering — silent garbage.

## 2. ISA on gfx1030

Wiki: [valu.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/valu.md).

| ISA / builtin | Status | Use |
|---|---|---|
| `fdot2` (`V_DOT2C_F32_F16`) | **Live dest** | W4A16, W8A16, FP8-storage, mxfp4, EXL3 after unpack, FA QK |
| `sdot4` (`V_DOT4C_I32_I8`) | native, extras spec | W8A8 INT8 / Sage — live "W8A8" on dest is FP8→`fdot2` |
| `V_DOT4_I32_IU8` / `sudot4` | **gfx11+** | not on gfx1030 |
| `WMMA` / `TMA` / `MFMA` | **not present** | Leave |
| `fdot2.bf16` | gfx11+ | force fp16 on gfx1030 |

A kernel compiled for gfx1100 fails to load on gfx1030 with
`hipErrorInvalidImage`. Inspect fatbin:

```bash
/opt/rocm/bin/hipcc --offload-arch=gfx1030 -O3 csrc/rocm/foo.cu -o /tmp/foo.o
/opt/rocm/bin/llvm-objdump -t /tmp/foo.o | grep gfx
# Must show: amdgcn-amd-amdhsa--gfx1030
```

If it shows gfx1100, build env leaked the wrong arch.

## 3. K_STEP = packed DOT coverage

Wiki: [w4a16-prefill-config.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/w4a16-prefill-config.md).
`K_STEP` is the inner packed-DOT window, not "K of the GEMM".

| Config | THREADS / N_TILE / M_TILE / K_STEP / LDS | Status |
|---|---|---|
| **A** | 256 / 1024 / 16 / **32** / **0** | **dest** large-M (`7ac98a26`) |
| V1 / C | K_STEP=32, LDS=0 | live picker cells |
| **H** | K_STEP=64 | **Leave** — half-K skip class |

If a kernel has ConfigH on by default, demote it. Do not resurrect
deleted ConfigA_Large / ConfigP.

## 4. LDS budget

W4A16 ConfigA/C prefill is **LDS=0** (register tiles). Else LDS ≤
**64 KiB/WG** (128 KB/WGP pool). >64 KiB/WG Reject.

Use occupancy dump / `llvm-objdump` metadata. Spills → reduce
`BLOCK_*` or split the accumulator — do not "add LDS to ConfigA".

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
[rdna-arch-family §2](../rdna-arch-family/SKILL.md).

A "let's share the tiles" PR is almost always wrong. Family binds
in the dest tree are deliberate.

## 9. Reject VGPR / SGPR spills

```bash
hipcc --save-temps -O3 ... -o /tmp/foo.o
grep -c "sgpr_spill_count" /tmp/foo.s   # > 0 → reject
```

**SGPR is not an occupancy limiter on gfx1030** (wiki
[sgpr-occupancy.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/sgpr-occupancy.md)).
`.sgpr_spill_count > 0` / VGPR spill is **latency poison** — still
reject. Do not shrink SGPRs to raise waves/EU.

## 10. µs/tok is not dest

tok/s is not a dest gate. A kernel that breaks capture, TP=4, or
a family bind is **never** dest regardless of speed. Soak:
[rdna-dest-review §7](../rdna-dest-review/SKILL.md).

## 11. Future-arch note

BC-250 is Cyan Skillfish (gfx1013), not gfx906 and not gfx1030.
Verify wave size and DOT ISA before sharing `dot.hpp` between
gfx1013 and gfx1030 — they are not the same arch.
