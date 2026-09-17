---
name: rdna-hip-runtime
description: use this when checking ROCm version pin, capture runtime, all-reduce mode, or cache wipe after a tip move on the gfx1030 dest fork.
---

# HIP runtime

Runtime layer under the dest kernels: ROCm pin, build env,
capture-time hookup, AR modes, queue / TunableOp defaults, and
the post-tip-move cache wipe.

Depth:
[rdna-hip-wiki toolchain](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/toolchain),
[rdna-hip-wiki rdna-allreduce](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/rdna-allreduce.md).

## 1. ROCm pin

The dest fork ships with one pinned ROCm per image tag. Nightly /
TheRock is watch-only — don't bench on it.

| Pin | Status | Why |
|---|---|---|
| **ROCm 7.14.x (rocm_sdk_core / rocm_sdk_libraries)** | **dest default** | bundled rocblas 5.5.0 ships matching ABI |
| ROCm 7.2.x (legacy venv) | Leave — Qwen2.5/3.5/3.6 bench matrix only | rccl 2.27.7 |
| ROCm 7.13.x | Leave | TunableOp first-run tuning bleed |
| Nightly / TheRock | watch | ABI drift |

`hipcc --version` should report `HIP version: 7.14.xxxxx`. If it
doesn't, the launcher has the wrong `ROCM_HOME`.

## 2. Build env vars — one arch per `.so`

Most common build failure: arch skew. The `.so` embeds
`amdgcn-amd-amdhsa--gfx1100` and the V620 (gfx1030) refuses to
load it with `hipErrorInvalidImage`.

```bash
export PYTORCH_ROCM_ARCH='gfx1030'
export CMAKE_HIP_ARCHITECTURES='gfx1030'
export AMDGPU_TARGETS='gfx1030'
```

Multi-arch (`gfx1030;gfx1100`) roughly doubles compile time and
`.so` size — only use in the Docker layer, never locally.

**Verify after every build**:

```python
import torch
schemas = torch._C._jit_get_all_schemas()
rocm = [str(s) for s in schemas if str(s).startswith('_rocm_C::')]
rdna = [s for s in rocm if 'rdna' in s]
assert len(rdna) > 10, f'expected >10 RDNA ops, got {len(rdna)}'
assert any('all_reduce' in s for s in rocm), 'vllm::all_reduce NOT registered'
print(f'OK: {len(rocm)} rocm_C ops, {len(rdna)} RDNA-specific')
```

`dir(torch.ops._rocm_C)` returns `['name']` even when ops are
registered — the schema query above is the real check.

## 3. Capture runtime — host-side guards only

During cudagraph capture (`g_rdna2_graph_capturing` atomic in
`torch_bindings.cpp`):

- `hipStreamIsCapturing(stream)` → true. No host-side AR timeout,
  no `.item()`, no D2H.
- AR staging buffer pre-pinned (uncached, §5).
- Persist hooks (the unfreeze callback) run on the host thread,
  not the GPU stream.
- Hard-fail leftover lazy `_ensure_*` paths during capture — log
  and raise, don't silently alloc.

Use the atomic as a guard:

```cpp
bool rdna2_in_capture() {
    return g_rdna2_graph_capturing.load(std::memory_order_acquire) != 0;
}
```

Capture invariants in detail: [rdna-graph-qa](rdna-graph-qa/SKILL.md).

## 4. JIT-built kernels (FA-RDNA2)

`torch.utils.cpp_extension.load_inline` at process startup needs:

```bash
export ROCM_HOME=/opt/rocm/...
export ROCM_PATH=/opt/rocm/...
export HIP_PATH=/opt/rocm/...
export HIP_ROOT_DIR=/opt/rocm/...
export CMAKE_HIP_COMPILER=/opt/rocm/.../bin/hipcc
export CPATH=/opt/rocm/include        # clang JIT: thrust/complex.h
export LIBRARY_PATH=/opt/rocm/lib     # link: -lamdhip64
```

`CPATH` and `LIBRARY_PATH` are the ones that silently fail.
Without them: `fatal error: 'thrust/complex.h'` and `cannot find
-lamdhip64`.

## 5. After-tip-move cache wipe

When you `git pull` / merge / rebase on the dest tree, wipe:

```bash
rm -rf ~/.cache/vllm ~/.cache/torch_extensions ~/.cache/torch
rm -rf build/ .deps/                # editable-install build dir
rm -rf ~/.triton/cache              # Triton JIT cache

pip install -e . --no-build-isolation --no-deps
```

Old `.so` may embed a stale fatbin or registered-schema set; new
Python may have removed/renamed ops. Mixed old-`.so` + new-Python
= silent "schema not found" or wrong-arch fatbin load.

`PYTORCH_TUNABLEOP_FILENAME` must point at a persistent path so
the TunableOp cache survives the wipe. Without this, every
restart from a new CWD re-tunes every unseen shape (up to ~25s
per shape on V620).

## 6. All-reduce

RDNA consumer/professional GPUs have no XGMI; only PCIe. AR is
PCIe P2P, not NVLink-class.

| AR mode | gfx1030 | Why |
|---|---|---|
| Uncached staging (Coarse-grained `store`) | **dest** | avoids L2-pollution hot cache; halves stale traffic on captured replay |
| Finegrained (cache-coherent) | leave | no hardware L2-coherence over PCIe; software flush adds latency |
| NVLink / XGMI class | not present | don't ship a config that asks for it |
| LL / LL128 protocol | leave | deadlocks on PCIe — fix is `NCCL_PROTO=Simple` |
| MSCCL scheduler | default off | stream-hungry, conflicts with the 8-HQD budget |

### PYNCCL (PIX mode)

```bash
export NCCL_P2P_LEVEL=pix           # P2P when GPUs on same PCI switch
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple            # only safe protocol on PCIe
export RCCL_MSCCL_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1  # required for cross-device PCIe P2P
```

`HSA_FORCE_FINE_GRAIN_PCIE=1` forces HSA to treat PCIe memory as
fine-grained (cache-coherent from the host's view), which is what
the dest launcher needs for cross-device P2P and for some custom-AR
staging paths. Without it, AR traffic falls back to slower
DMA-buffer paths.

### Custom AR — dest default

```bash
export VLLM_FORCE_CUSTOM_ALL_REDUCE=1   # dest launcher default
export VLLM_RDNA_AR_MAX_KB=20480         # AR staging buffer cap (KB)
```

Beats PYNCCL by 30–40% at c=1 (small-batch decode where AR is the
critical path). The dest launcher enables custom AR by default;
the upstream vLLM env-var default is `False`, but the dest
launcher flips it to `True`.

On AR correctness issues (compare values across ranks in a probe),
disable with `VLLM_FORCE_CUSTOM_ALL_REDUCE=0` and fall back to
PYNCCL.

## 7. Queue budget — RDNA2 has 8 HQDs

RDNA2 has 8 user-accessible compute HQDs (vs 24 on CDNA). vLLM
TP=4 creates ~3 streams per worker (attention, MoE, matmul) plus
the RCCL proxy thread = 12–16 streams per process. Without
`GPU_MAX_HW_QUEUES`, the HIP default of 4 streams × 4 workers
oversubscribes the budget:

```
amdgpu: Runlist is getting oversubscribed due to too many queues
```

| `GPU_MAX_HW_QUEUES` | Behaviour |
|---|---|
| `1` | all work serialises, stable but slow at c≥4 |
| **`2`** | **dest default** — one compute queue + one AR/copy queue |
| `4` (HIP default) | oversubscription at c≥4, 10–30% drop |
| `8` | hardware limit per process, no headroom |

## 8. TunableOp — always-on, persistent cache

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0   # gfx1030 doesn't speak hipBLASLt
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=30
export PYTORCH_TUNABLEOP_FILENAME=/home/$USER/.cache/tunableop/tunableop_results.csv
```

TunableOp autotunes rocBLAS GEMM shapes at first encounter.
Tuning is per-shape; without a persistent cache, every restart
from a new CWD re-tunes every unseen shape.

The hipBLASLt disable is critical: gfx1030 doesn't speak hipBLASLt
cleanly (bundled lazy catalog is a symlink to gfx1100 that fails
at runtime). `TORCH_BLAS_PREFER_HIPBLASLT=0` is the related flag
for torch's regular linear path.

## 9. Multiprocess — `spawn`

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

`fork` re-uses the parent's CUDA context after the worker
subprocess is forked, racing with subsequent allocations. On
ROCm: `RuntimeError: Cannot re-initialize CUDA before forking`
or silent memory corruption.

## 10. Cache root paths

| Cache | Path |
|---|---|
| Triton | `/home/$USER/.triton/cache` (or `TRITON_CACHE_DIR`) |
| vLLM | `/home/$USER/.cache/vllm` (or `VLLM_CACHE_ROOT`) |
| torch compile | `/home/$USER/.cache/torch` |
| torch_extensions | `/home/$USER/.cache/torch_extensions` |
| TunableOp | `/home/$USER/.cache/tunableop/tunableop_results.csv` |

Fedora / RHEL with SELinux enforcing: `restorecon -R ~/.cache`
after creation.

## 11. `HSA_OVERRIDE_GFX_VERSION` — never

Not a fix. The runtime pretends the GPU is a different arch, but
kernels are compiled for the real arch — silent garbage or
`hipErrorInvalidImage` at load. The only legitimate use is
upstream Torch's "fake gfx900 on gfx906 for testing" CI scenario.
