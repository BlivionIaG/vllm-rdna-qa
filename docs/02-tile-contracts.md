# Tile contracts

Locks that a dest tile must state **before** it is dest-signed. Numbers
live on hippihx tile READMEs and wiki occupancy pages. This chapter is
the QA gate, not the dump.

Zoo: [`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx)
`tiles/`. Wiki:
[lds-tiles](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/lds-tiles.md),
[w4a16-prefill-config](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/w4a16-prefill-config.md).

## LDS ≤ 64 KiB per workgroup

gfx1030: **64 KiB LDS / WG**, 128 KiB / WGP. Dest tiles must stay
**≤ 64 KiB**. Prefer a documented budget (FA prefill ~48 KiB class,
GDN prefill `o` ~45 KiB class) plus a pad/guard.

A tile that needs >64 KiB is Leave or a split, not dest.

`TODO(silicon):` fill per-tile LDS bytes from extras observations
(FA decode/prefill, GDN, W4 A-tile + `LDS_PAD`, EXL3, KDA Later). Do
not invent.

## 64 × 4 B banks

LDS is **64 banks × 4 bytes**. Bank conflicts are a dest bug class, not
a “tune later.”

- Pad formulas belong on the tile README (`LDS_PAD=8` on W4 A-tile is
  the existing harvest observation).
- Do not “fix” conflicts by jumping `K_STEP` past the packed DOT unit
  (see ConfigH below).

`TODO(silicon):` bank-conflict checklist per dest tile (A/B staging,
softmax scratch, scan state). Wiki formula, dest verdict.

## `__launch_bounds__`

Every dest HIP entry states `__launch_bounds__` (threads, `waves_per_eu`)
**or** an explicit “VGPR is the lever, no bounds” lock (EXL3 DOT harvest
note). Occupancy leftover stays FA-first.

Spill or a bad ISel is a drop, not dest.

`TODO(silicon):` record dest `__launch_bounds__` / VGPR / `waves_per_eu`
from `hipcc -Rpass-analysis=kernel-resource-usage` or NT_AMDGPU_METADATA.
Cite wiki occupancy pages.

## `K_STEP` must match packed DOT (ConfigH lesson)

Packed DOT has a native K grain. **`K_STEP` must match that grain** (and
group-size refresh). Dest large-M AWQ/GPTQ prefill is **ConfigA**:
`K_STEP=32`, `THREADS=256`, `N_TILE=1024`, `M_TILE=16`, `LDS=0`.

**ConfigH** (`K_STEP=64`) is **Leave**. The half-group / mid-block
refresh bug class is why: inner loops use `K_STEP/8`; group boundaries
must refresh per-j (`k + 8*j == nextgroup`). Doubling `K_STEP` without
matching packed DOT + groupsize produced wrong numerics and a dead
name.

Do not reintroduce `K_STEP=64` dispatch until fat-M numerics match
ConfigA. ConfigA_Large / ConfigP are also Leave (deleted on dest tip
after measuring slower / spill). See wiki `w4a16-prefill-config.md`.

| Config | `K_STEP` | Dest? |
|---|---|---|
| ConfigA | 32 | **Take** (large-M prefill) |
| ConfigH | 64 | **Leave** |
| ConfigA_Large / ConfigP | 32 / prefetch | **Leave** |

`TODO(silicon):` assert ConfigA `K_STEP` == packed `fdot2` consume grain
on the live dest object. Link hippihx `gemm/w4a16_fdot2`.
`TODO(engine):` keep `VLLM_RDNA2_PREFILL_FORCE_CONFIG` as bisection
only; default remains natural dispatch.
