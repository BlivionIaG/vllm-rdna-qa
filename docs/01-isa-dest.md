# ISA dest — gfx1030 DOT

Silicon contract for dest extras and hippihx dest fatbins. Full ISA
notes stay in the
[wiki silicon book](https://github.com/BlivionIaG/rdna-hip-wiki/tree/main/silicon).

## Inner ops (dest)

| Op | Role on gfx1030 | Not dest |
|---|---|---|
| **`fdot2`** / `v_dot2c` | W4A16, W8A16, FA, mxfp4-after-unpack, EXL3 after 3inst | `fdot2.bf16` |
| **`sdot4`** | W8A8 i8×i8 (Later until dest-signed) | hipBLASLt / CK XDL stand-in |

Dest inner loop is packed DOT. Scalar FMA is allowed for short-state
paths (e.g. `causal_conv`, `state_len≈4`), not as a GEMM dest.

## Wave32

V620 dest is **wave32**. DOT tiles do not ship a wave64 path.

- Deck gfx103x is also wave32 (portable, not dest-tuned).
- gfx1013 (BC-250) may report warp 64 — **never** force
  `-mwavefrontsize32` or load a gfx1030 object there.
- Serve bind keys on **arch + wave size**, not GFX name alone.

## One `--offload-arch` per fatbin

Shared DOT **source** is allowed (gfx1030 + gfx110x). Shared **objects**
are not.

- One CMake tree → one `libhippihx_<arch>.a`.
- Never `--offload-arch=gfx1030,gfx1100` in one artifact.
- gfx1100/1101/1102 are first-class DOT consumers of the same source,
  separate fatbin each. WMMA is a gfx110x-only Later overlay.

## No `HSA_OVERRIDE`

Never `HSA_OVERRIDE_GFX_VERSION`. Never load a foreign ISA (especially
never run gfx1030 objects on gfx1013 / Deck / gfx110x).

A fatbin that only runs under override is Leave, not dest.

## Leave: WMMA / MFMA / TMA

gfx1030 dest has **no** WMMA, **no** MFMA, **no** TMA, no FP8/FP4 unit,
no bf16 matrix.

| Feature | Why Leave on dest |
|---|---|
| WMMA | gfx11/12. `#ifdef WMMA` on shared DOT tiles is forbidden. |
| MFMA / XDL / AITER | CDNA. Not V620. |
| TMA / cp.async-class | NVIDIA / later GFX. Not dest. |

`q_gemm_rdna3_wmma.cu` and similar stay Later overlay. Do not gate dest
DOT on them.

## Specialist TODOs

- `TODO(silicon):` dest-signed list of builtins actually fired
  (`v_dot2c`, `v_dot4c`, waitcnt class). Cite wiki `valu.md` / occupancy
  dump; do not invent encodings here.
- `TODO(silicon):` CMake refuse list (multi-arch, gfx906 as dest, override
  hints) — confirm extras and hippihx still match.
- `TODO(engine):` serve fatbin probe: reject foreign ISA at load, not at
  first kernel launch.
