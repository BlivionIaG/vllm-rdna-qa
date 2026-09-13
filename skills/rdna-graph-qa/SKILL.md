---
name: rdna-graph-qa
description: use this when debugging or reviewing cudagraph/HIP graph capture on gfx1030 dest.
---

# Graph QA

Wiki:
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

1. Arenas **before** `BeginCapture`. No `register_buffer`. No lazy
   `_ensure` (`6ffcdece` / `74f47b6`).
2. **zeros** + no `.item()` (`854dd56`). `Tensor!` outs. Blocking D2H
   stays out (`rdna_ar_timed_out`).
3. Persist hooks must be **called**; write the **CAPTURE** slot on
   replay (`f7761592`). Eager slot only for larger prefill. Registered
   but uncalled = dead wiring.
4. `VLLM_CG_NAN_INPUT_CHECK` default **off** (`b7c77f08`). Fold Gemma
   `(1+w)` so AOT rms fires (`55ff0044`).
5. Breakable CG is **admission**, not a capture fix.
6. GDN arenas: slot 0 = `NULL_BLOCK_ID`; `max_bs+1` zeroed before
   capture. `causal_conv1d_fwd` / `gated_rms` are Qwen GDN-only.
