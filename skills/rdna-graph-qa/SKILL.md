---
name: rdna-graph-qa
description: use this when debugging or reviewing cudagraph/HIP graph capture on gfx1030 dest.
---

# Graph QA

Wiki:
[graph-capture.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/silicon/graph-capture.md).

1. **Zeros, not empty.** Dest: `zeros` for any buffer a captured
   kernel reads.
2. **No `.item()` / D2H under capture.** Gate probes with
   `is_current_stream_capturing()`. Keep blocking D2H out
   (`rdna_ar_timed_out`, NaN `.item()` scans).
   `VLLM_CG_NAN_INPUT_CHECK` default **off**.
3. **Arenas + persist CAPTURE before `BeginCapture`.** `bind` never
   allocates. Persist **writes the CAPTURE slot on replay** if the
   shape fits; eager slot only for larger prefill so the frozen ptr
   is never recycled. Hooks registered but never called = dead
   wiring.
4. **`Tensor!` outs.** Plain `Tensor` can drop the write.
5. **Hard-fail leftover `_ensure` under capture.** Do not probe.
6. **Breakable CG is admission** (poison ops leave the graph), not a
   capture fix.
7. GDN arenas: slot 0 = `NULL_BLOCK_ID`; allocate `max_bs+1` and
   zero before `BeginCapture`. `causal_conv1d_fwd` / `gated_rms`
   stay **Qwen GDN-only**.
