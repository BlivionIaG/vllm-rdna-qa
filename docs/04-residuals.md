# Residuals — three tracks

These are **three residuals**. Never collapse them into one “graph
bug” or one “TP bug.” Never land a **TP≤2 gate** or **Triton as dest**
to hide any of them.

| # | Residual | Not the same as |
|---|---|---|
| 1 | **mamba-split TP≤2 prefill** | W4A16 world size; APC |
| 2 | **in-graph W4A16 world>2** | mamba-split prefill; APC state |
| 3 | **APC state ≠ KV** | page-commit zeros; residual 1 or 2 |

A fix that “works at TP=2 eager” does not close 2 or 3. A Triton
fallback is harvest, not dest.

## 1 — mamba-split TP≤2 prefill

Mamba / GDN-class **split** prefill is residual at **TP≤2**. Prefill
chunk / wy / scan split across ranks is a different bug class from
decode capture.

- Do not “fix” this by refusing TP=2. Dest soak still has a TP=2 cell
  ([06-soak-matrix.md](06-soak-matrix.md)).
- Do not retarget GDN 16/48 onto KDA 64×128
  ([05-family-binds.md](05-family-binds.md)).
- GDN HIP prefill default-off (chunk-boundary corruption) is harvest
  status, not a collapse into residual 2.

`TODO(engine):` dest-signed repro: model, TP, chunk size, eager vs
graph. Split vs unsplit. What “green” means (numerics vs crash).
`TODO(silicon):` which HIP in the prefill chain is implicated
(`gdn_prefill_*`, `causal_conv` fwd). Cite wiki; no tok/s.

## 2 — in-graph W4A16 world>2

W4A16 **inside the captured graph** at **world size > 2** (dest TP=4
cell) is a separate residual.

- Collectives + captured W4 decode/prefill is the surface, not
  mamba-split.
- leapdragon `rdna_ar` stays opt-in (`VLLM_RDNA_AR=0` dest default).
  Do not use AR-off as a dest gate that hides world>2.
- Do not land “dest = TP≤2” to skip this cell.

`TODO(engine):` dest-signed repro: Qwen3.8-27B AWQ INT4 (or the soak
reference), TP=4, graph on, W4 in-graph. Eager-vs-graph delta.
`TODO(silicon):` whether the fault is GEMM capture, split-K partials,
or AR/RCCL. Keep it off residual 1.

## 3 — APC state ≠ KV

APC (automatic / prefix / piecewise cache — dest name as used on
extras) **state is not KV**. SSM / conv / GDN slot state must not be
treated as paged KV.

- Committing KV pages does not commit APC state pages.
- Graph QA zeros apply to state tensors too, but closing zeros does
  **not** close this residual.
- `NULL_BLOCK_ID` / invalid-slot → zero out, do not touch state. That
  sentinel is serve, not a zoo constant.

`TODO(engine):` dest-signed definition of APC vs KV vs SSM slot.
Warmup that writes KV only is **not** a pass.
`TODO(engine):` piecewise / hybrid GDN + cudagraph: eager correct ≠
capture correct. Keep this on residual 3 unless silicon proves it is
only page-commit (then it moves to [03-graph-qa.md](03-graph-qa.md),
not into residual 1 or 2).

## Never

- Never collapse 1+2+3 into one ticket, one env, or one Triton path.
- Never land **TP≤2 as dest**. TP=2 is a soak cell, not the product
  ceiling.
- Never land **Triton as dest**. One HIP / hippihx V1 consume path.
  Triton may remain harvest / debug.

`TODO(engine):` project-4 (or dest issue) IDs for each residual, once
filed. Three rows, three IDs.
